"""Plan A — causal within-window depth recurrence (Universal-Transformer style).

A shared transformer block is looped K times over the token representations under
a causal mask, then the tied LM head reads the refined state. Position t attends
only to positions <= t on every iteration, so depth refines the prediction with no
future-token leak (contrast docs/findings/2026-06-13-cross-window-hunch-inert.md,
where the loop could only reach the next window and the gradient killed it).

Self-contained and fully parametrized (dim, vocab, heads, depth, seq) so the exact
same architecture runs at tiny toy-task scale and at real scale — the ablation
harness tests the thing we'd ship, not a stand-in. See docs/design/plan-a.md.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _rope_tables(max_pos, head_dim):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
    freqs = jnp.outer(jnp.arange(max_pos), inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jax.lax.complex(x1, x2) * jax.lax.complex(cos, sin)
    return jnp.concatenate([rotated.real, rotated.imag], axis=-1).astype(x.dtype)


class CausalAttention(nnx.Module):
    """Multi-head self-attention, RoPE, causal mask folded into an additive bias."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32):
        assert dim % num_heads == 0, "dim must divide num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.k = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.v = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.o = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        cos, sin = _rope_tables(max_pos, self.head_dim)
        self.cos, self.sin = cos, sin

    def __call__(self, x, pad_bias=None):
        b, s, d = x.shape
        q = self.q_norm(self.q(x).reshape(b, s, self.num_heads, self.head_dim))
        k = self.k_norm(self.k(x).reshape(b, s, self.num_heads, self.head_dim))
        v = self.v(x).reshape(b, s, self.num_heads, self.head_dim)

        cos = self.cos[:s, None, :]
        sin = self.sin[:s, None, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        pos = jnp.arange(s)
        causal = pos[:, None] >= pos[None, :]                       # [s, s], True = allowed
        bias = jnp.where(causal, 0.0, -1e9)[None, None, :, :]       # [1, 1, s, s]
        if pad_bias is not None:
            bias = bias + pad_bias                                  # pad_bias [b, 1, 1, s]

        out = jax.nn.dot_product_attention(q, k, v, bias=bias.astype(x.dtype))
        return self.o(out.reshape(b, s, d))


class Block(nnx.Module):
    """Pre-norm transformer block: causal attention + SwiGLU MLP, zero-init residual."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32):
        self.attn = CausalAttention(dim, num_heads, max_pos, rngs, dtype)
        self.norm1 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        hidden = ((int(8 * dim / 3) + 63) // 64) * 64
        self.gate_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.up_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.down_proj = nnx.Linear(hidden, dim, kernel_init=jax.nn.initializers.zeros, rngs=rngs, dtype=dtype)

    def __call__(self, x, pad_bias=None):
        x = x + self.attn(self.norm1(x), pad_bias)
        h = self.norm2(x)
        x = x + self.down_proj(jax.nn.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class CausalRefiner(nnx.Module):
    """embed -> causal encoder -> (shared block looped K times) -> norm -> tied head.

    `depth` is the number of refinement iterations: sampled per step in training,
    fixed at inference. It is a static argument to __call__ so the loop unrolls and
    compiles cleanly (depth is small, <= max_depth).
    """

    def __init__(self, *, dim, vocab_size, num_heads=4, num_encoder_layers=2,
                 max_depth=8, max_seq_len=512, use_gate=True, gate_bias=0.0, rngs, dtype=jnp.float32):
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.use_gate = use_gate
        self.dtype = dtype

        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs, dtype=dtype)
        self.time_embed = nnx.Embed(max_depth + 1, dim, rngs=rngs, dtype=dtype)

        self.encoder = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype) for _ in range(num_encoder_layers)
        ])
        self.refine_block = Block(dim, num_heads, max_seq_len, rngs, dtype)  # shared, looped

        self.time_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.out_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        if use_gate:
            # gate_bias sets the init retention/refine balance: sigmoid(gate_bias).
            # 0.0 -> 0.5 (balanced); negative -> retention-biased (small early update
            # steps), which stabilizes deep recurrence where balanced steps compound
            # and diverge (the depth-8 collapse, ablation_results.md run 2).
            self.gate = nnx.Linear(2 * dim, dim, bias_init=jax.nn.initializers.constant(gate_bias), rngs=rngs, dtype=dtype)

    def __call__(self, tokens, depth=None, pad_mask=None):
        depth = self.max_depth if depth is None else depth

        pad_bias = None
        if pad_mask is not None:
            pad_bias = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
            pad_bias = pad_bias[:, None, None, :]

        z = self.embed(tokens)
        for blk in self.encoder:
            z = blk(z, pad_bias)

        for step in range(depth):
            t_signal = self.time_embed(jnp.asarray(step))
            z_in = self.time_norm(z) + self.time_signal_norm(t_signal)[None, None, :]
            z_new = self.refine_block(z_in, pad_bias)
            if self.use_gate:
                g = jax.nn.sigmoid(self.gate(jnp.concatenate([z_new, z], axis=-1)))
                z = g * z_new + (1.0 - g) * z
            else:
                z = z_new

        z = self.out_norm(z)
        embed_t = self.embed.embedding.value.astype(self.dtype).T
        logits = jnp.matmul(z.astype(self.dtype), embed_t, preferred_element_type=jnp.float32)
        return logits
