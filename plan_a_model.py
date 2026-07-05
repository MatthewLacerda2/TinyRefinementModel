"""Plan A — causal within-window depth recurrence (Universal-Transformer style).

A shared transformer block is looped K times over the token representations under
a causal mask, then the tied LM head reads the refined state. Position t attends
only to positions <= t on every iteration, so depth refines the prediction with no
future-token leak (contrast docs/findings/2026-06-13-cross-window-hunch-inert.md,
where the loop could only reach the next window and the gradient killed it).

Config-free and fully parametrized (dim, vocab, heads, depth, seq) so the exact
same architecture runs at tiny toy-task scale and at real scale — the ablation
harness tests the thing we'd ship, not a stand-in. See docs/design/plan-a.md.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from attention import chunked_causal_attention
from rope import rope_tables, apply_rope


class CausalAttention(nnx.Module):
    """Multi-head self-attention, RoPE, causal mask folded into an additive bias.

    chunked=True swaps the stock dot_product_attention for the blockwise
    memory-lean path (attention.py, #66): same math up to float summation
    order, but no [s, s] score/probability tensor is ever materialized or
    saved for the backward — the O(seq²) activation wall goes away."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32, chunked=False):
        assert dim % num_heads == 0, "dim must divide num_heads"
        self.num_heads = num_heads
        self.chunked = chunked
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.k = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.v = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.o = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        cos, sin = rope_tables(max_pos, self.head_dim)
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

        # q_norm/k_norm run in f32 for stability, so q/k come out f32 while v is in
        # the compute dtype. Cast q/k back so all three match (dot_product_attention
        # requires it) and attention takes the tensor-core path. No-op in f32 (CPU /
        # toy harness); the real-scale f16 run needs it.
        q = q.astype(x.dtype)
        k = k.astype(x.dtype)

        if self.chunked:
            # Blockwise path (#66): causal mask built per query block inside the
            # scan; only the key-padding bias is passed, as additive [b, s].
            pad_cols = (pad_bias[:, 0, 0, :] if pad_bias is not None
                        else jnp.zeros((b, s), jnp.float32))
            out = chunked_causal_attention(q, k, v, pad_cols)
        else:
            pos = jnp.arange(s)
            causal = pos[:, None] >= pos[None, :]                   # [s, s], True = allowed
            bias = jnp.where(causal, 0.0, -1e9)[None, None, :, :]   # [1, 1, s, s]
            if pad_bias is not None:
                bias = bias + pad_bias                              # pad_bias [b, 1, 1, s]
            out = jax.nn.dot_product_attention(q, k, v, bias=bias.astype(x.dtype))
        return self.o(out.reshape(b, s, d))


class Block(nnx.Module):
    """Pre-norm transformer block: causal attention + SwiGLU MLP, zero-init residual."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32, chunked_attention=False):
        self.attn = CausalAttention(dim, num_heads, max_pos, rngs, dtype, chunked=chunked_attention)
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
                 max_depth=8, max_seq_len=512, use_gate=True, gate_bias=0.0,
                 chunked_attention=False, rngs, dtype=jnp.float32):
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.use_gate = use_gate
        self.dtype = dtype

        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs, dtype=dtype)
        self.time_embed = nnx.Embed(max_depth + 1, dim, rngs=rngs, dtype=dtype)

        self.encoder = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype, chunked_attention=chunked_attention)
            for _ in range(num_encoder_layers)
        ])
        self.refine_block = Block(dim, num_heads, max_seq_len, rngs, dtype,
                                  chunked_attention=chunked_attention)  # shared, looped

        self.time_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.out_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        if use_gate:
            # gate_bias sets the init retention/refine balance: sigmoid(gate_bias).
            # 0.0 -> 0.5 (balanced); negative -> retention-biased (small early update
            # steps), which stabilizes deep recurrence where balanced steps compound
            # and diverge (the depth-8 collapse, ablation_results.md run 2).
            self.gate = nnx.Linear(2 * dim, dim, bias_init=jax.nn.initializers.constant(gate_bias), rngs=rngs, dtype=dtype)

    def __call__(self, tokens, depth=None, pad_mask=None, return_hidden=False,
                 grad_last=None, islands=False, return_all_iters=False):
        depth = self.max_depth if depth is None else depth

        pad_bias = None
        if pad_mask is not None:
            pad_bias = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
            pad_bias = pad_bias[:, None, None, :]

        z = self.embed(tokens)
        for blk in self.encoder:
            z = blk(z, pad_bias)

        # Truncated backprop through the refinement depth (#64): with grad_last=j,
        # gradient flows through only the last j refinement iterations — the
        # trajectory before the cut runs forward-only, so its activations need not
        # be kept for the backward (~O(1) activation memory in depth instead of
        # O(depth)). The cut detaches only the loop state's *deviation* from the
        # encoder output; the encoder itself keeps an identity gradient path, since
        # a full detach would leave it with no training signal at all.
        assert grad_last is None or grad_last >= 1, "grad_last must be >= 1 (or None for full backprop)"
        assert not (islands and grad_last is not None), "islands and grad_last are different cuts — pick one"
        z_enc = z
        cut = None if grad_last is None else depth - grad_last

        # islands (#75): cut the trajectory gradient at EVERY pass boundary, so
        # each iteration is graded only by its own loss (pair with per-pass
        # supervision — with a final-only loss this is just grad_last=1). Same
        # deviation-only detach as grad_last: the encoder keeps its identity path.
        # return_all_iters (#75): also return every pass's logits (toy scale only
        # — [depth, b, s, vocab] is cheap here, unaffordable at real vocab) plus
        # each pass's mean gate openness, for per-pass supervision and readouts.
        all_z, gate_means = [], []
        for step in range(depth):
            if islands and step > 0:
                z = z_enc + jax.lax.stop_gradient(z - z_enc)
            elif cut is not None and step == cut and cut > 0:
                z = z_enc + jax.lax.stop_gradient(z - z_enc)
            t_signal = self.time_embed(jnp.asarray(step))
            z_in = self.time_norm(z) + self.time_signal_norm(t_signal)[None, None, :]
            z_new = self.refine_block(z_in, pad_bias)
            if self.use_gate:
                g = jax.nn.sigmoid(self.gate(jnp.concatenate([z_new, z], axis=-1)))
                z = g * z_new + (1.0 - g) * z
                gate_means.append(jnp.mean(g))
            else:
                z = z_new
            if return_all_iters:
                all_z.append(z)

        if return_all_iters:
            z_all = self.out_norm(jnp.stack(all_z))  # [depth, b, s, dim]
            embed_t = self.embed.embedding[...].astype(self.dtype).T
            logits_all = jnp.matmul(z_all.astype(self.dtype), embed_t,
                                    preferred_element_type=jnp.float32)
            gates = jnp.stack(gate_means) if self.use_gate else None
            return logits_all, gates

        z = self.out_norm(z)
        if return_hidden:
            # Training path: let the loss project + score the LM head per-chunk
            # (chunked CE, #19) instead of materializing full [b, s, vocab] logits.
            return z.astype(self.dtype)
        embed_t = self.embed.embedding[...].astype(self.dtype).T
        logits = jnp.matmul(z.astype(self.dtype), embed_t, preferred_element_type=jnp.float32)
        return logits
