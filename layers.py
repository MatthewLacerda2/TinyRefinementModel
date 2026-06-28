import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from typing import Dict, Any

from config import MAX_SEQ_LEN, MAX_STEPS_LIMIT, SHARED_SLOTS, NUM_GROUPS, COMPUTE_DTYPE

@struct.dataclass
class ScanStepOutput:
    shared_state: jnp.ndarray
    forget_val: jnp.ndarray
    step_div: jnp.ndarray

@struct.dataclass
class ReasonerOutput:
    logits: jnp.ndarray
    forget_cost: float
    diversity_loss: float
    diag: Dict[str, Any]
    final_shared: jnp.ndarray
    # Pre-head states [b, s, d], set instead of `logits` when training: the loss
    # projects the LM head per-chunk (chunked CE, #19) to avoid the full
    # [b, s, vocab] f32 logit peak. None at inference, where `logits` is filled.
    hidden: jnp.ndarray = None

def apply_rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    x_complex = jax.lax.complex(x1, x2)
    rope_complex = jax.lax.complex(cos, sin)
    rotated = x_complex * rope_complex
    
    return jnp.concatenate([rotated.real, rotated.imag], axis=-1).astype(x.dtype)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=4, rngs=None):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        # Extended to cover MAX_SEQ_LEN + MAX_STEPS_LIMIT * SHARED_SLOTS positions
        # so that slot query positions can advance with each reasoning iteration
        t = jnp.arange(MAX_SEQ_LEN + MAX_STEPS_LIMIT * SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=COMPUTE_DTYPE)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=COMPUTE_DTYPE)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=COMPUTE_DTYPE)

        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)

        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=COMPUTE_DTYPE)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, is_causal=True):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        kv_input = context if context is not None else x
        
        s_kv = kv_input.shape[1]

        k = self.k_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)
        v = self.v_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if q_pos is None:
            q_pos = jnp.arange(s)
        if kv_pos is None:
            kv_pos = jnp.arange(s_kv)

        cos_q = self.cos_cached[q_pos, None, :]
        sin_q = self.sin_cached[q_pos, None, :]
        q = apply_rope(q, cos_q, sin_q)

        cos_kv = self.cos_cached[kv_pos, None, :]
        sin_kv = self.sin_cached[kv_pos, None, :]
        k = apply_rope(k, cos_kv, sin_kv)

        if self.num_heads != self.num_groups:
            repeats = self.num_heads // self.num_groups
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Causality is folded into the explicit mask (positions are not plain
        # 0..N here, so dot_product_attention's built-in is_causal cannot be used).
        if is_causal:
            pos_mask = q_pos[:, None] >= kv_pos[None, :]

            if mask is not None:
                if mask.dtype == jnp.bool_:
                    mask = mask & pos_mask
                else:
                    mask = mask + (pos_mask.astype(jnp.float32) - 1.0) * 1e9
            else:
                mask = pos_mask

        q = q.astype(COMPUTE_DTYPE)
        k = k.astype(COMPUTE_DTYPE)
        v = v.astype(COMPUTE_DTYPE)

        if mask is not None and mask.dtype != jnp.bool_:
            attn_bias = mask.astype(COMPUTE_DTYPE)
            mask_arg = None
        else:
            attn_bias = None
            mask_arg = mask

        # dot_product_attention scales the scores internally — never pre-scale q
        # (doing both blurred every attention layer until 2026-06-11).
        out = jax.nn.dot_product_attention(
            q, k, v,
            mask=mask_arg,
            bias=attn_bias,
        )
        return self.o_proj(out.reshape(b, s, d))

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=NUM_GROUPS, rngs=rngs)
        self.norm1 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        hidden_dim = int(256 * ((latent_dim * 8 / 3 + 255) // 256))
        self.gate_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=COMPUTE_DTYPE)
        self.up_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=COMPUTE_DTYPE)
        self.down_proj = nnx.Linear(
            hidden_dim, latent_dim,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs, dtype=COMPUTE_DTYPE,
        )

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, is_causal=True):
        normed_context = self.norm1(context) if context is not None else None
        attn_out = self.attn(self.norm1(x), context=normed_context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, is_causal=is_causal)
        x = x + attn_out

        mlp_in = self.norm2(x)

        hidden = jax.nn.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in)
        x = x + self.down_proj(hidden)
        return x

class BlockStack(nnx.Module):
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.float32, share_weights=False, use_remat=True):
        self.num_blocks = num_blocks
        self.share_weights = share_weights
        # Per-block remat: recompute each block's intermediates during the backward
        # pass instead of storing them. Benchmarked 2026-06-10: it saved no measurable
        # VRAM here (the reasoning stack is already memory-bounded by the scan-level
        # jax.checkpoint in model._reasoning_loop) and cost ~18% step time, so all
        # stacks now pass use_remat=False. The machinery stays for future configs
        # that might actually be memory-bound (larger dims, real batching).
        self.use_remat = use_remat
        if share_weights:
            self.blocks = nnx.List([
                StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
            ])
        else:
            self.blocks = nnx.List([
                StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
                for _ in range(num_blocks)
            ])

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, is_causal=True, training=False):
        def checkpoint_block(block, x_val, ctx_val, mask_val, q_val, kv_val):
            block_graph, block_state = nnx.split(block)

            @jax.checkpoint
            def _pure_block_fn(state, x_in, ctx_in, mask_in, q_in, kv_in):
                blk = nnx.merge(block_graph, state)
                out_x = blk(
                    x_in, context=ctx_val if ctx_in is None else ctx_in, mask=mask_in,
                    q_pos=q_in, kv_pos=kv_in,
                    is_causal=is_causal
                )
                _, out_state = nnx.split(blk)
                return out_x, out_state

            x_out, new_block_state = _pure_block_fn(block_state, x_val, ctx_val, mask_val, q_val, kv_val)
            nnx.update(block, new_block_state)
            return x_out

        apply_remat = training and self.use_remat

        if self.share_weights:
            block = self.blocks[0]
            for _ in range(self.num_blocks):
                if apply_remat:
                    x = checkpoint_block(block, x, context, mask, q_pos, kv_pos)
                else:
                    x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, is_causal=is_causal)
        else:
            for block in self.blocks:
                if apply_remat:
                    x = checkpoint_block(block, x, context, mask, q_pos, kv_pos)
                else:
                    x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, is_causal=is_causal)
        return x


def calculate_slot_stability_loss(new_shared, curr_shared, tau):
    """
    InfoNCE-based slot stability loss.

    For each slot i in new_shared, the matching slot i in curr_shared (stop-gradient)
    is the positive, and all other slots in curr_shared are negatives.

    Effect: each slot is pushed to be more similar to its own previous state than to
    any other slot's previous state — enforcing stable slot identity across reasoning
    steps (specialization), not diversity within a step.
    """
    b, s, d = new_shared.shape
    
    anchor = new_shared / jnp.sqrt(jnp.sum(jnp.square(new_shared), axis=-1, keepdims=True) + 1e-5)
    positive = jax.lax.stop_gradient(
        curr_shared / jnp.sqrt(jnp.sum(jnp.square(curr_shared), axis=-1, keepdims=True) + 1e-5)
    )
    
    pos_logits = jnp.sum(anchor * positive, axis=-1, keepdims=True) / tau
    
    neg_logits = jnp.einsum('bsd,btd->bst', anchor, positive, precision=jax.lax.Precision.HIGHEST) / tau
    
    identity = jnp.eye(s)[None, :, :]
    neg_logits = neg_logits + (identity * -1e9)
    logits = jnp.concatenate([pos_logits, neg_logits], axis=-1)
    
    labels = jnp.zeros((b, s), dtype=jnp.int32)
    
    loss_per_slot = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    
    return jnp.mean(loss_per_slot, axis=-1)
