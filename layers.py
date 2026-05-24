import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from typing import Dict, Any

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
    halt_diag: Dict[str, Any]
    expected_shared: jnp.ndarray

# Keep (most) values powers of 2 if you know what's good for you

# Params
LATENT_DIM = 1024
NUM_BLOCKS = 8
SHARED_SLOTS = 32
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100352

# Training
MAX_STEPS_LIMIT = 8
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100257

NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4

def apply_rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    x_complex = jax.lax.complex(x1, x2)
    rope_complex = jax.lax.complex(cos, sin)
    rotated = x_complex * rope_complex
    
    return jnp.concatenate([rotated.real, rotated.imag], axis=-1).astype(x.dtype)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=4, rngs=None, dtype=jnp.float32):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)

        self.k_cache = nnx.Cache(None)
        self.v_cache = nnx.Cache(None)
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float16)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float16)

        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)

        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)

    def reset_state(self):
        self.k_cache.value = None
        self.v_cache.value = None
        self.cache_index.value = jnp.zeros_like(self.cache_index.value)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=True):
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
        q = q * self.scale

        cos_kv = self.cos_cached[kv_pos, None, :]
        sin_kv = self.sin_cached[kv_pos, None, :]
        k = apply_rope(k, cos_kv, sin_kv)

        if use_cache:
            if self.k_cache.value is None:
                cache_shape = (b, MAX_SEQ_LEN + SHARED_SLOTS, self.num_groups, self.head_dim)
                self.k_cache.value = jnp.zeros(cache_shape, dtype=x.dtype)
                self.v_cache.value = jnp.zeros(cache_shape, dtype=x.dtype)
            
            idx = self.cache_index.value
            k_cache = jax.lax.dynamic_update_slice(self.k_cache.value, k, (0, idx, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(self.v_cache.value, v, (0, idx, 0, 0))
            self.k_cache.value = k_cache
            self.v_cache.value = v_cache
            new_idx = idx + s_kv
            self.cache_index.value = new_idx
            
            k = k_cache[:, :new_idx, :, :]
            v = v_cache[:, :new_idx, :, :]
        if self.num_heads != self.num_groups:
            repeats = self.num_heads // self.num_groups
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        if is_causal:
            pos_mask = q_pos[:, None] >= kv_pos[None, :]
            
            if mask is not None:
                if mask.dtype == jnp.bool_:
                    mask = mask & pos_mask
                else:
                    mask = mask + (pos_mask.astype(jnp.float32) - 1.0) * 1e9
            else:
                mask = pos_mask
            
            effective_is_causal = False
        else:
            effective_is_causal = False
            
        q = q.astype(jnp.float16)
        k = k.astype(jnp.float16)
        v = v.astype(jnp.float16)
        
        if mask is not None and mask.dtype != jnp.bool_:
            attn_bias = mask.astype(jnp.float16)
            mask_arg = None
        else:
            attn_bias = None
            mask_arg = mask

        out = jax.nn.dot_product_attention(
            q, k, v,
            mask=mask_arg, 
            bias=attn_bias,
            is_causal=effective_is_causal,
        )
        return self.o_proj(out.reshape(b, s, d))

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=NUM_GROUPS, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        hidden_dim = int(256 * ((latent_dim * 8 / 3 + 255) // 256))
        self.gate_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=jnp.float16)
        self.up_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=jnp.float16)
        self.down_proj = nnx.Linear(
            hidden_dim, latent_dim,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs, dtype=jnp.float16,
        )

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=True):
        normed_context = self.norm1(context) if context is not None else None
        attn_out = self.attn(self.norm1(x), context=normed_context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
        x = x + attn_out

        mlp_in = self.norm2(x)

        hidden = jax.nn.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in)
        x = x + self.down_proj(hidden)
        return x

class BlockStack(nnx.Module):
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.float32, share_weights=False):
        self.num_blocks = num_blocks
        self.share_weights = share_weights
        if share_weights:
            self.blocks = nnx.List([
                StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
            ])
        else:
            self.blocks = nnx.List([
                StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
                for _ in range(num_blocks)
            ])

    def reset_state(self):
        for block in self.blocks:
            block.attn.reset_state()

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=True, training=False):
        def checkpoint_block(block, x_val, ctx_val, mask_val, q_val, kv_val):
            block_graph, block_state = nnx.split(block)
            
            @jax.checkpoint
            def _pure_block_fn(state, x_in, ctx_in, mask_in, q_in, kv_in):
                blk = nnx.merge(block_graph, state)
                out_x = blk(
                    x_in, context=ctx_val if ctx_in is None else ctx_in, mask=mask_in, 
                    q_pos=q_in, kv_pos=kv_in, 
                    use_cache=use_cache, is_causal=is_causal
                )
                _, out_state = nnx.split(blk)
                return out_x, out_state
            
            x_out, new_block_state = _pure_block_fn(block_state, x_val, ctx_val, mask_val, q_val, kv_val)
            nnx.update(block, new_block_state)
            return x_out

        if self.share_weights:
            block = self.blocks[0]
            for _ in range(self.num_blocks):
                if training:
                    x = checkpoint_block(block, x, context, mask, q_pos, kv_pos)
                else:
                    x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
        else:
            for block in self.blocks:
                if training:
                    x = checkpoint_block(block, x, context, mask, q_pos, kv_pos)
                else:
                    x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
        return x

def calculate_infonce_loss(new_shared, curr_shared, tau):
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

