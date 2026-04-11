import jax
import optax
from flax import nnx
import jax.numpy as jnp
from schedulers import (
    ponder_lambda_schedule,
    forget_lambda_schedule,
    diversity_lambda_schedule
)

NUM_BLOCKS = 4
LATENT_DIM = 512
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
MIN_STEPS = 4
MAX_STEPS_LIMIT = 16
SHARED_SLOTS = 32
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100352
PAD_TOKEN_ID = 100257
HUNCH_REFRESH_EVERY = 4
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4


def apply_rope(x, sin_table, cos_table):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos_table - x2 * sin_table
    out2 = x1 * sin_table + x2 * cos_table
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape).astype(x.dtype)

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

        self.k_cache = nnx.Cache(jnp.zeros((BATCH_SIZE, MAX_SEQ_LEN + SHARED_SLOTS, self.num_groups, self.head_dim), dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros((BATCH_SIZE, MAX_SEQ_LEN + SHARED_SLOTS, self.num_groups, self.head_dim), dtype=dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        kv_input = context if context is not None else x
        s_kv = kv_input.shape[1]

        k = self.k_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)
        v = self.v_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)

        if q_pos is None:
            q_pos = jnp.arange(s)
        if kv_pos is None:
            kv_pos = jnp.arange(s_kv)

        sin_q = self.sin_cached[q_pos, None, :]
        cos_q = self.cos_cached[q_pos, None, :]
        q = apply_rope(q, sin_q, cos_q)
        q = q * self.scale

        sin_kv = self.sin_cached[kv_pos, None, :]
        cos_kv = self.cos_cached[kv_pos, None, :]
        k = apply_rope(k, sin_kv, cos_kv)

        if use_cache:
            idx = self.cache_index.value
            k_cache = jax.lax.dynamic_update_slice(self.k_cache.value, k, (0, idx, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(self.v_cache.value, v, (0, idx, 0, 0))
            self.k_cache.value = k_cache
            self.v_cache.value = v_cache
            self.cache_index.value = idx + s_kv
            k = k_cache
            v = v_cache

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.repeat(k, repeats, axis=2)
        v_expanded = jnp.repeat(v, repeats, axis=2)

        out = jax.nn.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
            if mask is not None else None,
        )
        return self.o_proj(out.reshape(b, s, d))


class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=2, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)

        hidden_dim = int(256 * ((latent_dim * 8 / 3 + 255) // 256))
        self.gate_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=dtype)
        self.up_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=dtype)
        self.down_proj = nnx.Linear(
            hidden_dim, latent_dim,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs, dtype=dtype,
        )

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        normed_context = self.norm1(context) if context is not None else None
        attn_out = self.attn(self.norm1(x), context=normed_context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache)
        x = x + attn_out

        mlp_in = self.norm2(x)

        hidden = jax.nn.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in)
        x = x + self.down_proj(hidden)
        return x


class BlockStack(nnx.Module):
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.blocks = nnx.List([
            StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
            for _ in range(num_blocks)
        ])
        self.num_blocks = num_blocks

    def reset_state(self):
        for block in self.blocks:
            block.attn.cache_index.value = jnp.array(0, dtype=jnp.int32)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        for block in self.blocks:
            x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache)
        return x


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), jnp.float32)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.main_stack = BlockStack(num_blocks, latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

        halt_pre_dim = latent_dim // 4
        self.halt_pre = nnx.Linear(latent_dim, halt_pre_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(halt_pre_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), -1.0) 
        
        self.time_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.forget_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)

        self.hunch_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.hunch_gate = nnx.Linear(
            latent_dim, latent_dim,
            bias_init=jax.nn.initializers.constant(-2.0),
            rngs=rngs, dtype=dtype,
        )

        self.use_forget = use_forget
        if self.use_forget:
            self.forget_head = nnx.Linear(latent_dim, latent_dim, bias_init=jax.nn.initializers.constant(3.0), rngs=rngs, dtype=dtype)

        self.hunch_cache = nnx.Cache(None)


    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True, prev_hunch=None):
        batch_size, seq_len = tokens.shape
        
        current_hunch = None
        if training:
            current_hunch = prev_hunch
            self.main_stack.reset_state()
        else:
            self.main_stack.reset_state()
            if not should_refresh and self.hunch_cache.value is not None:
                current_hunch = self.hunch_cache.value
                
        seq_pos, shared_pos = jnp.arange(seq_len), jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        pad_mask = tokens != PAD_TOKEN_ID
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        seq_attn_mask = pad_mask[:, None, None, :] & causal_mask[None, None, :, :]

        z_seq = self.embed(tokens)
        z_seq = self.main_stack(z_seq, mask=seq_attn_mask, q_pos=seq_pos, kv_pos=seq_pos)

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))

        if current_hunch is not None:
            gate = jax.nn.sigmoid(self.hunch_gate(self.hunch_norm(current_hunch)))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        all_time_embeds = self.time_embed(jnp.arange(max_steps))
        
        # 1. Sequence part: all slots can see all valid sequence tokens
        seq_mask = jnp.broadcast_to(
            pad_mask[:, None, None, :], 
            (batch_size, 1, SHARED_SLOTS, seq_len)
        )
        
        # 2. Shared part: Causal mask so Slot N can only attend to Slots <= N
        causal_shared = jnp.tril(jnp.ones((SHARED_SLOTS, SHARED_SLOTS), dtype=jnp.bool_))
        shared_mask = jnp.broadcast_to(
            causal_shared[None, None, :, :], 
            (batch_size, 1, SHARED_SLOTS, SHARED_SLOTS)
        )
        
        # Merge them to create the final mask for the scratchpad
        extended_ctx_mask = jnp.concatenate([seq_mask, shared_mask], axis=-1)

        def scan_step(carry, inputs):
            (curr_shared, p_remain_prev, 
             expected_shared_accum, ponder_cost_accum, forget_loss_accum, actual_steps_accum,
             forget_density_sum, logits_sum, logits_sq_sum, logits_min, logits_max,
             halts_sq_sum, halts_sum, first_shared, step_count) = carry
             
            t_signal, step_id = inputs
            
            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

            stack_input = self.time_norm(curr_shared) + self.time_signal_norm(t_signal[None, None, :])
            
            new_shared = self.main_stack(
                stack_input, context=shared_ctx, mask=extended_ctx_mask,
                q_pos=shared_pos, kv_pos=shared_kv_pos
            )

            if self.use_forget:
                forget_gate_input = self.forget_norm(new_shared)
                forget = jax.nn.sigmoid(self.forget_head(forget_gate_input))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = jnp.zeros((batch_size,))
            
            pooled = jnp.mean(new_shared, axis=1)
            halt_logits = self.halt_head(jax.nn.gelu(self.halt_pre(pooled))).squeeze(-1)
            halt_prob = jax.nn.sigmoid(halt_logits)
            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, halt_prob)
            halt_prob = jnp.clip(halt_prob, 0.0, 1.0 - 1e-7)
            
            w_t = halt_prob * p_remain_prev
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            
            w_t = jnp.where(step_id == max_steps - 1, w_t + p_remain_next, w_t)
            
            w_t_expanded = w_t[:, None, None]
            
            expected_shared_accum += w_t_expanded * new_shared
            ponder_cost_accum += w_t * jnp.maximum(0, step_id + 1 - MIN_STEPS)
            forget_loss_accum += w_t * forget_val
            actual_steps_accum += w_t * (step_id + 1)
            
            forget_density_sum += jnp.mean(forget_val)
            logits_sum += jnp.mean(halt_logits)
            logits_sq_sum += jnp.mean(jnp.square(halt_logits))
            logits_min = jnp.minimum(logits_min, jnp.min(halt_logits))
            logits_max = jnp.maximum(logits_max, jnp.max(halt_logits))
            halts_sum += jnp.mean(halt_prob)
            halts_sq_sum += jnp.mean(jnp.square(halt_prob))
            
            first_shared = jnp.where(step_id == 0, new_shared, first_shared)
            
            new_carry = (
                new_shared, p_remain_next, 
                expected_shared_accum, ponder_cost_accum, forget_loss_accum, actual_steps_accum,
                forget_density_sum, logits_sum, logits_sq_sum, logits_min, logits_max,
                halts_sq_sum, halts_sum, first_shared, step_count + 1
            )
            return new_carry, None

        init_carry = (
            z_shared, 
            jnp.ones((batch_size,)), 
            jnp.zeros_like(z_shared_base), 
            jnp.zeros((batch_size,)), 
            jnp.zeros((batch_size,)), 
            jnp.zeros((batch_size,)),
            jnp.array(0.0), 
            jnp.array(0.0), 
            jnp.array(0.0), 
            jnp.array(jnp.inf), 
            jnp.array(-jnp.inf), 
            jnp.array(0.0), 
            jnp.array(0.0), 
            jnp.zeros_like(z_shared_base), 
            jnp.array(0)
        )

        final_carry, _ = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry, (all_time_embeds, jnp.arange(max_steps))
        )
        
        (final_shared, _, expected_shared, ponder_cost, forget_loss, actual_steps,
         forget_density_sum, logits_sum, logits_sq_sum, logits_min, logits_max,
         halts_sq_sum, halts_sum, first_shared, step_count) = final_carry

        N = max_steps
        logits_mean = logits_sum / N
        logits_var = jnp.maximum(0.0, (logits_sq_sum / N) - jnp.square(logits_mean))
        halts_mean = halts_sum / N
        halts_var = jnp.maximum(0.0, (halts_sq_sum / N) - jnp.square(halts_mean))
        
        norms = jnp.sqrt(jnp.sum(jnp.square(expected_shared), axis=-1, keepdims=True) + 1e-8)
        normalized = expected_shared / norms
        slot_corr = jnp.einsum('bsd,btd->bst', normalized, normalized)
        saturation_score = jnp.mean(jnp.abs(slot_corr))

        diff_sq = jnp.sum(jnp.square(final_shared - first_shared), axis=-1)
        base_sq = jnp.sum(jnp.square(first_shared), axis=-1)
        drift = jnp.mean(jnp.sqrt(diff_sq + 1e-8) / (jnp.sqrt(base_sq + 1e-8)))

        halt_diag = {
            'logits_mean': logits_mean,
            'logits_std': jnp.sqrt(logits_var),
            'logits_min': logits_min,
            'logits_max': logits_max,
            'prob_std': jnp.sqrt(halts_var),
            'saturation': saturation_score,
            'temporal_drift': drift,
            'forget_density': forget_density_sum / N,
            'logit_spread': logits_max - logits_min,
            'prob_mean': jnp.mean(actual_steps),
            'actual_steps': jnp.mean(actual_steps)
        }

        z_out = self.main_stack(
            z_seq, 
            context=expected_shared, 
            mask=jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_), 
            q_pos=seq_pos, 
            kv_pos=shared_pos
        )
        
        logits = self.seq_norm(z_out) @ self.embed.embedding.value.T
        
        if not training:
            self.hunch_cache.value = expected_shared
            
        return logits, ponder_cost, forget_loss, halt_diag, expected_shared

def calculate_diversity_loss_margin(expected_shared, margin):
    expected_shared = expected_shared.astype(jnp.float32)
    norm = jnp.linalg.norm(expected_shared, axis=-1, keepdims=True)
    normalized_shared = expected_shared / (norm + 1e-8)
    
    dots = jnp.einsum('bsd,btd->bst', normalized_shared, normalized_shared)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]
    
    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)
    
    return jnp.mean(jnp.square(violation * mask))

@nnx.jit
def compute_grad_step(model, batch_tokens, step, prev_hunch=None, should_truncate=False):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        preds, ponder_cost, forget_cost, halt_diag, expected_shared = model(
            inputs, training=True, prev_hunch=prev_hunch
        )
        div_loss = calculate_diversity_loss_margin(expected_shared, margin=0.5)

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        non_pad_mask = (targets != PAD_TOKEN_ID)
        num_valid = jnp.sum(non_pad_mask).clip(min=1)
        token_loss = jnp.sum(ce_loss * non_pad_mask) / num_valid

        current_p_lambda = ponder_lambda_schedule(step)
        current_f_lambda = forget_lambda_schedule(step)
        current_d_lambda = diversity_lambda_schedule(step)

        total_loss = (
            token_loss
            + current_p_lambda * jnp.mean(ponder_cost)
            + current_f_lambda * jnp.mean(forget_cost)
            + current_d_lambda * div_loss
        ) / ACCUMULATION_STEPS
        
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        halt_diag['diversity_loss'] = div_loss
        
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    *metrics, next_hunch = aux
    
    # If should_truncate is True, we break the gradient chain here.
    # Also, we break it if the global step says so.
    should_refresh = jnp.any(should_truncate | (step % HUNCH_REFRESH_EVERY == 0)).squeeze()
    
    next_hunch = jax.lax.cond(
        should_refresh,
        lambda x: jax.lax.stop_gradient(x),
        lambda x: x,
        next_hunch
    )
    
    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))
    
    return loss, tuple(metrics), next_hunch, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)