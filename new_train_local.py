import jax
import optax
from flax import nnx
import jax.numpy as jnp
from train_local import (
    ponder_lambda_schedule, 
    forget_lambda_schedule
)


#Keep (most) values powers of 2 if you know what's good for you

#Params
LATENT_DIM = 512    #Must be multiple of 128
NUM_BLOCKS = 4
SHARED_SLOTS = 32
VOCAB_SIZE = 100352 #Must be multiple of 128
MAX_STEPS_LIMIT = 16

#Training
MAX_SEQ_LEN = 1024
MIN_STEPS = 4
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100257
HUNCH_REFRESH_EVERY = 4

def calculate_diversity_loss_per_batch(shared_state, margin):
    shared_state = shared_state.astype(jnp.float32)
    norm = jnp.linalg.norm(shared_state, axis=-1, keepdims=True)
    normalized = shared_state / (norm + 1e-8)

    dots = jnp.einsum('bsd,btd->bst', normalized, normalized)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]

    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)

    return jnp.mean(jnp.square(violation * mask), axis=(1, 2))

# Standard Next-Token Prediction Settings
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4 #Ideal ratio is 4:1

def apply_rope(x, sin_table, cos_table):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos_table - x2 * sin_table
    out2 = x1 * sin_table + x2 * cos_table
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape).astype(x.dtype)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=NUM_GROUPS, rngs=None, dtype=jnp.float32):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_table = jnp.sin(freqs).astype(jnp.float32)
        self.cos_table = jnp.cos(freqs).astype(jnp.float32)

        self.k_cache = nnx.Cache(None)
        self.v_cache = nnx.Cache(None)
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)

        self.o_proj = nnx.Linear(
            in_features, in_features,
            rngs=rngs, dtype=dtype
        )

        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=dtype)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=dtype)

    def reset_state(self):
        self.k_cache.value = None
        self.v_cache.value = None
        self.cache_index.value = jnp.zeros_like(self.cache_index.value)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
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

        sin_q = self.sin_table[q_pos, None, :]
        cos_q = self.cos_table[q_pos, None, :]
        q = apply_rope(q, sin_q, cos_q)
        q = q * self.scale

        sin_kv = self.sin_table[kv_pos, None, :]
        cos_kv = self.cos_table[kv_pos, None, :]
        k = apply_rope(k, sin_kv, cos_kv)

        if use_cache:
            if self.k_cache.value is None:
                # Pre-allocate cache to maximum possible length
                cache_shape = (b, MAX_SEQ_LEN + SHARED_SLOTS, self.num_groups, self.head_dim)  
                self.k_cache.value = jnp.zeros(cache_shape, dtype=x.dtype)
                self.v_cache.value = jnp.zeros(cache_shape, dtype=x.dtype)
                self.cache_index.value = jnp.array(0, dtype=jnp.int32)

            # Start position for current update
            curr_idx = self.cache_index.value

            # Update cache using dynamic_update_slice (O(1) in XLA)
            self.k_cache.value = jax.lax.dynamic_update_slice(self.k_cache.value, k, (0, curr_idx, 0, 0))
            self.v_cache.value = jax.lax.dynamic_update_slice(self.v_cache.value, v, (0, curr_idx, 0, 0))

            # Advance the pointer
            self.cache_index.value = curr_idx + s_kv

            # Use the full pre-allocated buffer for attention
            k = self.k_cache.value
            v = self.v_cache.value

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.repeat(k, repeats, axis=2)
        v_expanded = jnp.repeat(v, repeats, axis=2)

        if mask is not None:
             mask_expanded = jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
        else:
             mask_expanded = None

        out = jax.nn.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=mask_expanded,
            is_causal=is_causal,
        )
        return self.o_proj(out.reshape(b, s, d))


class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=NUM_GROUPS, rngs=rngs, dtype=dtype)
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

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
        normed_context = self.norm1(context) if context is not None else None
        attn_out = self.attn(self.norm1(x), context=normed_context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
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
            block.attn.reset_state()

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False, reverse=False):
        blocks_iter = reversed(self.blocks) if reverse else self.blocks
        for block in blocks_iter:
            x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
        return x


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)   

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), dtype)     
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype)
        self.encoder_stack = self.stack
        self.reasoning_stack = self.stack
        self.decoder_stack = self.stack

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
        pad_mask = tokens != PAD_TOKEN_ID

        current_hunch = None
        if training:
            current_hunch = prev_hunch
            self.encoder_stack.reset_state()
            self.reasoning_stack.reset_state()
            self.decoder_stack.reset_state()
        else:
            self.encoder_stack.reset_state()
            self.reasoning_stack.reset_state()
            self.decoder_stack.reset_state()
            if not should_refresh and self.hunch_cache.value is not None:
                current_hunch = self.hunch_cache.value

        seq_pos, shared_pos = jnp.arange(seq_len), jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        z_seq_raw = self.embed(tokens)

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        seq_attn_mask = pad_mask[:, None, None, :] & causal_mask[None, None, :, :]
        z_seq = self.encoder_stack(z_seq_raw, mask=seq_attn_mask, q_pos=seq_pos, kv_pos=seq_pos)

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))

        if current_hunch is not None:
            gate = jax.nn.sigmoid(self.hunch_gate(self.hunch_norm(current_hunch)))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        seq_mask = jnp.broadcast_to(
            pad_mask[:, None, None, :],
            (batch_size, 1, SHARED_SLOTS, seq_len)
        )

        causal_shared = jnp.tril(jnp.ones((SHARED_SLOTS, SHARED_SLOTS), dtype=jnp.bool_))      
        shared_mask = jnp.broadcast_to(causal_shared[None, None, :, :], (batch_size, 1, SHARED_SLOTS, SHARED_SLOTS))
        extended_ctx_mask = jnp.concatenate([seq_mask, shared_mask], axis=-1)
        shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

        c_steps = max_steps // 2

        def scan_step(carry, inputs):
            curr_shared, p_remain_prev, weighted_shared_acc, ponder_acc, div_accum = carry     
            t_signal, step_id = inputs

            stack_input = self.time_norm(curr_shared) + self.time_signal_norm(t_signal[None, None, :])

            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)
            new_shared = self.reasoning_stack(
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

            step_weight = halt_prob * p_remain_prev
            p_remain_next = p_remain_prev * (1.0 - halt_prob)

            step_div = calculate_diversity_loss_per_batch(new_shared, margin=0.5)
            div_accum = div_accum + jax.lax.stop_gradient(step_weight) * step_div

            weighted_shared_acc = weighted_shared_acc + step_weight[:, None, None] * new_shared
            step_idx = (step_id + 1).astype(jnp.float32)
            ponder_acc = ponder_acc + step_weight * jnp.maximum(0.0, step_idx - MIN_STEPS)     

            return (new_shared, p_remain_next, weighted_shared_acc, ponder_acc, div_accum), (new_shared, halt_prob, forget_val, halt_logits)

        # Seed with z_shared so the decoder always receives a non-zero context,
        # even at init when halt probs are very low and weighted_shared_acc accumulates slowly.
        init_weighted_shared = jnp.zeros_like(z_shared)
        init_ponder = jnp.zeros((batch_size,))
        init_div = jnp.zeros((batch_size,))
        init_carry = (z_shared, jnp.ones((batch_size,)), init_weighted_shared, init_ponder, init_div)

        scan_inputs = (all_time_embeds, jnp.arange(max_steps, dtype=jnp.int32))

        final_carry, (all_shared, all_halts, all_forget_l1, all_logits) = jax.lax.scan(        
            jax.checkpoint(scan_step),
            init_carry,
            scan_inputs,
        )

        final_shared, p_remain_final, weighted_shared_acc, ponder_cost_acc, div_cost_acc = final_carry

        last_step_weight = p_remain_final
        weighted_shared_acc = weighted_shared_acc + last_step_weight[:, None, None] * all_shared[-1]
        expected_shared = weighted_shared_acc

        last_step_idx = jnp.float32(max_steps)
        ponder_cost = ponder_cost_acc + p_remain_final * jnp.maximum(0.0, last_step_idx - MIN_STEPS)

        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain
        step_weights = step_weights.at[-1].add(p_remain[-1] * (1.0 - all_halts[-1]))

        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        actual_steps = jnp.sum(step_weights * step_indices, axis=0)

        obs_logits = all_logits[c_steps:]
        halt_diag = {
            'logits_mean': jnp.mean(obs_logits),
            'logits_std': jnp.std(obs_logits),
            'logits_min': jnp.min(obs_logits),
            'logits_max': jnp.max(obs_logits),
            'logit_spread': jnp.max(obs_logits) - jnp.min(obs_logits),
            'prob_std': jnp.std(all_halts[c_steps:]),
            'prob_mean': jnp.mean(all_halts[c_steps:]),
            'actual_steps': jnp.mean(actual_steps),
            'forget_density': jnp.mean(all_forget_l1[c_steps:]),
            'saturation': jnp.mean(jnp.abs(all_halts[c_steps:] - 0.5) * 2.0),
            'temporal_drift': jnp.mean(jnp.abs(all_halts[c_steps+1:] - all_halts[c_steps:-1])) 
        }

        decoder_mask = jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_)

        z_out = self.decoder_stack(
            z_seq,
            context=expected_shared,
            mask=decoder_mask,
            q_pos=seq_pos,
            kv_pos=shared_pos,
            is_causal=False
        )

        logits = self.seq_norm(z_out) @ self.embed.embedding.value.T

        if not training:
            self.hunch_cache.value = expected_shared

        return logits, ponder_cost, jnp.sum(step_weights * all_forget_l1, axis=0), jnp.mean(div_cost_acc), halt_diag, expected_shared


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

        logits, ponder_cost, forget_cost, div_loss, halt_diag, expected_shared = model(        
            inputs, training=True, prev_hunch=prev_hunch
        )

        non_pad_mask = (targets != PAD_TOKEN_ID)

        p_lambda = ponder_lambda_schedule(step)
        f_lambda = forget_lambda_schedule(step)

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        non_pad_mask = (targets != PAD_TOKEN_ID)
        num_valid = jnp.sum(non_pad_mask).clip(min=1)
        token_loss = jnp.sum(ce_loss * non_pad_mask) / num_valid

        total_loss = token_loss + (p_lambda * jnp.mean(ponder_cost)) + \
                     (f_lambda * jnp.mean(forget_cost))
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        #total_loss = total_loss / ACCUMULATION_STEPS

        halt_diag.update({
            'diversity_loss': jax.lax.stop_gradient(div_loss),
        })

        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    grad_norm = optax.global_norm(grads)

    *metrics, next_hunch = aux

    # Apply periodic refresh or reset based on should_truncate
    should_refresh = jnp.any(should_truncate | (step % HUNCH_REFRESH_EVERY == 0)).squeeze()    

    next_hunch = jax.lax.cond(
        should_refresh,
        lambda h: jax.lax.stop_gradient(h),
        lambda h: h,
        next_hunch,
    )

    return loss, tuple(metrics), next_hunch, grads, grad_norm

@nnx.jit
def apply_grads(optimizer, grads, model):
    optimizer.update(grads, model)