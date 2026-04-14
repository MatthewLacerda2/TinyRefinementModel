import jax
import optax
from flax import nnx, struct
import jax.numpy as jnp
from typing import Dict, Any

#Keep (most) values powers of 2 if you know what's good for you

#Params
LATENT_DIM = 512
NUM_BLOCKS = 4
SHARED_SLOTS = 64
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100352

#Training
MAX_STEPS_LIMIT = 16
MIN_STEPS = 4
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100257

HUNCH_REFRESH_EVERY = 4
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4

@struct.dataclass
class ScanStepOutput:
    shared_state: jnp.ndarray
    halt_prob: jnp.ndarray
    forget_val: jnp.ndarray
    storage_val: jnp.ndarray
    halt_logit: jnp.ndarray
    step_div: jnp.ndarray
    step_weight: jnp.ndarray

@struct.dataclass
class ReasonerOutput:
    logits: jnp.ndarray
    ponder_cost: float
    forget_cost: float
    storage_cost: float
    diversity_loss: float
    halt_diag: Dict[str, Any]
    expected_shared: jnp.ndarray

def apply_rope(x, cos, sin):
    d = x.shape[-1]
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

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)

        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)

        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

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
            self.cache_index.value = idx + s_kv
            k = k_cache
            v = v_cache

        # GQA broadcasting (G -> H) is handled implicitly by jax.nn.dot_product_attention
        out = jax.nn.dot_product_attention(
            q, k, v,
            mask=mask if (mask is not None and mask.dtype == jnp.bool_) else None, 
            bias=mask if (mask is not None and mask.dtype != jnp.bool_) else None,
            is_causal=is_causal,
        )
        return self.o_proj(out.reshape(b, s, d))


class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=NUM_GROUPS, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        hidden_dim = int(256 * ((latent_dim * 8 / 3 + 255) // 256))
        self.gate_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=dtype)
        self.up_proj = nnx.Linear(latent_dim, hidden_dim, rngs=rngs, dtype=dtype)
        self.down_proj = nnx.Linear(
            hidden_dim, latent_dim,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs, dtype=dtype,
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

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=True):
        if self.share_weights:
            block = self.blocks[0]
            for _ in range(self.num_blocks):
                x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
        else:
            for block in self.blocks:
                x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
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
        
        self.coder_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False)
        self.reasoner_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=True)

        self.encoder_stack = self.coder_stack
        self.decoder_stack = self.coder_stack
        
        self.reasoning_stack = self.reasoner_stack

        # Feedback loop for costs/budget (ponder, forget, diversity)
        self.meta_proj = nnx.Linear(3, latent_dim, rngs=rngs, dtype=dtype)

        halt_pre_dim = latent_dim // 4
        self.halt_pre = nnx.Linear(latent_dim, halt_pre_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(halt_pre_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), -1.0) 
        
        self.time_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.forget_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        self.hunch_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.hunch_gate = nnx.Linear(
            latent_dim, latent_dim,
            bias_init=jax.nn.initializers.constant(-2.0),
            rngs=rngs, dtype=dtype,
        )

        self.use_forget = use_forget
        if self.use_forget:
            self.forget_head = nnx.Linear(latent_dim, latent_dim, bias_init=jax.nn.initializers.constant(1.0), rngs=rngs, dtype=dtype)

        self.hunch_cache = nnx.Variable(jnp.zeros((BATCH_SIZE, SHARED_SLOTS, latent_dim)))


    def _encode_sequence(self, tokens):
        pad_mask = tokens != PAD_TOKEN_ID
        pad_bias = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
        pad_bias = pad_bias[:, None, None, :]
        
        seq_len = tokens.shape[1]
        seq_pos = jnp.arange(seq_len)
        
        z_seq_base = self.embed(tokens)
        z_seq = self.encoder_stack(z_seq_base, mask=pad_bias, q_pos=seq_pos, kv_pos=seq_pos, is_causal=True)
        return z_seq, pad_mask, seq_pos

    def _reasoning_loop(self, z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared):
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        all_time_embeds = self.time_embed(jnp.arange(max_steps))
        
        pad_part = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
        slot_part = jnp.zeros((batch_size, SHARED_SLOTS), dtype=jnp.float32)
        extended_ctx_bias = jnp.concatenate([pad_part, slot_part], axis=-1)[:, None, None, :]

        def scan_step(carry, inputs):
            curr_shared, p_remain_prev, weighted_shared_acc, prev_forget, prev_div = carry
            t_signal, step_id = inputs
            
            # Feed back ponder, forget, and diversity info as meta-signals
            meta_input = jnp.stack([p_remain_prev, prev_forget, prev_div], axis=-1)
            meta_signal = self.meta_proj(meta_input)[:, None, :] # (B, 1, D)
            
            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

            stack_input = self.time_norm(curr_shared) + self.time_signal_norm(t_signal[None, None, :]) + meta_signal
            new_shared = self.reasoning_stack(stack_input, context=shared_ctx, mask=extended_ctx_bias, q_pos=shared_pos, kv_pos=shared_kv_pos, is_causal=True)

            if self.use_forget:
                forget = jax.nn.sigmoid(self.forget_head(self.forget_norm(new_shared)))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = jnp.zeros((batch_size,))

            storage_val = jnp.mean(jnp.abs(new_shared), axis=(1, 2))

            pooled = jnp.mean(new_shared, axis=1)
            halt_logit = self.halt_head(jax.nn.gelu(self.halt_pre(pooled))).squeeze(-1)
            halt_prob = jax.nn.sigmoid(halt_logit)
            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, jnp.clip(halt_prob, 0.0, 1.0 - 1e-7))

            step_weight = halt_prob * p_remain_prev
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            weighted_shared_acc = weighted_shared_acc + step_weight[:, None, None] * new_shared

            step_div = calculate_diversity_loss_per_batch(new_shared, margin=0.5)

            return (new_shared, p_remain_next, weighted_shared_acc, forget_val, step_div), ScanStepOutput(
                shared_state=new_shared,
                halt_prob=halt_prob, forget_val=forget_val, storage_val=storage_val,
                halt_logit=halt_logit, step_div=step_div, step_weight=step_weight
            )

        init_weighted_shared = jnp.zeros_like(z_shared)
        init_carry = (
            z_shared, 
            jnp.ones((batch_size,)), 
            init_weighted_shared,
            jnp.zeros((batch_size,)),
            jnp.zeros((batch_size,))
        )
        
        final_carry_all, all_outputs = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry, (all_time_embeds, jnp.arange(max_steps))
        )
        # Unpack first three for standard logic
        final_carry = (final_carry_all[0], final_carry_all[1], final_carry_all[2])
        return final_carry, all_outputs, shared_pos

    def _compute_ponder_kl(self, step_weights, p_remain_final, max_steps):
        step_ids = jnp.arange(1, max_steps + 1)[:, None]
        full_step_weights = step_weights.at[-1].add(p_remain_final)

        lambda_p = 0.01 
        active_steps = jnp.maximum(0, step_ids - MIN_STEPS)
        prior_prob = lambda_p * ((1.0 - lambda_p) ** active_steps)
        valid_steps_mask = (step_ids >= MIN_STEPS).astype(jnp.float32)

        prior_prob = (prior_prob * valid_steps_mask)
        prior_prob = prior_prob / (jnp.sum(prior_prob, axis=0, keepdims=True) + 1e-8)

        p_x = full_step_weights + 1e-8 
        q_x = prior_prob + 1e-8

        kl_div_per_step = p_x * (jnp.log(p_x) - jnp.log(q_x))
        kl_div_per_batch = jnp.sum(kl_div_per_step * valid_steps_mask, axis=0)
        return jnp.mean(kl_div_per_batch), full_step_weights

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True):
        batch_size = tokens.shape[0]
        self.encoder_stack.reset_state()
        self.reasoning_stack.reset_state()
        self.decoder_stack.reset_state()
        
        z_seq, pad_mask, seq_pos = self._encode_sequence(tokens)

        if should_refresh:
            self.hunch_cache.value = None

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        current_hunch = self.hunch_cache.value
        
        if current_hunch is not None:
            gate = jax.nn.sigmoid(self.hunch_gate(self.hunch_norm(current_hunch)))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        final_carry, all_outputs, shared_pos = self._reasoning_loop(z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared)
        final_shared, p_remain_final, weighted_shared_acc = final_carry
        
        expected_shared = weighted_shared_acc + p_remain_final[:, None, None] * final_shared
        
        total_p_cost, full_step_weights = self._compute_ponder_kl(all_outputs.step_weight, p_remain_final, max_steps)
        
        total_f_cost = jnp.mean(jnp.sum(full_step_weights * all_outputs.forget_val, axis=0))
        total_s_cost = jnp.mean(jnp.sum(full_step_weights * all_outputs.storage_val, axis=0))
        total_div_cost = jnp.mean(jnp.sum(jax.lax.stop_gradient(full_step_weights) * all_outputs.step_div, axis=0))
        
        actual_steps = jnp.sum(full_step_weights * jnp.arange(1, max_steps + 1)[:, None], axis=0)

        # Calculate Initial Logits (Step 1) to track refinement efficiency
        z_first_out = self.decoder_stack(
            z_seq,
            context=all_outputs.shared_state[0],
            mask=jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_),
            q_pos=seq_pos, 
            kv_pos=shared_pos,
            is_causal=False
        )
        first_logits = self.seq_norm(z_first_out) @ self.embed.embedding.value.T

        z_seq_out = self.decoder_stack(
            z_seq, 
            context=expected_shared, 
            mask=jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_), 
            q_pos=seq_pos, 
            kv_pos=shared_pos,
            is_causal=False
        )
        logits = self.seq_norm(z_seq_out) @ self.embed.embedding.value.T

        # Diagnostics for logging
        
        # Temporal drift of shared state across reasoning steps
        states = all_outputs.shared_state # (max_steps, batch, slots, dim)
        diffs = states[1:] - states[:-1]
        temporal_drift = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(diffs), axis=-1) + 1e-8))
        
        halt_diag = {
            'ponder_kl': total_p_cost,
            'expected_steps': jnp.mean(actual_steps),
            'first_logits': first_logits,
            'temporal_drift': temporal_drift,
            'forget_density': jnp.mean(all_outputs.forget_val),
            'saturation': jnp.mean(p_remain_final) * 100.0, # Percentage of tokens hitting max steps
        }
        
        # Always update the hunch cache; nnx.split will decide whether to carry it
        self.hunch_cache.value = expected_shared

        return ReasonerOutput(
            logits=logits, ponder_cost=total_p_cost, forget_cost=total_f_cost,
            storage_cost=total_s_cost, diversity_loss=total_div_cost,
            halt_diag=halt_diag, expected_shared=expected_shared
        )

def calculate_diversity_loss_per_batch(shared_state, margin):
    shared_state = shared_state.astype(jnp.float32)
    norm = jnp.linalg.norm(shared_state, axis=-1, keepdims=True)
    # Pro Way: Use 1e-6 for epsilon in norms to be bfloat16-ready
    normalized = shared_state / (norm + 1e-6)
    
    dots = jnp.einsum('bsd,btd->bst', normalized, normalized, precision=jax.lax.Precision.HIGHEST)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]
    
    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)
    
    return jnp.mean(jnp.square(violation * mask), axis=(1, 2))

@nnx.jit
def compute_grad_step(model, batch_tokens, step, should_truncate=False):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        out = model(inputs, training=True)
        preds = out.logits
        
        mask = targets != PAD_TOKEN_ID
        token_loss = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets) * mask) / jnp.sum(mask).clip(min=1)

        from schedules import ponder_lambda_schedule, forget_lambda_schedule, storage_lambda_schedule
        p_lambda, f_lambda, s_lambda = ponder_lambda_schedule(step), forget_lambda_schedule(step), storage_lambda_schedule(step)

        total_loss = token_loss + p_lambda * out.ponder_cost + f_lambda * out.forget_cost + s_lambda * out.storage_cost
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        # Calculate First Step CE (Initial Loss)
        first_logits = out.halt_diag.pop('first_logits')
        first_ce = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=first_logits, labels=targets) * mask) / jnp.sum(mask).clip(min=1)

        # Pro Way: Remove logits from the aux return during training to save massive VRAM (400MB+)
        out = out.replace(logits=None) 
        out.halt_diag['diversity_loss'] = jax.lax.stop_gradient(out.diversity_loss)
        out.halt_diag['token_loss'] = jax.lax.stop_gradient(token_loss)
        out.halt_diag['first_ce'] = jax.lax.stop_gradient(first_ce)
        return total_loss, out

    (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    # Pro Way: Implementation-specific state refresh logic.
    should_refresh = jnp.any(should_truncate | (step % HUNCH_REFRESH_EVERY == 0)).squeeze()
    
    # Access the model attribute directly - NNX handles the JIT synchronization
    current_hunch = model.hunch_cache.value
    cleared_hunch = jnp.zeros_like(current_hunch)
    
    carried_hunch = jax.lax.cond(
        should_refresh,
        lambda: jax.lax.stop_gradient(cleared_hunch),
        lambda: jax.lax.stop_gradient(current_hunch)
    )
    
    # Commit the carry-over back to the model's internal cache
    model.hunch_cache.value = carried_hunch

    # Calculate global grad norm
    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))
    
    return loss, out, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)