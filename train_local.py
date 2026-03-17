from flax import linen as nn
import jax
import optax
from flax import nnx
import jax.numpy as jnp

#Params
LATENT_DIM = 768
NUM_BLOCKS = 8
SHARED_SLOTS = 64
VOCAB_SIZE = 100277
MAX_STEPS_LIMIT = 32

#Training
MAX_SEQ_LEN = 1024
MIN_STEPS = 8
BATCH_SIZE = 32
PAD_TOKEN_ID = 100257
FORGET_LAMBDA = 1e-6
DIVERSITY_LAMBDA = 0.5
HUNCH_REFRESH_EVERY = 4

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=4, rngs=None, dtype=jnp.bfloat16):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin_cached = jnp.sin(freqs).astype(dtype)
        self.cos_cached = jnp.cos(freqs).astype(dtype)

        self.cache = nnx.Cache(None)

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)

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
#
        sin_q = self.sin_cached[q_pos, None, :]
        cos_q = self.cos_cached[q_pos, None, :]
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        q = q * self.scale

        sin_kv = self.sin_cached[kv_pos, None, :]
        cos_kv = self.cos_cached[kv_pos, None, :]
        k = (k * cos_kv) + (rotate_half(k) * sin_kv)

        if use_cache:
            if self.cache.value is not None:
                prev_k, prev_v = self.cache.value
                k = jnp.concatenate([prev_k, k], axis=1)
                v = jnp.concatenate([prev_v, v], axis=1)
            self.cache.value = (k, v)

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.repeat(k, repeats, axis=2)
        v_expanded = jnp.repeat(v, repeats, axis=2)

        out = nn.attention.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
            if mask is not None else None,
        )
        return self.o_proj(out.reshape(b, s, d))


class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=4, rngs=rngs, dtype=dtype)
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
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
        self.blocks = [
            StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
            for _ in range(num_blocks)
        ]
        self.num_blocks = num_blocks

    def reset_state(self):
        for block in self.blocks:
            block.attn.cache.value = None

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        for block in self.blocks:
            x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache)
        return x


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.bfloat16, use_forget=True):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), dtype)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.main_stack = BlockStack(num_blocks, latent_dim, num_heads=16, rngs=rngs, dtype=dtype)

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
            self.forget_head = nnx.Linear(latent_dim, latent_dim, bias_init=jax.nn.initializers.constant(0.0), rngs=rngs, dtype=dtype)

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
            curr_shared, p_remain_prev = carry
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
                forget_val = 0.0
            
            pooled = jnp.mean(new_shared, axis=1)
            halt_logits = self.halt_head(jax.nn.gelu(self.halt_pre(pooled))).squeeze(-1)
            halt_prob = jax.nn.sigmoid(halt_logits)
            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, halt_prob)
            
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            return (new_shared, p_remain_next), (new_shared, halt_prob, forget_val, halt_logits)

        (final_shared, _), (all_shared, all_halts, all_forget_l1, all_logits) = jax.lax.scan(
            jax.checkpoint(scan_step), (z_shared, jnp.ones((batch_size,))), (all_time_embeds, jnp.arange(max_steps))
        )


        all_halts = jnp.clip(all_halts, 0.0, 1.0 - 1e-7)
        
        p_remain = jnp.concatenate(
            [jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0
        )
        
        step_weights = all_halts * p_remain
        last_step_extra = p_remain[-1] * (1.0 - all_halts[-1])
        step_weights = step_weights.at[-1].add(last_step_extra)

        weights_for_shared = step_weights[:, :, None, None]
        expected_shared = jnp.sum(weights_for_shared * all_shared, axis=0)

        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        
        actual_steps = jnp.sum(step_weights * step_indices, axis=0) 

        ponder_cost = jnp.sum(step_weights * jnp.maximum(0, step_indices - MIN_STEPS), axis=0)
        forget_loss = jnp.sum(step_weights * all_forget_l1, axis=0)
        
        norms = jnp.sqrt(jnp.sum(jnp.square(expected_shared), axis=-1, keepdims=True) + 1e-8)
        normalized = expected_shared / norms
        slot_corr = jnp.einsum('bsd,btd->bst', normalized, normalized)
        saturation_score = jnp.mean(jnp.abs(slot_corr))

        diff_sq = jnp.sum(jnp.square(all_shared[-1] - all_shared[0]), axis=-1)
        base_sq = jnp.sum(jnp.square(all_shared[0]), axis=-1)
        drift = jnp.mean(jnp.sqrt(diff_sq + 1e-8) / (jnp.sqrt(base_sq + 1e-8)))

        forget_density = jnp.mean(all_forget_l1)
        logit_spread = jnp.max(all_logits) - jnp.min(all_logits)

        halt_diag = {
            'logits_mean': jnp.mean(all_logits),
            'logits_std': jnp.std(all_logits),
            'logits_min': jnp.min(all_logits),
            'logits_max': jnp.max(all_logits),
            'prob_std': jnp.std(all_halts),
            'saturation': saturation_score,
            'temporal_drift': drift,
            'forget_density': forget_density,
            'logit_spread': logit_spread,
        }

        halt_diag.update({
            'prob_mean': jnp.mean(actual_steps),
            'actual_steps': jnp.mean(actual_steps) 
        })

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



schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=2e-4,  # Slightly lower peak for stability
    warmup_steps=400, 
    decay_steps=2000, 
    end_value=1e-5
)
ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=400, 
    decay_steps=1000, 
    end_value=2e-4
)
optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule),
)


def calculate_diversity_loss(expected_shared):
    expected_shared = expected_shared.astype(jnp.float32)
    norms = jnp.sqrt(jnp.sum(jnp.square(expected_shared), axis=-1, keepdims=True) + 1e-8)
    normalized_shared = expected_shared / norms
    
    dots = jnp.einsum('bsd,btd->bst', normalized_shared, normalized_shared)
    
    identity = jnp.eye(SHARED_SLOTS)[None, :, :]
    
    return jnp.mean(jnp.square(dots - identity))

@nnx.jit
def train_step(model, opt, batch_tokens, step, f_lambda, prev_hunch=None, should_truncate=False):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        preds, ponder_cost, forget_cost, halt_diag, expected_shared = model(
            inputs, training=True, prev_hunch=prev_hunch
        )
        div_loss = calculate_diversity_loss(expected_shared)

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        non_pad_mask = (targets != PAD_TOKEN_ID)
        num_valid = jnp.sum(non_pad_mask).clip(min=1)
        token_loss = jnp.sum(ce_loss * non_pad_mask) / num_valid

        current_p_lambda = ponder_lambda_schedule(step)

        num_devices = jax.device_count()

        total_loss = (
            token_loss
            + current_p_lambda * jnp.mean(ponder_cost)
            + f_lambda * jnp.mean(forget_cost)
            + DIVERSITY_LAMBDA * div_loss
        )
        
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        total_loss = total_loss / num_devices
        
        halt_diag['diversity_loss'] = div_loss
        
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    opt.update(grads)
    
    *metrics, next_hunch = aux
    
    # If should_truncate is True, we break the gradient chain here.
    # Also, we break it if the global step says so.
    should_refresh = should_truncate | (step % HUNCH_REFRESH_EVERY == 0)
    
    next_hunch = jax.lax.cond(
        should_refresh,
        lambda x: jax.lax.stop_gradient(x),
        lambda x: x,
        next_hunch
    )
    
    return loss, tuple(metrics), next_hunch
