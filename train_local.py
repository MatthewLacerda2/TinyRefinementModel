import jax
import optax
from flax import nnx
import jax.numpy as jnp

#Keep (most) values powers of 2 if you know what's good for you

#Params
LATENT_DIM = 512
NUM_BLOCKS = 4
SHARED_SLOTS = 32
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

WARMUP_STEPS = 100
DECAY_STEPS = 600

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=5e-5,
    peak_value=3e-4,
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=5e-5
)

ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=1e-5
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=4e-3
)



def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)

weight_decay_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=1e-2
)

optimizer_chain = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_schedule,
            weight_decay=weight_decay_schedule,
            mask=weight_decay_mask,
        ),
    ),
    every_k_schedule=128,
    use_grad_mean=True
)

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

        self.k_cache = nnx.Cache(None)
        self.v_cache = nnx.Cache(None)
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)

        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=jnp.float32)

        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

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

        sin_q = self.sin_cached[q_pos, None, :]
        cos_q = self.cos_cached[q_pos, None, :]
        q = apply_rope(q, sin_q, cos_q)
        q = q * self.scale

        sin_kv = self.sin_cached[kv_pos, None, :]
        cos_kv = self.cos_cached[kv_pos, None, :]
        k = apply_rope(k, sin_kv, cos_kv)

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

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.repeat(k, repeats, axis=2)
        v_expanded = jnp.repeat(v, repeats, axis=2)

        out = jax.nn.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
            if mask is not None else None,
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

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
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
            self.forget_head = nnx.Linear(latent_dim, latent_dim, bias_init=jax.nn.initializers.constant(2.0), rngs=rngs, dtype=dtype)

        self.hunch_cache = nnx.Cache(None)


    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True, prev_hunch=None):
        batch_size, seq_len = tokens.shape
        self.encoder_stack.reset_state()
        self.reasoning_stack.reset_state()
        self.decoder_stack.reset_state()
        
        seq_pos = jnp.arange(seq_len)
        pad_mask = tokens != PAD_TOKEN_ID
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        seq_attn_mask = pad_mask[:, None, None, :] & causal_mask[None, None, :, :]

        z_seq_base = self.embed(tokens)
        z_seq = self.encoder_stack(z_seq_base, mask=seq_attn_mask, q_pos=seq_pos, kv_pos=seq_pos)

        current_hunch = prev_hunch if prev_hunch is not None else None
        if not training and not should_refresh and self.hunch_cache.value is not None:
            current_hunch = self.hunch_cache.value

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        if current_hunch is not None:
            gate = jax.nn.sigmoid(self.hunch_gate(self.hunch_norm(current_hunch)))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        all_time_embeds = self.time_embed(jnp.arange(max_steps))
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        
        s_mask = jnp.broadcast_to(pad_mask[:, None, None, :], (batch_size, 1, SHARED_SLOTS, seq_len))
        causal_shared = jnp.tril(jnp.ones((SHARED_SLOTS, SHARED_SLOTS), dtype=jnp.bool_))
        sh_mask = jnp.broadcast_to(causal_shared[None, None, :, :], (batch_size, 1, SHARED_SLOTS, SHARED_SLOTS))
        extended_ctx_mask = jnp.concatenate([s_mask, sh_mask], axis=-1)

        def scan_step(carry, inputs):
            curr_shared, p_remain_prev, weighted_shared_acc = carry
            t_signal, step_id = inputs
            
            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

            stack_input = self.time_norm(curr_shared) + self.time_signal_norm(t_signal[None, None, :])
            new_shared = self.reasoning_stack(stack_input, context=shared_ctx, mask=extended_ctx_mask, q_pos=shared_pos, kv_pos=shared_kv_pos)

            if self.use_forget:
                forget = jax.nn.sigmoid(self.forget_head(self.forget_norm(new_shared)))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = jnp.zeros((batch_size,))

            pooled = jnp.mean(new_shared, axis=1)
            halt_logit = self.halt_head(jax.nn.gelu(self.halt_pre(pooled))).squeeze(-1)
            halt_prob = jax.nn.sigmoid(halt_logit)
            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, jnp.clip(halt_prob, 0.0, 1.0 - 1e-7))

            step_weight = halt_prob * p_remain_prev
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            weighted_shared_acc = weighted_shared_acc + step_weight[:, None, None] * new_shared

            step_div = calculate_diversity_loss_per_batch(new_shared, margin=0.5)

            return (new_shared, p_remain_next, weighted_shared_acc), (halt_prob, forget_val, halt_logit, step_div, step_weight)

        init_weighted_shared = jnp.zeros_like(z_shared)
        init_carry = (z_shared, jnp.ones((batch_size,)), init_weighted_shared)
        
        final_carry, (all_halts, all_forget_l1, all_halt_logits, all_divs, all_step_weights) = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry, (all_time_embeds, jnp.arange(max_steps))
        )
        final_shared, p_remain_final, weighted_shared_acc = final_carry
        
        expected_shared = weighted_shared_acc + p_remain_final[:, None, None] * final_shared
        
        step_weights = all_step_weights.at[-1].add(p_remain_final)
        
        step_ids = jnp.arange(1, max_steps + 1)[:, None]
        total_p_cost = jnp.mean(jnp.sum(step_weights * jnp.maximum(0, step_ids - MIN_STEPS), axis=0))
        total_f_cost = jnp.mean(jnp.sum(step_weights * all_forget_l1, axis=0))
        total_div_cost = jnp.mean(jnp.sum(jax.lax.stop_gradient(step_weights) * all_divs, axis=0))
        
        actual_steps = jnp.sum(step_weights * step_ids, axis=0)

        z_seq_out = self.decoder_stack(
            z_seq, 
            context=expected_shared, 
            mask=jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_), 
            q_pos=seq_pos, 
            kv_pos=shared_pos,
            is_causal=False
        )
        logits = self.seq_norm(z_seq_out) @ self.embed.embedding.value.T

        c_steps = max_steps // 2
        obs_logits = all_halt_logits[c_steps:]
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
            'temporal_drift': jnp.mean(jnp.abs(all_halts[c_steps+1:] - all_halts[c_steps:-1])),
        }
        
        if not training:
            self.hunch_cache.value = expected_shared

        return logits, total_p_cost, total_f_cost, total_div_cost, halt_diag, expected_shared

def calculate_diversity_loss_per_batch(shared_state, margin):
    shared_state = shared_state.astype(jnp.float32)
    norm = jnp.linalg.norm(shared_state, axis=-1, keepdims=True)
    normalized = shared_state / (norm + 1e-8)
    
    dots = jnp.einsum('bsd,btd->bst', normalized, normalized)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]
    
    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)
    
    return jnp.mean(jnp.square(violation * mask), axis=(1, 2))

@nnx.jit
def compute_grad_step(model, batch_tokens, step, prev_hunch=None, should_truncate=False):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        preds, ponder_cost, forget_cost, div_loss, halt_diag, expected_shared = model(
            inputs, training=True, prev_hunch=prev_hunch
        )

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        non_pad_mask = (targets != PAD_TOKEN_ID)
        num_valid = jnp.sum(non_pad_mask).clip(min=1)
        token_loss = jnp.sum(ce_loss * non_pad_mask) / num_valid

        current_p_lambda = ponder_lambda_schedule(step)
        current_f_lambda = forget_lambda_schedule(step)

        total_loss = (
            token_loss
            + current_p_lambda * jnp.mean(ponder_cost)
            + current_f_lambda * jnp.mean(forget_cost)
        )
        
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        halt_diag['diversity_loss'] = jax.lax.stop_gradient(div_loss)
        
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    *metrics, next_hunch = aux
    
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