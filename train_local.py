from flax import linen as nn
import jax
import optax
from flax import nnx
import jax.numpy as jnp

#Params
LATENT_DIM = 512
NUM_BLOCKS = 8
SHARED_SLOTS = 32
VOCAB_SIZE = 100352
MAX_STEPS_LIMIT = 16

#Training
MAX_SEQ_LEN = 1024
MIN_STEPS = 4
BATCH_SIZE = 128 # Effective batch size
GRAD_ACCUM_STEPS = 128 # Number of micro-batches to accumulate
PAD_TOKEN_ID = 100351
HUNCH_REFRESH_EVERY = 4

# Soft label settings
SOFT_LABEL_K = 64
SOFT_LABEL_TEMP = 0.1
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4 # Ideal ratio is 4:1


def apply_rope(x, freqs_complex):
    d = x.shape[-1]
    x_complex = jax.lax.complex(
        x[..., :d // 2].astype(jnp.float32), 
        x[..., d // 2:].astype(jnp.float32)
    )
    x_rotated = x_complex * freqs_complex
    x_out = jnp.concatenate([jnp.real(x_rotated), jnp.imag(x_rotated)], axis=-1)
    return x_out.astype(x.dtype)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=NUM_GROUPS, rngs=None, dtype=jnp.float32):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.freqs_complex = jnp.exp(1j * freqs).astype(jnp.complex64)

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
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

        freqs_q = self.freqs_complex[q_pos, None, :]
        q = apply_rope(q, freqs_q)
        q = q * self.scale

        freqs_kv = self.freqs_complex[kv_pos, None, :]
        k = apply_rope(k, freqs_kv)

        if use_cache:
            if not hasattr(self, 'cache'):
                self.cache = nnx.Cache(None)
            if self.cache.value is not None:
                prev_k, prev_v = self.cache.value
                k = jnp.concatenate([prev_k, k], axis=1)
                v = jnp.concatenate([prev_v, v], axis=1)
            self.cache.value = (k, v)

        out = jax.nn.dot_product_attention(
            q, k, v,
            mask=mask,
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
        pass # Not using cache for now

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
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), jnp.float32).astype(dtype)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.main_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype)

        halt_pre_dim = latent_dim // 4
        self.halt_pre = nnx.Linear(latent_dim, halt_pre_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(halt_pre_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), -3.0)
        
        self.time_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.forget_norm = nnx.RMSNorm(latent_dim * 2, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)

        self.hunch_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.hunch_gate = nnx.Linear(
            latent_dim * 2, 1,
            bias_init=jax.nn.initializers.constant(-2.0),
            rngs=rngs, dtype=dtype,
        )

        self.use_forget = use_forget
        if self.use_forget:
            self.forget_head = nnx.Linear(latent_dim * 2, 1, bias_init=jax.nn.initializers.constant(2.0), rngs=rngs, dtype=dtype)

        self.hunch_cache = nnx.Cache(None)


    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True, prev_hunch=None):
        batch_size, seq_len = tokens.shape
        pad_mask = tokens != PAD_TOKEN_ID
        
        current_hunch = None
        if training:
            current_hunch = prev_hunch
            self.main_stack.reset_state()
        else:
            self.main_stack.reset_state()
            if not should_refresh and self.hunch_cache.value is not None:
                current_hunch = self.hunch_cache.value
                
        seq_pos = jnp.arange(seq_len)
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        full_pos = jnp.concatenate([seq_pos, shared_pos])

        z_seq = self.embed(tokens)
        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))

        if current_hunch is not None:
            valid_lengths = jnp.sum(pad_mask, axis=1)
            last_token_idx = jnp.maximum(0, valid_lengths - 1)
            batch_indices = jnp.arange(batch_size)
            seq_summary = z_seq[batch_indices, last_token_idx][:, None, :]

            hunch_input = jnp.concatenate([
                self.hunch_norm(current_hunch), 
                jnp.broadcast_to(seq_summary, current_hunch.shape)
            ], axis=-1)
            
            gate = jax.nn.sigmoid(self.hunch_gate(hunch_input))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        all_time_embeds = self.time_embed(jnp.arange(max_steps))
        
        seq_self_mask = nn.make_causal_mask(jnp.ones((batch_size, seq_len)), dtype=jnp.bool_)
        seq_self_mask = seq_self_mask & pad_mask[:, None, None, :]
        
        # Sequence tokens see all scratchpad slots
        seq_shared_mask = jnp.ones((batch_size, 1, seq_len, SHARED_SLOTS), dtype=jnp.bool_)
        
        # Scratchpad slots see all valid sequence tokens
        shared_seq_mask = jnp.broadcast_to(pad_mask[:, None, None, :], (batch_size, 1, SHARED_SLOTS, seq_len))
        
        # Scratchpad slots see themselves causally
        shared_shared_mask = nn.make_causal_mask(jnp.ones((batch_size, SHARED_SLOTS)), dtype=jnp.bool_)
        
        mask_row1 = jnp.concatenate([seq_self_mask, seq_shared_mask], axis=-1)
        mask_row2 = jnp.concatenate([shared_seq_mask, shared_shared_mask], axis=-1)
        unified_mask = jnp.concatenate([mask_row1, mask_row2], axis=-2)

        def scan_step(carry, inputs):
            curr_seq, curr_shared, p_remain_prev = carry
            t_signal, step_id = inputs
            
            x = jnp.concatenate([curr_seq, curr_shared], axis=1)
            
            x_shared = x[:, seq_len:, :]
            x_shared = self.time_norm(x_shared) + self.time_signal_norm(t_signal[None, None, :])
            x = jnp.concatenate([x[:, :seq_len, :], x_shared], axis=1)
            
            x_new = self.main_stack(
                x, context=None, mask=unified_mask,
                q_pos=full_pos, kv_pos=full_pos
            )
            
            new_seq = x_new[:, :seq_len, :]
            new_shared = x_new[:, seq_len:, :]

            if self.use_forget:
                gate_context = jnp.concatenate([curr_shared, new_shared], axis=-1)
                forget_gate_input = self.forget_norm(gate_context)
                forget = jax.nn.sigmoid(self.forget_head(forget_gate_input))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = 0.0
            
            pooled = jnp.max(new_shared, axis=1)
            halt_logits = self.halt_head(jax.nn.gelu(self.halt_pre(pooled))).squeeze(-1)
            halt_prob = jax.nn.sigmoid(halt_logits)
            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, halt_prob)
            
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            return (new_seq, new_shared, p_remain_next), (new_seq, new_shared, halt_prob, forget_val, halt_logits)

        (final_seq, final_shared, _), (all_seq, all_shared, all_halts, all_forget_l1, all_logits) = nnx.scan(
            nnx.remat(scan_step),
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0)
        )((z_seq, z_shared, jnp.ones((batch_size,))), (all_time_embeds, jnp.arange(max_steps)))

        all_halts = jnp.clip(all_halts, 0.0, 1.0 - 1e-7)
        p_remain = jnp.concatenate(
            [jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0
        )
        
        step_weights = all_halts * p_remain
        last_step_extra = p_remain[-1] * (1.0 - all_halts[-1])
        step_weights = step_weights.at[-1].add(last_step_extra)

        weights_for_stream = step_weights[:, :, None, None]

        expected_seq = jnp.sum(weights_for_stream * all_seq, axis=0)
        expected_shared = jnp.sum(weights_for_stream * all_shared, axis=0)

        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        actual_steps = jnp.sum(step_weights * step_indices, axis=0) 
        ponder_cost = jnp.sum(step_weights * jnp.maximum(0, step_indices - MIN_STEPS), axis=0)
        forget_loss = jnp.sum(step_weights * all_forget_l1, axis=0)
        
        norm = jnp.sqrt(jnp.sum(jnp.square(expected_shared), axis=-1, keepdims=True) + 1e-8)
        normalized = expected_shared / norm
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
            'prob_mean': jnp.mean(actual_steps),
            'actual_steps': jnp.mean(actual_steps) 
        }

        logits = self.seq_norm(expected_seq) @ self.embed.embedding.value.T
        
        if not training:
            self.hunch_cache.value = expected_shared
            
        return logits, ponder_cost, forget_loss, halt_diag, expected_shared

def soft_label_loss(logits, targets, embed_table, non_pad_mask, k=SOFT_LABEL_K, temperature=SOFT_LABEL_TEMP):
    B, L, V = logits.shape
    
    embed_table = jax.lax.stop_gradient(embed_table.astype(jnp.float32))
    logits = logits.astype(jnp.float32)

    embed_norm = jnp.sqrt(jnp.sum(jnp.square(embed_table), axis=-1, keepdims=True) + 1e-8)
    embed_normed = embed_table / embed_norm

    target_emb = embed_normed[targets] 

    flat_targets = target_emb.reshape(B * L, -1)
    flat_logits = logits.reshape(B * L, V)
    flat_mask = non_pad_mask.reshape(B * L)

    sims = jnp.matmul(flat_targets, embed_normed.T)
    topk_vals, topk_idx = jax.lax.top_k(sims, k)
    soft_targets = jax.nn.softmax(topk_vals / temperature, axis=-1)
    
    lse = jax.nn.logsumexp(flat_logits, axis=-1, keepdims=True)
    topk_logits = jnp.take_along_axis(flat_logits, topk_idx, axis=-1)
    topk_log_probs = topk_logits - lse
    
    per_token_losses = -jnp.sum(soft_targets * topk_log_probs, axis=-1)

    num_valid = jnp.sum(flat_mask).clip(min=1)
    loss = jnp.sum(per_token_losses * flat_mask) / num_valid
    
    return loss

def calculate_diversity_loss_margin(expected_shared, margin):
    norm_shared = jnp.sqrt(jnp.sum(jnp.square(expected_shared), axis=-1, keepdims=True) + 1e-8)
    normalized_shared = expected_shared / norm_shared
    
    dots = jnp.einsum('bsd,btd->bst', normalized_shared, normalized_shared)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]
    
    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)
    
    return jnp.mean(jnp.square(violation * mask))

@nnx.jit
def train_step(model, opt, batch_tokens, step, ponder_lambda, forget_lambda, diversity_lambda, semantic_alpha, prev_hunch=None, should_truncate=False):
    # Split tokens into micro-batches for gradient accumulation
    micro_batch_size = BATCH_SIZE // GRAD_ACCUM_STEPS
    micro_tokens = batch_tokens.reshape(GRAD_ACCUM_STEPS, micro_batch_size, -1)
    
    if prev_hunch is None:
        prev_hunch = jnp.zeros((micro_batch_size, SHARED_SLOTS, LATENT_DIM), dtype=jnp.float32)

    initial_grads = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))

    graphdef, state = nnx.split(model)

    def micro_step_accum(carry, micro_batch):
        hunch, accum_grads, current_step = carry
        
        def loss_fn(state_inner, tokens, h):
            m = nnx.merge(graphdef, state_inner)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            
            logits, ponder_cost, forget_cost, halt_diag, expected_shared = m(
                inputs, training=True, prev_hunch=h
            )
            div_loss = calculate_diversity_loss_margin(expected_shared, margin=0.3)
            non_pad_mask = (targets != PAD_TOKEN_ID)

            soft_loss = soft_label_loss(
                logits, targets,
                embed_table=m.embed.embedding.value,
                non_pad_mask=non_pad_mask,
            )
            
            hard_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            hard_loss = jnp.sum(hard_loss * non_pad_mask) / jnp.sum(non_pad_mask).clip(min=1)

            token_loss = (1.0 - semantic_alpha) * hard_loss + semantic_alpha * soft_loss
            total_loss = token_loss + (ponder_lambda * jnp.mean(ponder_cost)) + (forget_lambda * jnp.mean(forget_cost)) + (diversity_lambda * div_loss)
            total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
            
            halt_diag['diversity_loss'] = div_loss
            
            return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

        (loss, aux), grads = nnx.value_and_grad(loss_fn, wrt=nnx.Param, has_aux=True)(state, micro_batch, hunch)
        
        token_loss_val, p_cost, f_cost, halt_diag, next_hunch = aux
        
        should_refresh = should_truncate | (current_step % HUNCH_REFRESH_EVERY == 0)
        next_hunch = jax.lax.cond(should_refresh, lambda x: jnp.zeros_like(x), lambda x: x, next_hunch)
        
        new_accum_grads = jax.tree_util.tree_map(jnp.add, accum_grads, grads)
        
        return (next_hunch, new_accum_grads, current_step + 1), (loss, (token_loss_val, p_cost, f_cost, halt_diag))

    (final_hunch, total_grads, _), (losses, metrics_history) = jax.lax.scan(
        micro_step_accum, 
        (prev_hunch, initial_grads, step * GRAD_ACCUM_STEPS), 
        micro_tokens
    )
    
    avg_grads = jax.tree_util.tree_map(lambda x: x / GRAD_ACCUM_STEPS, total_grads)
    opt.update(avg_grads)
    
    avg_loss = jnp.mean(losses)
    avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_history)
    
    return avg_loss, avg_metrics, final_hunch
