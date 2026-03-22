from flax import linen as nn
import jax
import optax
from flax import nnx
import jax.numpy as jnp

#Params
LATENT_DIM = 1024
NUM_BLOCKS = 16
SHARED_SLOTS = 128
VOCAB_SIZE = 100352
MAX_STEPS_LIMIT = 64

#Training
MAX_SEQ_LEN = 2048
MIN_STEPS = 8
BATCH_SIZE = 16
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
    def __init__(self, num_heads, in_features, num_groups=NUM_GROUPS, rngs=None, dtype=jnp.bfloat16):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.freqs_complex = jnp.exp(1j * freqs).astype(jnp.complex64)

        self.cache = nnx.Cache(None)

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
            if self.cache.value is not None:
                prev_k, prev_v = self.cache.value
                k = jnp.concatenate([prev_k, k], axis=1)
                v = jnp.concatenate([prev_v, v], axis=1)
            self.cache.value = (k, v)

        # jax.nn.dot_product_attention is optimized (FlashAttention/Fused) and handles GQA internally.
        out = jax.nn.dot_product_attention(
            q, k, v,
            mask=mask,
            is_causal=is_causal,
        )
        return self.o_proj(out.reshape(b, s, d))


class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
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
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
        @nnx.split_rngs(splits=num_blocks)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs_in: nnx.Rngs):
            return StandardReasoningBlock(latent_dim, num_heads, rngs=rngs_in, dtype=dtype)
            
        self.blocks = create_block(rngs)
        self.num_blocks = num_blocks

    def reset_state(self):
        self.blocks.attn.cache.value = None

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(curr_x, block):
            return block(curr_x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache, is_causal=is_causal)
            
        return forward(x, self.blocks)


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.bfloat16, use_forget=True):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), dtype)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.main_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype)

        halt_pre_dim = latent_dim // 4
        self.halt_pre = nnx.Linear(latent_dim, halt_pre_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(halt_pre_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), 2.0)
        
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
                
        seq_pos, shared_pos = jnp.arange(seq_len), jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        z_seq = self.embed(tokens)

        # Use is_causal=True and only pass the pad_mask to take advantage of optimized kernels.
        z_seq = self.main_stack(z_seq, mask=pad_mask[:, None, None, :], q_pos=seq_pos, kv_pos=seq_pos, is_causal=True)

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))

        if current_hunch is not None:
            seq_summary = jnp.mean(z_seq, axis=1, keepdims=True)
            hunch_input = jnp.concatenate([self.hunch_norm(current_hunch), jnp.broadcast_to(seq_summary, current_hunch.shape)], axis=-1)
            gate = jax.nn.sigmoid(self.hunch_gate(hunch_input))
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
        shared_mask = jax.nn.make_causal_mask(jnp.ones((batch_size, SHARED_SLOTS)), dtype=jnp.bool_)
        
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
                gate_context = jnp.concatenate([curr_shared, new_shared], axis=-1)
                forget_gate_input = self.forget_norm(gate_context)
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
        
        normalized = optax.l2_normalize(expected_shared, axis=-1, eps=1e-8)
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

def soft_label_loss(logits, targets, embed_table, non_pad_mask, k=SOFT_LABEL_K, temperature=SOFT_LABEL_TEMP):
    B, L, V = logits.shape
    
    embed_table = jax.lax.stop_gradient(embed_table.astype(jnp.float32))
    logits = logits.astype(jnp.float32)

    embed_normed = optax.l2_normalize(embed_table, axis=-1, eps=1e-8)

    target_emb = embed_normed[targets] 

    flat_targets = target_emb.reshape(B * L, -1)
    flat_logits = logits.reshape(B * L, V)
    flat_mask = non_pad_mask.reshape(B * L)

    sims = jnp.matmul(flat_targets, embed_normed.T)
    topk_vals, topk_idx = jax.lax.top_k(sims, k)
    soft_targets = jax.nn.softmax(topk_vals / temperature, axis=-1)
    
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    topk_log_probs = jnp.take_along_axis(log_probs, topk_idx, axis=-1)
    
    per_token_losses = -jnp.sum(soft_targets * topk_log_probs, axis=-1)

    num_valid = jnp.sum(flat_mask).clip(min=1)
    loss = jnp.sum(per_token_losses * flat_mask) / num_valid
    
    return loss


#We removed the diversity loss but still use it just for metrics
def calculate_diversity_loss_margin(expected_shared, margin):
    expected_shared = expected_shared.astype(jnp.float32)
    normalized_shared = optax.l2_normalize(expected_shared, axis=-1, eps=1e-8)
    
    dots = jnp.einsum('bsd,btd->bst', normalized_shared, normalized_shared)
    mask = 1.0 - jnp.eye(SHARED_SLOTS)[None, :, :]
    
    violation = jnp.maximum(0.0, jnp.abs(dots) - margin)
    
    return jnp.mean(jnp.square(violation * mask))

@nnx.jit
def train_step(model, opt, batch_tokens, step, prev_hunch=None, should_truncate=False):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        logits, ponder_cost, forget_cost, halt_diag, expected_shared = model(
            inputs, training=True, prev_hunch=prev_hunch
        )
        div_loss = calculate_diversity_loss_margin(expected_shared, margin=0.3)

        non_pad_mask = (targets != PAD_TOKEN_ID)

        token_loss = soft_label_loss(
            logits, targets,
            embed_table=model.embed.embedding.value,
            non_pad_mask=non_pad_mask,
        )

        total_loss = token_loss
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
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
        lambda x: jnp.zeros_like(x),
        lambda x: x,
        next_hunch
    )
    
    return loss, tuple(metrics), next_hunch