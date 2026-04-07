from flax import linen as nn
import jax
import optax
from flax import nnx
import jax.numpy as jnp
from schedulers import (
    ponder_lambda_schedule, 
    forget_lambda_schedule, 
    diversity_lambda_schedule
)

#Params
LATENT_DIM = 512
NUM_BLOCKS = 4
SHARED_SLOTS = 32
VOCAB_SIZE = 100352
MAX_STEPS_LIMIT = 16

#Training
MAX_SEQ_LEN = 512
MIN_STEPS = 4
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100351

#Soft label settings
SOFT_LABEL_K = 64
SOFT_LABEL_TEMP = 0.1
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4 #Ideal ratio is 4:1


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

        self.cache = nnx.Cache(None)

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=dtype)
        
        self.o_proj = nnx.Linear(
            in_features, in_features, 
            kernel_init=jax.nn.initializers.zeros, 
            rngs=rngs, dtype=dtype
        )

        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=dtype)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=dtype)

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False, is_causal=False):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        kv_input = x
        if context is not None:
            kv_input = jnp.concatenate([x, context], axis=1)

        s_kv = kv_input.shape[1]

        k = self.k_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)
        v = self.v_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

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
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True):
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
        z_seq_raw = self.embed(tokens)

        z_seq = self.main_stack(z_seq_raw, mask=pad_mask[:, None, None, :], q_pos=seq_pos, kv_pos=seq_pos, is_causal=True)

        z_shared_base = jnp.broadcast_to(self.shared_token.value, (batch_size, SHARED_SLOTS, self.latent_dim))

        if current_hunch is not None:
            #We use the previous hidden state (hunch) to seed the current thinking process
            hunch_input = jnp.concatenate([self.hunch_norm(current_hunch), jnp.broadcast_to(self.hunch_norm(z_shared_base), current_hunch.shape)], axis=-1)
            gate = jax.nn.sigmoid(self.hunch_gate(hunch_input))
            z_shared = gate * current_hunch + (1.0 - gate) * z_shared_base
        else:
            z_shared = z_shared_base

        all_time_embeds = self.time_embed(jnp.arange(max_steps))
        
        seq_mask = jnp.broadcast_to(pad_mask[:, None, None, :], (batch_size, 1, SHARED_SLOTS, seq_len))
        shared_mask = jnp.ones((SHARED_SLOTS, SHARED_SLOTS), dtype=jnp.bool_)
        shared_mask = jnp.broadcast_to(shared_mask[None, None, :, :], (batch_size, 1, SHARED_SLOTS, SHARED_SLOTS))
        
        c_steps = max_steps // 2

        def scan_step_fn(m, carry, inputs):
            curr_shared, p_remain_prev = carry
            t_signal, step_id = inputs
            
            is_observation = (step_id >= c_steps)
            
            phase_seq_mask = jnp.where(is_observation, seq_mask, jnp.bool_(False))

            shared_kv_pos = jnp.concatenate([shared_pos, seq_pos])
            phase_extended_mask = jnp.concatenate([shared_mask, phase_seq_mask], axis=-1)

            stack_input = m.time_norm(curr_shared) + m.time_signal_norm(t_signal[None, None, :])
            
            new_shared = m.main_stack(
                stack_input, context=z_seq, mask=phase_extended_mask,
                q_pos=shared_pos, kv_pos=shared_kv_pos
            )

            if m.use_forget:
                gate_context = jnp.concatenate([curr_shared, new_shared], axis=-1)
                forget_gate_input = m.forget_norm(gate_context)
                forget = jax.nn.sigmoid(m.forget_head(forget_gate_input))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = 0.0
            
            pooled = jnp.mean(new_shared, axis=1)
            halt_logits_raw = m.halt_head(jax.nn.gelu(m.halt_pre(pooled))).squeeze(-1)
            halt_logits = 15.0 * jnp.tanh(halt_logits_raw / 15.0)
            halt_prob = jax.nn.sigmoid(halt_logits)
            
            halt_prob = jnp.where((step_id < MIN_STEPS) | (~is_observation), 0.0, halt_prob)
            
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            return (new_shared, p_remain_next), (new_shared, halt_prob, forget_val, halt_logits)

        graphdef, state = nnx.split(self)
        
        @jax.checkpoint
        def pure_step(carry, inputs, state):
            m = nnx.merge(graphdef, state)
            res_carry, res_out = scan_step_fn(m, carry, inputs)
            _, next_state = nnx.split(m)
            return res_carry, res_out, next_state

        curr_carry = (z_shared, jnp.ones((batch_size,)))
        curr_state = state
        step_outputs = []
        
        for t in range(max_steps):
            t_inputs = (all_time_embeds[t], jnp.array(t, dtype=jnp.int32))
            curr_carry, out, curr_state = pure_step(curr_carry, t_inputs, curr_state)
            step_outputs.append(out)

        nnx.update(self, curr_state)
        final_shared, _ = curr_carry
        
        all_shared, all_halts, all_forget_l1, all_logits = jax.tree.map(lambda *xs: jnp.stack(xs), *step_outputs)

        all_halts = jnp.clip(all_halts, 0.0, 1.0 - 1e-7)
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        
        step_weights = all_halts * p_remain
        last_step_extra = p_remain[-1] * (1.0 - all_halts[-1])
        step_weights = step_weights.at[-1].add(last_step_extra)

        past_wisdom = all_shared[c_steps - 1]

        weights_for_shared = step_weights[:, :, None, None]
        expected_shared = jnp.sum(weights_for_shared * all_shared, axis=0)

        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        actual_steps = jnp.sum(step_weights * step_indices, axis=0) 
        ponder_cost = jnp.sum(step_weights * jnp.maximum(0, step_indices - MIN_STEPS), axis=0)
        
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

        q_idx = jnp.arange(seq_len)
        causal_mask = q_idx[:, None] >= q_idx[None, :]
        
        seq_logits_mask = causal_mask[None, :, :] & pad_mask[:, None, :]
        
        thoughts_mask = jnp.ones((batch_size, seq_len, SHARED_SLOTS), dtype=jnp.bool_)
        
        joint_kv_mask = jnp.concatenate([seq_logits_mask, thoughts_mask], axis=-1)
        
        joint_kv_mask = joint_kv_mask[:, None, :, :]

        z_out = self.main_stack(
            z_seq, 
            context=past_wisdom, 
            mask=joint_kv_mask, 
            q_pos=seq_pos, 
            kv_pos=jnp.concatenate([seq_pos, shared_pos]),
            is_causal=False
        )
        
        logits_raw = self.seq_norm(z_out) @ self.embed.embedding.value.T
        logits = 30.0 * jnp.tanh(logits_raw / 30.0)
        
        if not training:
            self.hunch_cache.value = expected_shared
            
        return logits, ponder_cost, jnp.sum(step_weights * all_forget_l1, axis=0), halt_diag, expected_shared

def soft_label_loss(logits, targets, embed_table, non_pad_mask, k=SOFT_LABEL_K, temperature=SOFT_LABEL_TEMP):
    B, L, V = logits.shape
    
    embed_table_gradless = jax.lax.stop_gradient(embed_table)
    norm = jnp.linalg.norm(embed_table_gradless, axis=-1, keepdims=True)
    embed_normed = embed_table_gradless / (norm + 1e-8)
    target_emb = embed_normed[targets] 

    flat_targets = target_emb.reshape(B * L, -1)
    
    sims = jnp.matmul(flat_targets, embed_normed.T)
    topk_vals, topk_idx = jax.lax.top_k(sims, k)
    
    topk_vals = topk_vals.astype(jnp.float32)
    soft_targets = jax.nn.softmax(topk_vals / temperature, axis=-1)
    
    logits_fp32 = logits.astype(jnp.float32).reshape(B * L, V)
    lse = jax.nn.logsumexp(logits_fp32, axis=-1, keepdims=True)
    topk_logits = jnp.take_along_axis(logits_fp32, topk_idx, axis=-1)
    topk_log_probs = topk_logits - lse
    
    per_token_losses = -jnp.sum(soft_targets * topk_log_probs, axis=-1)

    flat_mask = non_pad_mask.reshape(B * L)
    num_valid = jnp.sum(flat_mask).clip(min=1)
    loss = jnp.sum(per_token_losses * flat_mask) / num_valid
    
    return loss



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

        logits, ponder_cost, forget_cost, halt_diag, expected_shared = model(
            inputs, training=True, prev_hunch=prev_hunch
        )
        div_loss = calculate_diversity_loss_margin(expected_shared, margin=0.3)

        non_pad_mask = (targets != PAD_TOKEN_ID)

        p_lambda = ponder_lambda_schedule(step)
        f_lambda = forget_lambda_schedule(step)
        d_lambda = diversity_lambda_schedule(step)

        token_loss = soft_label_loss(
            logits, targets,
            embed_table=model.embed.embedding.value,
            non_pad_mask=non_pad_mask,
        )

        total_loss = token_loss + (p_lambda * jnp.mean(ponder_cost)) + \
                     (f_lambda * jnp.mean(forget_cost)) + (d_lambda * div_loss)
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        halt_diag.update({
            'diversity_loss': div_loss,
            'p_lambda': p_lambda,
            'f_lambda': f_lambda,
            'd_lambda': d_lambda
        })
        
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost), halt_diag, expected_shared)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    *metrics, next_hunch = aux
    next_hunch = jax.vmap(lambda m, h: jnp.where(m, jnp.zeros_like(h), h))(
        should_truncate, next_hunch
    )
    
    return loss / ACCUMULATION_STEPS, tuple(metrics), next_hunch, grads

@nnx.jit
def apply_grads(optimizer, grads, model):
    optimizer.update(grads, model)