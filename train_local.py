import jax
import optax
from flax import nnx, struct
import jax.numpy as jnp
from typing import Dict, Any

#Keep (most) values powers of 2 if you know what's good for you

#Params
LATENT_DIM = 512
NUM_BLOCKS = 4
SHARED_SLOTS = 32
MAX_SEQ_LEN = 512
VOCAB_SIZE = 100352

#Training
MAX_STEPS_LIMIT = 16
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100257

NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4

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
        
        self.encoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=True)
        self.decoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=True)
        self.reasoning_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False)

        self.meta_proj = nnx.Linear(2, latent_dim, rngs=rngs, dtype=dtype)
        
        self.time_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.forget_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        self.hunch_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.hunch_gate = nnx.Linear(
            latent_dim, latent_dim,
            bias_init=jax.nn.initializers.constant(-2.0),
            rngs=rngs, dtype=dtype,
        )

        self.raw_tau = nnx.Param(jnp.array(-2.3))

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

        modules = (
            self.meta_proj, self.time_norm, self.time_signal_norm, 
            self.reasoning_stack,
            self.forget_norm if self.use_forget else None,
            self.forget_head if self.use_forget else None,
            self.raw_tau
        )
        model_graph, model_state = nnx.split(modules)

        def scan_step(carry, inputs):
            curr_shared, prev_forget, prev_div, current_state = carry
            t_signal, step_id = inputs
            
            (
                m_proj, t_norm, ts_norm, 
                r_stack,
                f_norm, f_head, raw_tau_param
            ) = nnx.merge(model_graph, current_state)
            
            meta_input = jnp.stack([prev_forget, prev_div], axis=-1)
            meta_signal = m_proj(meta_input)[:, None, :]
            
            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

            stack_input = t_norm(curr_shared) + ts_norm(t_signal[None, None, :]) + meta_signal
            new_shared = r_stack(stack_input, context=shared_ctx, mask=extended_ctx_bias, q_pos=shared_pos, kv_pos=shared_kv_pos, is_causal=False)

            if self.use_forget:
                forget = jax.nn.sigmoid(f_head(f_norm(new_shared)))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = jnp.zeros((batch_size,))

            tau = jax.nn.softplus(raw_tau_param.value) + 1e-4

            step_div = calculate_infonce_loss(new_shared, curr_shared, tau)

            _, next_state = nnx.split((m_proj, t_norm, ts_norm, r_stack, f_norm, f_head, raw_tau_param))

            return (new_shared, forget_val, step_div, next_state), ScanStepOutput(
                shared_state=new_shared,
                forget_val=forget_val,
                step_div=step_div
            )

        init_carry = (
            z_shared, 
            jnp.zeros((batch_size,)),
            jnp.zeros((batch_size,)),
            model_state
        )
        
        final_carry_all, all_outputs = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry, (all_time_embeds, jnp.arange(max_steps))
        )

        final_carry = (final_carry_all[0],)
        return final_carry, all_outputs, shared_pos



    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True):
        batch_size = tokens.shape[0]
        self.encoder_stack.reset_state()
        self.reasoning_stack.reset_state()
        self.decoder_stack.reset_state()
        
        z_seq, pad_mask, seq_pos = self._encode_sequence(tokens)

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        
        def get_fresh():
            return z_shared_base
            
        def get_carried():
            current_hunch = self.hunch_cache.value
            gate = jax.nn.sigmoid(self.hunch_gate(self.hunch_norm(current_hunch)))
            return gate * current_hunch + (1.0 - gate) * z_shared_base
        
        z_shared = jax.lax.cond(should_refresh, get_fresh, get_carried)

        past_shared_pos = jnp.arange(-SHARED_SLOTS, 0)
        
        decoder_kv_pos = jnp.concatenate([seq_pos, past_shared_pos], axis=0)
        
        decoder_pad_mask = jnp.concatenate([pad_mask, jnp.ones((batch_size, SHARED_SLOTS), dtype=jnp.bool_)], axis=1)
        decoder_bias = (decoder_pad_mask.astype(jnp.float32) - 1.0) * 1e9
        decoder_bias = decoder_bias[:, None, None, :]

        final_carry, all_outputs, shared_pos = self._reasoning_loop(z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared)
        expected_shared = final_carry[0]
        
        decoder_ctx_final = jnp.concatenate([z_seq, expected_shared], axis=1)
        
        z_seq_out = self.decoder_stack(
            z_seq, 
            context=decoder_ctx_final, 
            mask=decoder_bias, 
            q_pos=seq_pos, 
            kv_pos=decoder_kv_pos,
            is_causal=True
        )
        logits = self.seq_norm(z_seq_out) @ self.embed.embedding.value.T
        
        total_f_cost = jnp.mean(jnp.sum(all_outputs.forget_val, axis=0))
        total_div_cost = jnp.mean(jnp.sum(all_outputs.step_div, axis=0))
        
        states = all_outputs.shared_state
        diffs = states[1:] - states[:-1]
        temporal_drift = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(diffs), axis=-1) + 1e-8))
        
        halt_diag = {
            'temporal_drift': temporal_drift,
            'forget_density': jnp.mean(all_outputs.forget_val),
            'tau': jax.nn.softplus(self.raw_tau.value) + 1e-4,
        }
        
        self.hunch_cache.value = expected_shared

        return ReasonerOutput(
            logits=logits, forget_cost=total_f_cost,
            diversity_loss=total_div_cost,
            halt_diag=halt_diag, expected_shared=expected_shared
        )

def calculate_infonce_loss(new_shared, curr_shared, tau):
    b, s, d = new_shared.shape
    
    anchor = new_shared / jnp.sqrt(jnp.sum(jnp.square(new_shared), axis=-1, keepdims=True) + 1e-5)
    positive = jax.lax.stop_gradient(
        curr_shared / jnp.sqrt(jnp.sum(jnp.square(curr_shared), axis=-1, keepdims=True) + 1e-5)
    )
    
    pos_logits = jnp.sum(anchor * positive, axis=-1, keepdims=True) / tau
    
    neg_logits = jnp.einsum('bsd,btd->bst', anchor, anchor, precision=jax.lax.Precision.HIGHEST) / tau
    
    identity = jnp.eye(s)[None, :, :]
    neg_logits = neg_logits + (identity * -1e9)
    logits = jnp.concatenate([pos_logits, neg_logits], axis=-1)
    
    labels = jnp.zeros((b, s), dtype=jnp.int32)
    
    loss_per_slot = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    
    return jnp.mean(loss_per_slot, axis=-1)

@nnx.jit
def compute_grad_step(model, batch_tokens, step, should_truncate=False):
    should_refresh = jnp.any(should_truncate).squeeze()

    def loss_fn(model):
        def compute_ce(logits, targets):
            mask = targets != PAD_TOKEN_ID
            return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets) * mask) / jnp.sum(mask).clip(min=1)

        seq1_in, seq1_out = batch_tokens[:, :MAX_SEQ_LEN], batch_tokens[:, 1:MAX_SEQ_LEN+1]
        seq2_in, seq2_out = batch_tokens[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN], batch_tokens[:, MAX_SEQ_LEN+1:2*MAX_SEQ_LEN+1]

        out1 = model(seq1_in, training=True, should_refresh=should_refresh)
        ce1 = compute_ce(out1.logits, seq1_out)
        
        out2 = model(seq2_in, training=True, should_refresh=False)
        ce2 = compute_ce(out2.logits, seq2_out)

        refinement_regression = jnp.maximum(0.0, ce2 - ce1) 
        refinement_loss = refinement_regression * 0.08
        
        early_penalty = ce1 * 0.03

        from schedules import forget_lambda_schedule, diversity_lambda_schedule
        opt_step = step // ACCUMULATION_STEPS
        f_lambda = forget_lambda_schedule(opt_step)
        d_lambda = diversity_lambda_schedule(opt_step)

        total_loss = (ce1 + ce2) \
                     + f_lambda * (out1.forget_cost + out2.forget_cost) \
                     + d_lambda * (out1.diversity_loss + out2.diversity_loss) \
                     + refinement_loss \
                     + early_penalty

        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        new_halt_diag = {
            **out2.halt_diag,
            'ce1': jax.lax.stop_gradient(ce1),
            'token_loss': jax.lax.stop_gradient(ce2),
        }
        out2 = out2.replace(logits=None, halt_diag=new_halt_diag)
        return total_loss, out2

    (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    current_hunch = model.hunch_cache.value
    cleared_hunch = jnp.zeros_like(current_hunch)
    
    # After the step, we carry forward the hunch UNLESS a truncation was requested
    carried_hunch = jax.lax.cond(
        should_refresh,
        lambda: jax.lax.stop_gradient(cleared_hunch),
        lambda: jax.lax.stop_gradient(current_hunch)
    )
    
    model.hunch_cache.value = carried_hunch

    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))
    
    return loss, out, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)