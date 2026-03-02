import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import optax
from flax import nnx
import jax.numpy as jnp

LATENT_DIM = 512
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
MAX_STEPS_LIMIT = 16
SHARED_SLOTS = 256  # Context window
OUTPUT_SLOTS = 256  # Output window
MAX_SEQ_LEN = 1024   # Output Tokens window
VOCAB_SIZE = 100277
PAD_TOKEN_ID = 100257
PONDER_LAMBDA = 0.005
TEMP_LAMBDA = 1e-4
HALT_TEMP = 2.5
FORGET_LAMBDA = 1e-4
BUDGET_GATE_SHARPNESS = 10.0
AWAKE_PROB_THRESHOLD = 1e-2

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, num_groups=2, rngs=None, dtype=jnp.float32):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))

        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS + OUTPUT_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

    def __call__(self, x, context=None, mask=None, cache=None, q_pos=None, kv_pos=None):
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
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        q = q * self.scale
        
        sin_kv = self.sin_cached[kv_pos, None, :]
        cos_kv = self.cos_cached[kv_pos, None, :]
        k = (k * cos_kv) + (rotate_half(k) * sin_kv)

        if cache is not None:
            prev_k, prev_v = cache
            k = jnp.concatenate([prev_k, k], axis=1)
            v = jnp.concatenate([prev_v, v], axis=1)
        new_cache = (k, v) if cache is not None else None

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.repeat(k, repeats, axis=2)
        v_expanded = jnp.repeat(v, repeats, axis=2)
        
        out = jax.nn.dot_product_attention(q, k_expanded, v_expanded, mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1])) if mask is not None else None)
        return self.o_proj(out.reshape(b, s, d)), new_cache

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
            rngs=rngs, dtype=dtype
        )

    def __call__(self, x, context=None, mask=None, cache=None, q_pos=None, kv_pos=None, hyper_mods=None):
        attn_out, new_cache = self.attn(self.norm1(x), context=context, mask=mask, cache=cache, q_pos=q_pos, kv_pos=kv_pos)
        x = x + attn_out
        
        mlp_in = self.norm2(x)
        
        if hyper_mods is not None:
            gamma, beta = hyper_mods
            gamma_scaled = jax.nn.tanh(gamma)
            mlp_in = mlp_in * (1.0 + gamma_scaled) + beta

        hidden = jax.nn.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in)
        mlp_out = self.down_proj(hidden)
        
        x = x + mlp_out
        return x, new_cache

class AttentionPooling(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float32):
        self.attention_net = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim // 2, rngs=rngs, dtype=dtype),
            jax.nn.tanh,
            nnx.Linear(latent_dim // 2, 1, rngs=rngs, dtype=dtype)
        )

    def __call__(self, x, mask=None):
        scores = self.attention_net(x)
        
        if mask is not None:
            scores = jnp.where(mask[..., None], scores, -1e9)
            
        attn_weights = jax.nn.softmax(scores, axis=1)
        pooled_output = jnp.sum(attn_weights * x, axis=1)
        
        return pooled_output

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float32):
        self.latent_dim = latent_dim
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        
        self.shared_token = nnx.Param(
            jax.random.normal(rngs(), (1, SHARED_SLOTS, latent_dim)).astype(jnp.float32) * 0.02
        )
        
        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.shared_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)

        self.pooler = AttentionPooling(latent_dim, rngs=rngs, dtype=dtype)
        
        self.hyper_net = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim // 2, rngs=rngs, dtype=dtype),
            jax.nn.gelu,
            nnx.Linear(
                latent_dim // 2, 
                latent_dim * 4, 
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs, 
                dtype=dtype
            )
        )

        self.processor1 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        self.processor2 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

        self.budget_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)

        self.salience_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.salience_head.bias.value = jnp.full((1,), 0.0)
        
        self.halt_pooler = AttentionPooling(latent_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(latent_dim + 1, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), 0.0)

        self.forget_head = nnx.Linear(
            latent_dim,
            latent_dim,
            bias_init=jax.nn.initializers.constant(2.0),
            rngs=rngs,
            dtype=dtype,
        )

    def _get_positions(self, seq_len):
        seq_pos = jnp.arange(seq_len)
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        return seq_pos, shared_pos
        
    def _get_hyper_mods(self, z_seq, mask=None):
        prompt_context = self.pooler(z_seq, mask=mask)
        hyper_out = self.hyper_net(prompt_context)
        
        gamma1, beta1, gamma2, beta2 = jnp.split(hyper_out, 4, axis=-1)
        mods = [x[:, None, :] for x in (gamma1, beta1, gamma2, beta2)]
        
        return (mods[0], mods[1]), (mods[2], mods[3])

    def _get_sliding_divider_masks(self, z_seq, mask=None):
        seq_repr = self.pooler(z_seq, mask=mask)
        reason_ratio = jax.nn.sigmoid(self.budget_head(seq_repr)) 
        divider_pos = reason_ratio * SHARED_SLOTS
        
        indices = jnp.arange(SHARED_SLOTS)
        dist = (divider_pos - indices[None, :]) * BUDGET_GATE_SHARPNESS
        reason_mask = jax.nn.sigmoid(dist)[:, :, None] 
        know_mask = 1.0 - reason_mask 
        
        return reason_mask, know_mask

    def _prepare_reasoning_context(self, tokens, max_steps):
        batch_size, seq_len = tokens.shape
        seq_pos, shared_pos = self._get_positions(seq_len)

        pad_mask = tokens != PAD_TOKEN_ID

        # Fully bidirectional sequence attention for the encoder
        seq_attn_mask = pad_mask[:, None, None, :]

        pad_mask_1d = pad_mask[:, None, None, :]
        memory_mask = jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_)
        extended_ctx_mask = jnp.concatenate([pad_mask_1d, memory_mask], axis=-1)

        z_seq = self.embed(tokens)

        mods1, mods2 = self._get_hyper_mods(z_seq, mask=pad_mask)
        reason_mask, know_mask = self._get_sliding_divider_masks(z_seq, mask=pad_mask)

        h_seq, _ = self.processor1(
            z_seq,
            mask=seq_attn_mask,
            q_pos=seq_pos,
            kv_pos=seq_pos,
            hyper_mods=mods1,
        )
        z_seq, _ = self.processor2(
            h_seq,
            mask=seq_attn_mask,
            q_pos=seq_pos,
            kv_pos=seq_pos,
            hyper_mods=mods2,
        )

        z_shared = jnp.broadcast_to(
            self.shared_token.value, (batch_size, SHARED_SLOTS, self.latent_dim)
        )
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        ctx = {
            "seq_pos": seq_pos,
            "shared_pos": shared_pos,
            "extended_ctx_mask": extended_ctx_mask,
            "mods1": mods1,
            "mods2": mods2,
            "reason_mask": reason_mask,
            "know_mask": know_mask,
            "batch_size": batch_size,
        }
        return z_seq, z_shared, all_time_embeds, ctx

    def _core_reasoning_step(self, curr_seq, curr_shared, t_signal, ctx):
        scaled_t_signal = t_signal[None, None, :] * 0.1
        z_reason_in = curr_shared + scaled_t_signal

        shared_ctx = jnp.concatenate([curr_seq, curr_shared], axis=1)
        shared_kv_pos = jnp.concatenate([ctx["seq_pos"], ctx["shared_pos"]])

        h_reason, _ = self.processor1(
            z_reason_in,
            context=shared_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["shared_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods1"],
        )
        new_reason_raw, _ = self.processor2(
            h_reason,
            context=shared_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["shared_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods2"],
        )
        shared_after_reason = new_reason_raw * ctx["reason_mask"] + curr_shared * (
            1.0 - ctx["reason_mask"]
        )

        z_know_in = shared_after_reason + scaled_t_signal
        know_ctx = jnp.concatenate([curr_seq, shared_after_reason], axis=1)

        h_know, _ = self.processor1(
            z_know_in,
            context=know_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["shared_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods1"],
        )
        new_know_raw, _ = self.processor2(
            h_know,
            context=know_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["shared_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods2"],
        )
        new_shared = new_know_raw * ctx["know_mask"] + shared_after_reason * (
            1.0 - ctx["know_mask"]
        )

        # Sequence refines itself directly using the sequence and new latents
        h_proposed, _ = self.processor1(
            curr_seq,
            context=know_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["seq_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods1"],
        )
        proposed_updates, _ = self.processor2(
            h_proposed,
            context=know_ctx,
            mask=ctx["extended_ctx_mask"],
            q_pos=ctx["seq_pos"],
            kv_pos=shared_kv_pos,
            hyper_mods=ctx["mods2"],
        )

        salience_logits = self.salience_head(curr_seq)
        salience = jax.nn.sigmoid(salience_logits)
        new_seq = curr_seq + salience * (proposed_updates - curr_seq)

        forget = jax.nn.sigmoid(self.forget_head(new_shared))
        new_shared = forget * new_shared + (1.0 - forget) * curr_shared

        latent_shift = jnp.mean(jnp.abs(new_shared - curr_shared), axis=(1, 2))

        halt_pooled = self.halt_pooler(new_shared)
        halt_features = jnp.concatenate([halt_pooled, latent_shift[:, None]], axis=-1)

        halt_logits = self.halt_head(halt_features).squeeze(axis=-1)
        halt_prob = jax.nn.sigmoid(halt_logits * HALT_TEMP)

        return new_seq, new_shared, proposed_updates, salience, halt_prob, forget

    def encode(self, prompt_tokens, max_steps=MAX_STEPS_LIMIT, training=False):
        z_seq, z_shared, all_time_embeds, ctx = self._prepare_reasoning_context(
            prompt_tokens, max_steps
        )

        def scan_step(carry, t_signal):
            curr_seq, curr_shared, p_remain_prev = carry

            def full_compute(_):
                return self._core_reasoning_step(
                    curr_seq, curr_shared, t_signal, ctx
                )

            def skip_compute(_):
                proposed_updates = curr_seq
                salience = jnp.zeros_like(curr_seq[..., :1])
                halt_prob = jnp.ones_like(p_remain_prev)
                forget = jnp.zeros_like(curr_shared)
                return (
                    curr_seq,
                    curr_shared,
                    proposed_updates,
                    salience,
                    halt_prob,
                    forget,
                )

            should_run = jnp.any(p_remain_prev > AWAKE_PROB_THRESHOLD)
            (
                new_seq,
                new_shared,
                proposed_updates,
                salience,
                halt_prob,
                forget,
            ) = jax.lax.cond(should_run, full_compute, skip_compute, operand=None)

            step_temp_loss = jnp.mean(
                (1.0 - salience) * jnp.square(proposed_updates - curr_seq),
                axis=(1, 2),
            )
            step_forget_l1 = jnp.mean(jnp.abs(forget), axis=(1, 2))
            p_remain_next = p_remain_prev * (1.0 - halt_prob)

            new_seq = self.seq_norm(new_seq)
            new_shared = self.shared_norm(new_shared)

            return (
                (new_seq, new_shared, p_remain_next),
                (new_shared, halt_prob, step_temp_loss, step_forget_l1),
            )

        scan_fn = nnx.scan(nnx.remat(scan_step), in_axes=(nnx.Carry, 0))
        p_remain0 = jnp.ones((ctx["batch_size"],), dtype=z_seq.dtype)

        final_state, (all_shared, all_halts, all_temp_loss, all_forget_l1) = scan_fn(
            (z_seq, z_shared, p_remain0), all_time_embeds
        )

        _, final_pondered_shared, _ = final_state

        ponder_cost = jnp.mean(all_halts)
        temporal_loss = jnp.mean(all_temp_loss)
        forget_loss = jnp.mean(all_forget_l1)

        return final_pondered_shared, ponder_cost, temporal_loss, forget_loss

    def decode(self, decoder_inputs, encoded_shared):
        batch_size, seq_len = decoder_inputs.shape
        
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[None, None, :, :]
        
        cross_mask = jnp.ones((batch_size, 1, seq_len, SHARED_SLOTS), dtype=jnp.bool_)

        z_dec = self.embed(decoder_inputs)
        
        seq_pos = jnp.arange(seq_len)
        h_dec, _ = self.processor1(z_dec, context=z_dec, mask=causal_mask, q_pos=seq_pos, kv_pos=seq_pos)
        
        shared_pos = jnp.arange(SHARED_SLOTS)
        out_dec, _ = self.processor2(h_dec, context=encoded_shared, mask=cross_mask, q_pos=seq_pos, kv_pos=shared_pos)
        
        logits = out_dec @ self.embed.embedding.value.T
        return logits

    def __call__(self, prompt_tokens, decoder_inputs=None, max_steps=MAX_STEPS_LIMIT, training=False):
        if decoder_inputs is None:
            decoder_inputs = prompt_tokens

        encoded_shared, ponder_cost, temporal_loss, forget_loss = self.encode(prompt_tokens, max_steps, training)
        logits = self.decode(decoder_inputs, encoded_shared)
        return logits, ponder_cost, temporal_loss, forget_loss

model = UniversalReasoner(LATENT_DIM, rngs=nnx.Rngs(0))

ponder_lambda_schedule = optax.linear_schedule(
    init_value=0.0, 
    end_value=0.005, 
    transition_steps=200
)

temp_lambda_schedule = optax.linear_schedule(
    init_value=1e-4, 
    end_value=1e-5, 
    transition_steps=200
)

forget_lambda_schedule = optax.linear_schedule(
    init_value=1e-5, 
    end_value=1e-4, 
    transition_steps=200
)

schedule = optax.warmup_cosine_decay_schedule(1e-6, 8e-5, 200, 800, 1e-6)
base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), 
    optax.adafactor(learning_rate=schedule, multiply_by_parameter_scale=True)   #dropped adamw and added the multiply parameter
)
optimizer_chain = optax.MultiSteps(base_optimizer, every_k_schedule=ACCUMULATION_STEPS)
optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

@nnx.jit
def train_step(m, opt, batch_tokens, step):

    macro_step = step // ACCUMULATION_STEPS

    curr_ponder_lambda = ponder_lambda_schedule(macro_step)
    curr_temp_lambda = temp_lambda_schedule(macro_step)
    curr_forget_lambda = forget_lambda_schedule(macro_step)

    def loss_fn(model_instance):
        seq_len = batch_tokens.shape[1]
        split_idx = seq_len // 2
        
        prompt_tokens = batch_tokens[:, :split_idx]
        
        decoder_inputs = batch_tokens[:, split_idx-1 : -1]
        labels = batch_tokens[:, split_idx:]
        
        preds, ponder_cost, temporal_cost, forget_cost = model_instance(
            prompt_tokens, decoder_inputs, training=True
        )
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=labels)
        token_loss = jnp.mean(ce_loss, where=(labels != PAD_TOKEN_ID))
        
        temporal_cost_clipped = jnp.clip(jnp.mean(temporal_cost), a_max=10.0)
        total_loss = (
            token_loss
            + curr_ponder_lambda * jnp.mean(ponder_cost)
            + curr_temp_lambda * temporal_cost_clipped
            + curr_forget_lambda * jnp.mean(forget_cost)
        ) / ACCUMULATION_STEPS
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(temporal_cost), jnp.mean(forget_cost))

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
    opt.update(m, grads)
    return loss, aux