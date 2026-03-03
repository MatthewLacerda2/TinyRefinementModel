import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import optax
from flax import nnx
import jax.numpy as jnp

NUM_BLOCKS = 4
LATENT_DIM = 512
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
MAX_STEPS_LIMIT = 16
SHARED_SLOTS = 256
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100277
PAD_TOKEN_ID = 100257
PONDER_LAMBDA = 0.001
TEMP_LAMBDA = 1e-4
HALT_TEMP = 2.5
FORGET_LAMBDA = 5e-5
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
        t = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS)
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

        out = jax.nn.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
            if mask is not None else None,
        )
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
            rngs=rngs, dtype=dtype,
        )

    def __call__(self, x, context=None, mask=None, cache=None, q_pos=None, kv_pos=None, hyper_mods=None):
        attn_out, new_cache = self.attn(self.norm1(x), context=context, mask=mask, cache=cache, q_pos=q_pos, kv_pos=kv_pos)
        x = x + attn_out

        mlp_in = self.norm2(x)
        if hyper_mods is not None:
            gamma, beta = hyper_mods
            mlp_in = mlp_in * (1.0 + jax.nn.tanh(gamma)) + beta

        hidden = jax.nn.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in)
        x = x + self.down_proj(hidden)
        return x, new_cache


class BlockStack(nnx.Module):
    def __init__(self, num_blocks, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.blocks = nnx.List([
            StandardReasoningBlock(latent_dim, num_heads, rngs=rngs, dtype=dtype)
            for _ in range(num_blocks)
        ])
        self.num_blocks = num_blocks

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, hyper_mods_list=None):
        if hyper_mods_list is None:
            mods = [None] * self.num_blocks
        elif isinstance(hyper_mods_list, tuple):
            mods = [hyper_mods_list] * self.num_blocks
        else:
            mods = hyper_mods_list
            assert len(mods) == self.num_blocks

        for block, mod in zip(self.blocks, mods):
            x, _ = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, hyper_mods=mod)
        return x


class AttentionPooling(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float32):
        self.attention_net = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim // 2, rngs=rngs, dtype=dtype),
            jax.nn.tanh,
            nnx.Linear(latent_dim // 2, 1, rngs=rngs, dtype=dtype),
        )

    def __call__(self, x, mask=None):
        scores = self.attention_net(x)
        if mask is not None:
            scores = jnp.where(mask[..., None], scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=1)
        return jnp.sum(attn_weights * x, axis=1)


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32):
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
                latent_dim * 2 * num_blocks,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
                dtype=dtype,
            ),
        )
        self.num_blocks = num_blocks

        self.reason_stack = BlockStack(num_blocks, latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        self.know_stack = BlockStack(num_blocks, latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

        self.budget_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)

        self.budget_temp = nnx.Param(jnp.zeros((1,), dtype=jnp.float32))

        self.halt_pooler = AttentionPooling(latent_dim, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(latent_dim + 1, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), -2.0)

        self.forget_head = nnx.Linear(
            latent_dim, latent_dim,
            bias_init=jax.nn.initializers.constant(2.0),
            rngs=rngs, dtype=dtype,
        )

    def _get_positions(self, seq_len):
        seq_pos = jnp.arange(seq_len)
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        return seq_pos, shared_pos

    def _get_hyper_mods(self, z_seq, mask=None):
        prompt_context = self.pooler(z_seq, mask=mask)
        hyper_out = self.hyper_net(prompt_context)
        chunks = jnp.split(hyper_out, 2 * self.num_blocks, axis=-1)
        return [(chunks[2 * i][:, None, :], chunks[2 * i + 1][:, None, :]) for i in range(self.num_blocks)]

    def _get_sliding_divider_masks(self, z_seq, mask=None):
        seq_repr = self.pooler(z_seq, mask=mask)
        reason_ratio = jax.nn.sigmoid(self.budget_head(seq_repr))
        normalized_indices = jnp.arange(SHARED_SLOTS) / SHARED_SLOTS
        raw_dist = reason_ratio - normalized_indices[None, :]
        sharpness = jnp.exp(self.budget_temp.value) 
        reason_mask = jax.nn.sigmoid(raw_dist * sharpness)[:, :, None]
        know_mask = 1.0 - reason_mask
        return reason_mask, know_mask

    def _prepare_reasoning_context(self, tokens, max_steps):
        batch_size, seq_len = tokens.shape
        seq_pos, shared_pos = self._get_positions(seq_len)

        pad_mask = tokens != PAD_TOKEN_ID

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        seq_attn_mask = pad_mask[:, None, None, :] & causal_mask[None, None, :, :]

        memory_mask = jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_)
        extended_ctx_mask = jnp.concatenate([pad_mask[:, None, None, :], memory_mask], axis=-1)

        z_seq = self.embed(tokens)

        hyper_mods = self._get_hyper_mods(z_seq, mask=pad_mask)
        reason_mask, know_mask = self._get_sliding_divider_masks(z_seq, mask=pad_mask)

        z_seq = self.know_stack(z_seq, mask=seq_attn_mask, q_pos=seq_pos, kv_pos=seq_pos, hyper_mods_list=hyper_mods)

        z_shared = jnp.broadcast_to(self.shared_token.value, (batch_size, SHARED_SLOTS, self.latent_dim))
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        ctx = {
            'seq_pos': seq_pos, 'shared_pos': shared_pos,
            'seq_attn_mask': seq_attn_mask,
            'extended_ctx_mask': extended_ctx_mask,
            'hyper_mods': hyper_mods,
            'reason_mask': reason_mask, 'know_mask': know_mask,
            'batch_size': batch_size,
        }
        return z_seq, z_shared, all_time_embeds, ctx

    def _core_reasoning_step(self, curr_seq, curr_shared, t_signal, ctx):
        scaled_t = t_signal[None, None, :] * 0.1
        hyper_mods = ctx['hyper_mods']

        shared_ctx = jnp.concatenate([curr_seq, curr_shared], axis=1)
        shared_kv_pos = jnp.concatenate([ctx['seq_pos'], ctx['shared_pos']])

        new_reason_raw = self.reason_stack(
            curr_shared + scaled_t,
            context=shared_ctx,
            mask=ctx['extended_ctx_mask'],
            q_pos=ctx['shared_pos'],
            kv_pos=shared_kv_pos,
            hyper_mods_list=hyper_mods,
        )
        shared_after_reason = new_reason_raw * ctx['reason_mask'] + curr_shared * (1.0 - ctx['reason_mask'])

        know_ctx = jnp.concatenate([curr_seq, shared_after_reason], axis=1)
        new_know_raw = self.know_stack(
            shared_after_reason + scaled_t,
            context=know_ctx,
            mask=ctx['extended_ctx_mask'],
            q_pos=ctx['shared_pos'],
            kv_pos=shared_kv_pos,
            hyper_mods_list=hyper_mods,
        )
        new_shared = new_know_raw * ctx['know_mask'] + shared_after_reason * (1.0 - ctx['know_mask'])

        forget = jax.nn.sigmoid(self.forget_head(new_shared))
        new_shared = forget * new_shared + (1.0 - forget) * curr_shared

        latent_shift = jnp.mean(jnp.abs(new_shared - curr_shared), axis=(1, 2))
        halt_pooled = self.halt_pooler(new_shared)
        halt_features = jnp.concatenate([halt_pooled, latent_shift[:, None]], axis=-1)
        halt_prob = jax.nn.sigmoid(self.halt_head(halt_features).squeeze(-1) * HALT_TEMP)

        return new_shared, halt_prob, forget

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False):
        z_seq, z_shared, all_time_embeds, ctx = self._prepare_reasoning_context(tokens, max_steps)

        def scan_step(carry, t_signal):
            curr_shared, p_remain_prev = carry

            def full_compute(_):
                return self._core_reasoning_step(z_seq, curr_shared, t_signal, ctx)

            def skip_compute(_):
                return curr_shared, jnp.ones_like(p_remain_prev), jnp.zeros_like(curr_shared)

            should_run = jnp.any(p_remain_prev > AWAKE_PROB_THRESHOLD)
            new_shared, halt_prob, forget = jax.lax.cond(
                should_run, full_compute, skip_compute, operand=None
            )

            step_forget_l1 = jnp.mean(jnp.abs(forget), axis=(1, 2))
            p_remain_next = p_remain_prev * (1.0 - halt_prob)
            new_shared = self.shared_norm(new_shared)

            return (new_shared, p_remain_next), (halt_prob, step_forget_l1)

        scan_fn = nnx.scan(nnx.remat(scan_step), in_axes=(nnx.Carry, 0))
        p_remain0 = jnp.ones((ctx['batch_size'],), dtype=z_seq.dtype)
        (final_shared, _), (all_halts, all_forget_l1) = scan_fn(
            (z_shared, p_remain0), all_time_embeds
        )

        all_halts = jnp.clip(all_halts, 0.0, 1.0 - 1e-7)
        p_remain = jnp.concatenate(
            [jnp.ones((1, ctx['batch_size'])), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0
        )
        step_weights = all_halts * p_remain
        step_weights = step_weights.at[-1].add(p_remain[-1] * (1.0 - all_halts[-1]))

        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0)
        forget_loss = jnp.sum(step_weights * all_forget_l1, axis=0)

        shared_kv_pos = jnp.concatenate([ctx['seq_pos'], ctx['shared_pos']])
        cross_mask = jnp.ones((ctx['batch_size'], 1, z_seq.shape[1], SHARED_SLOTS), dtype=jnp.bool_)
        extended_cross_mask = jnp.concatenate([ctx['seq_attn_mask'], cross_mask], axis=-1)

        final_ctx = jnp.concatenate([z_seq, final_shared], axis=1)
        z_out = self.reason_stack(
            z_seq,
            context=final_ctx,
            mask=extended_cross_mask,
            q_pos=ctx['seq_pos'],
            kv_pos=shared_kv_pos,
            hyper_mods_list=ctx['hyper_mods'],
        )
        z_out = self.seq_norm(z_out)

        logits = z_out @ self.embed.embedding.value.T
        return logits, ponder_cost, forget_loss


model = UniversalReasoner(LATENT_DIM, rngs=nnx.Rngs(0), num_blocks=NUM_BLOCKS)

ponder_lambda_schedule = optax.linear_schedule(init_value=0.0, end_value=0.001, transition_steps=200)
forget_lambda_schedule = optax.linear_schedule(init_value=0.0, end_value=5e-5, transition_steps=200)

schedule = optax.warmup_cosine_decay_schedule(1e-6, 1.5e-4, 200, 800, 5e-6)
base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adafactor(learning_rate=schedule, multiply_by_parameter_scale=True),
)
optimizer_chain = optax.MultiSteps(base_optimizer, every_k_schedule=ACCUMULATION_STEPS)
optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)


@nnx.jit
def train_step(m, opt, batch_tokens, step):
    macro_step = step // ACCUMULATION_STEPS
    curr_ponder_lambda = ponder_lambda_schedule(macro_step)
    curr_forget_lambda = forget_lambda_schedule(macro_step)

    def loss_fn(model_instance):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        preds, ponder_cost, forget_cost = model_instance(inputs, training=True)

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        token_loss = jnp.mean(ce_loss, where=(targets != PAD_TOKEN_ID))

        total_loss = (
            token_loss
            + curr_ponder_lambda * jnp.mean(ponder_cost)
            + curr_forget_lambda * jnp.mean(forget_cost)
        ) / ACCUMULATION_STEPS
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost))

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
    opt.update(m, grads)
    return loss, aux