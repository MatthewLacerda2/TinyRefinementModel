import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import optax
from flax import nnx
import jax.numpy as jnp

NUM_BLOCKS = 8
LATENT_DIM = 512
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
MIN_STEPS = 4
MAX_STEPS_LIMIT = 16
SHARED_SLOTS = 512
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100277
PAD_TOKEN_ID = 100257
PONDER_LAMBDA = 1e-4
TEMP_LAMBDA = 1e-4
HALT_TEMP = 0.5
FORGET_LAMBDA = 5e-5
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

        self.cache = nnx.Cache(None)

        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

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

        out = jax.nn.dot_product_attention(
            q, k_expanded, v_expanded,
            mask=jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, q.shape[1], k_expanded.shape[1]))
            if mask is not None else None,
        )
        return self.o_proj(out.reshape(b, s, d))


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

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        attn_out = self.attn(self.norm1(x), context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache)
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
            block.attn.cache.value = None

    def __call__(self, x, context=None, mask=None, q_pos=None, kv_pos=None, use_cache=False):
        for block in self.blocks:
            x = block(x, context=context, mask=mask, q_pos=q_pos, kv_pos=kv_pos, use_cache=use_cache)
        return x


class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True):
        self.latent_dim = latent_dim

        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.random.normal(rngs(), (1, SHARED_SLOTS, latent_dim)).astype(jnp.float32) * 0.02
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.shared_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)

        self.num_blocks = num_blocks

        self.stack = BlockStack(num_blocks, latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

        halt_pre_dim = latent_dim // 4
        self.halt_pre = nnx.Linear(
            latent_dim, halt_pre_dim, 
            kernel_init=nnx.initializers.variance_scaling(0.02, mode="fan_in", distribution="normal"),
            rngs=rngs, dtype=dtype
        )
        self.halt_head = nnx.Linear(halt_pre_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias.value = jnp.full((1,), -2.5)  # Slightly stronger negative bias for conservatism

        self.use_forget = use_forget
        if self.use_forget:
            self.forget_head = nnx.Linear(
                latent_dim, latent_dim,
                bias_init=jax.nn.initializers.constant(2.0),
                rngs=rngs, dtype=dtype,
            )

    def _get_positions(self, seq_len):
        seq_pos = jnp.arange(seq_len)
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        return seq_pos, shared_pos

    def _prepare_reasoning_context(self, tokens, max_steps):
        batch_size, seq_len = tokens.shape
        seq_pos, shared_pos = self._get_positions(seq_len)

        pad_mask = tokens != PAD_TOKEN_ID

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        seq_attn_mask = pad_mask[:, None, None, :] & causal_mask[None, None, :, :]

        memory_mask = jnp.ones((batch_size, 1, 1, SHARED_SLOTS), dtype=jnp.bool_)
        extended_ctx_mask = jnp.concatenate([pad_mask[:, None, None, :], memory_mask], axis=-1)

        z_seq = self.embed(tokens)

        z_seq = self.stack(z_seq, mask=seq_attn_mask, q_pos=seq_pos, kv_pos=seq_pos)

        # Initialize shared memory from token context (one-time expensive step)
        z_shared = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        init_ctx = jnp.concatenate([z_seq, z_shared], axis=1)
        shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])

        z_shared = self.stack(
            z_shared,
            context=init_ctx,
            mask=extended_ctx_mask,
            q_pos=shared_pos,
            kv_pos=shared_kv_pos,
            use_cache=False,
        )
        
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        ctx = {
            'seq_pos': seq_pos, 'shared_pos': shared_pos,
            'seq_attn_mask': seq_attn_mask,
            'extended_ctx_mask': extended_ctx_mask,
            'batch_size': batch_size,
            'z_seq': z_seq,
        }
        return z_seq, z_shared, all_time_embeds, ctx

    def _core_reasoning_step(self, curr_seq, curr_shared, t_signal, ctx, awake_mask):
        scaled_t = t_signal[None, None, :] * 0.1

        shared_ctx = jnp.concatenate([curr_seq, curr_shared], axis=1)
        shared_kv_pos = jnp.concatenate([ctx['seq_pos'], ctx['shared_pos']])

        stack_input = curr_shared + scaled_t
        stack_input = curr_shared + awake_mask * (stack_input - curr_shared)

        new_shared_raw = self.stack(
            stack_input,
            context=shared_ctx,
            mask=ctx['extended_ctx_mask'],
            q_pos=ctx['shared_pos'],
            kv_pos=shared_kv_pos,
            use_cache=False,
        )
        new_shared = curr_shared + awake_mask * (new_shared_raw - curr_shared)

        if self.use_forget:
            forget = jax.nn.sigmoid(self.forget_head(new_shared))
            new_shared = forget * new_shared + (1.0 - forget) * curr_shared
        else:
            forget = jnp.ones_like(new_shared)  # No-op

        pooled = jnp.mean(new_shared, axis=1)
        pre = jax.nn.gelu(self.halt_pre(pooled))
        halt_logits = self.halt_head(pre).squeeze(-1)
        halt_prob = jax.nn.sigmoid(halt_logits / HALT_TEMP)

        return new_shared, halt_prob, forget

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False):
        self.stack.reset_state()

        z_seq, z_shared, all_time_embeds, ctx = self._prepare_reasoning_context(tokens, max_steps)

        shared_norm_scale = self.shared_norm.scale.value

        graphdef, state = nnx.split(self)

        def scan_step(carry, inputs):
            curr_shared, p_remain_prev = carry
            t_signal, step_id = inputs

            awake_mask = p_remain_prev[:, None, None]

            m = nnx.merge(graphdef, state)
            computed_new_shared, halt_prob, forget = m._core_reasoning_step(z_seq, curr_shared, t_signal, ctx, awake_mask)

            candidate_shared = computed_new_shared

            halt_prob = jnp.where(step_id < MIN_STEPS, 0.0, halt_prob)

            new_shared = jnp.where(
                p_remain_prev[:, None, None] > 0,
                candidate_shared,
                curr_shared
            )

            step_forget_l1 = jnp.mean(jnp.abs(forget), axis=(1, 2)) if self.use_forget else jnp.zeros((ctx['batch_size'],))
            p_remain_next = p_remain_prev * (1.0 - halt_prob)

            rms = jnp.sqrt(jnp.mean(new_shared ** 2, axis=-1, keepdims=True) + 1e-6)
            new_shared = (new_shared / rms) * shared_norm_scale

            return (new_shared, p_remain_next), (new_shared, halt_prob, step_forget_l1)

        p_remain0 = jnp.ones((ctx['batch_size'],), dtype=z_seq.dtype)

        step_ids = jnp.arange(max_steps)

        (final_shared, _), (all_shared, all_halts, all_forget_l1) = jax.lax.scan(
            jax.checkpoint(scan_step),
            (z_shared, p_remain0),
            (all_time_embeds, step_ids),
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
        cross_mask = jnp.ones(
            (ctx['batch_size'], 1, z_seq.shape[1], SHARED_SLOTS),
            dtype=jnp.bool_
        )
        extended_cross_mask = jnp.concatenate([ctx['seq_attn_mask'], cross_mask], axis=-1)

        final_ctx = jnp.concatenate([z_seq, final_shared], axis=1)
        z_out = self.stack(
            z_seq,
            context=final_ctx,
            mask=extended_cross_mask,
            q_pos=ctx['seq_pos'],
            kv_pos=shared_kv_pos,
            use_cache=False,
        )
        z_out = self.seq_norm(z_out)

        logits = z_out @ self.embed.embedding.value.T
        return logits, ponder_cost, forget_loss


model = UniversalReasoner(LATENT_DIM, rngs=nnx.Rngs(0), num_blocks=NUM_BLOCKS)

schedule = optax.warmup_cosine_decay_schedule(1e-6, 1.5e-4, 200, 800, 5e-6)

base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adafactor(learning_rate=schedule, multiply_by_parameter_scale=True),
)

optimizer_chain = optax.MultiSteps(base_optimizer, every_k_schedule=ACCUMULATION_STEPS)
optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)


@nnx.jit
def train_step(model, opt, batch_tokens, p_lambda, f_lambda):
    def loss_fn(model):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]

        preds, ponder_cost, forget_cost = model(inputs, training=True)

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        token_loss = jnp.mean(ce_loss, where=(targets != PAD_TOKEN_ID))

        total_loss = (
            token_loss
            + p_lambda * jnp.mean(ponder_cost)
            + f_lambda * jnp.mean(forget_cost)
        ) / ACCUMULATION_STEPS
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(forget_cost))

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    opt.update(model, grads)
    return loss, aux