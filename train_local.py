import os
import jax
import optax
from flax import nnx
import jax.numpy as jnp

LATENT_DIM = 384
MAX_STEPS_LIMIT = 16
ACCUMULATION_STEPS = 64
SHARED_SLOTS = 256
OUTPUT_SLOTS = 256
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
VOCAB_SIZE = 100277
PAD_TOKEN_ID = 100257
PONDER_LAMBDA = 0.005
TEMP_LAMBDA = 1e-5
HALT_TEMP = 5.0 
BUDGET_GATE_SHARPNESS = 10.0

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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
        
        sin_kv = self.sin_cached[kv_pos, None, :]
        cos_kv = self.cos_cached[kv_pos, None, :]
        k = (k * cos_kv) + (rotate_half(k) * sin_kv)

        if cache is not None:
            prev_k, prev_v = cache
            k = jnp.concatenate([prev_k, k], axis=1)
            v = jnp.concatenate([prev_v, v], axis=1)
        new_cache = (k, v) if cache is not None else None

        repeats = self.num_heads // self.num_groups
        k_expanded = jnp.broadcast_to(
            k[:, :, :, None, :], 
            (b, k.shape[1], self.num_groups, repeats, self.head_dim)
        ).reshape(b, k.shape[1], self.num_heads, self.head_dim)

        v_expanded = jnp.broadcast_to(
            v[:, :, :, None, :], 
            (b, k.shape[1], self.num_groups, repeats, self.head_dim)
        ).reshape(b, k.shape[1], self.num_heads, self.head_dim)

        out = jax.nn.dot_product_attention(q, k_expanded, v_expanded, mask=mask)
        return self.o_proj(out.reshape(b, s, d)), new_cache

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=2, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        
        self.mlp_fc1 = nnx.Linear(latent_dim, latent_dim * 4, rngs=rngs, dtype=dtype)
        self.mlp_fc2 = nnx.Linear(latent_dim * 4, latent_dim, rngs=rngs, dtype=dtype)

    def __call__(self, x, context=None, mask=None, cache=None, q_pos=None, kv_pos=None, hyper_mods=None):
        attn_out, new_cache = self.attn(self.norm1(x), context=context, mask=mask, cache=cache, q_pos=q_pos, kv_pos=kv_pos)
        x = x + attn_out
        
        mlp_in = self.norm2(x)
        
        if hyper_mods is not None:
            gamma, beta = hyper_mods
            mlp_in = mlp_in * (1.0 + gamma) + beta
            
        hidden = jax.nn.gelu(self.mlp_fc1(mlp_in))
        mlp_out = self.mlp_fc2(hidden)
        
        x = x + mlp_out
        return x, new_cache

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float32):
        self.latent_dim = latent_dim
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        
        self.shared_token = nnx.Param(jax.random.normal(rngs(), (1, SHARED_SLOTS, latent_dim)).astype(jnp.float32) * 0.02)
        self.output_token = nnx.Param(jax.random.normal(rngs(), (1, OUTPUT_SLOTS, latent_dim)).astype(jnp.float32) * 0.02)
        
        self.hyper_net = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim // 2, rngs=rngs, dtype=dtype),
            jax.nn.gelu,
            nnx.Linear(latent_dim // 2, latent_dim * 4, rngs=rngs, dtype=dtype)
        )
        
        self.processor1 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        self.processor2 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

        self.budget_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)

        self.salience_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.salience_head.bias = jnp.full((1,), 1.0)
        
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias = jnp.full((1,), -3.0)

    def _get_positions(self, seq_len):
        seq_pos = jnp.arange(seq_len)
        shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        output_pos = jnp.arange(MAX_SEQ_LEN + SHARED_SLOTS, MAX_SEQ_LEN + SHARED_SLOTS + OUTPUT_SLOTS)
        return seq_pos, shared_pos, output_pos
        
    def _get_hyper_mods(self, z_seq):
        prompt_context = jnp.mean(z_seq, axis=1) 
        hyper_out = self.hyper_net(prompt_context)
        
        gamma1, beta1, gamma2, beta2 = jnp.split(hyper_out, 4, axis=-1)
        mods = [x[:, None, :] for x in (gamma1, beta1, gamma2, beta2)]
        
        return (mods[0], mods[1]), (mods[2], mods[3])

    def _get_sliding_divider_masks(self, z_seq):
        seq_repr = jnp.mean(z_seq, axis=1)
        reason_ratio = jax.nn.sigmoid(self.budget_head(seq_repr)) 
        divider_pos = reason_ratio * SHARED_SLOTS
        
        indices = jnp.arange(SHARED_SLOTS)
        dist = (divider_pos - indices[None, :]) * BUDGET_GATE_SHARPNESS
        reason_mask = jax.nn.sigmoid(dist)[:, :, None] 
        know_mask = 1.0 - reason_mask 
        
        return reason_mask, know_mask

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False):
        batch_size, seq_len = tokens.shape
        seq_pos, shared_pos, output_pos = self._get_positions(seq_len)

        z_seq = self.embed(tokens)
        
        mods1, mods2 = self._get_hyper_mods(z_seq)
        reason_mask, know_mask = self._get_sliding_divider_masks(z_seq)
        
        h_seq, _ = self.processor1(z_seq, mask=None, q_pos=seq_pos, kv_pos=seq_pos, hyper_mods=mods1)
        z_seq, _ = self.processor2(h_seq, mask=None, q_pos=seq_pos, kv_pos=seq_pos, hyper_mods=mods2)

        z_shared = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        z_output = jnp.tile(self.output_token.value, (batch_size, 1, 1))
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        def scan_step(carry, t_signal):
            curr_seq, curr_shared, curr_output = carry
            
            z_reason_in = curr_shared + t_signal[None, None, :]
            shared_ctx = jnp.concatenate([curr_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])
            
            h_reason, _ = self.processor1(z_reason_in, context=shared_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods1)
            new_reason_raw, _ = self.processor2(h_reason, context=shared_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods2)
            
            shared_after_reason = new_reason_raw * reason_mask + curr_shared * (1.0 - reason_mask)
            
            z_know_in = shared_after_reason + t_signal[None, None, :]
            know_ctx = jnp.concatenate([curr_seq, shared_after_reason], axis=1)
            
            h_know, _ = self.processor1(z_know_in, context=know_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods1)
            new_know_raw, _ = self.processor2(h_know, context=know_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods2)
            
            new_shared = new_know_raw * know_mask + shared_after_reason * (1.0 - know_mask)
            
            z_output_in = curr_output + t_signal[None, None, :]
            output_ctx = jnp.concatenate([curr_seq, new_shared], axis=1)
            output_kv_pos = jnp.concatenate([seq_pos, shared_pos])
            
            h_output, _ = self.processor1(z_output_in, context=output_ctx, q_pos=output_pos, kv_pos=output_kv_pos, hyper_mods=mods1)
            new_output, _ = self.processor2(h_output, context=output_ctx, q_pos=output_pos, kv_pos=output_kv_pos, hyper_mods=mods2)
            
            seq_kv_pos = output_pos
            h_proposed, _ = self.processor1(curr_seq, context=new_output, q_pos=seq_pos, kv_pos=seq_kv_pos, hyper_mods=mods1)
            proposed_updates, _ = self.processor2(h_proposed, context=new_output, q_pos=seq_pos, kv_pos=seq_kv_pos, hyper_mods=mods2)
            
            salience_logits = self.salience_head(curr_seq)
            salience = jax.nn.sigmoid(salience_logits)
            new_seq = curr_seq + salience * (proposed_updates - curr_seq)
            
            step_temp_loss = jnp.mean((1.0 - salience) * jnp.square(proposed_updates - curr_seq), axis=(1, 2))
            
            mean_salience = jnp.mean(salience, axis=(1, 2))
            latent_shift = jnp.mean(jnp.abs(new_shared - curr_shared), axis=(1, 2))
            
            halt_logits = self.halt_head(new_shared).mean(axis=(1, 2)) - latent_shift - mean_salience
            halt_prob = jax.nn.sigmoid(halt_logits * HALT_TEMP)
            
            return (new_seq, new_shared, new_output), (new_seq, halt_prob, step_temp_loss)

        scan_fn = nnx.scan(nnx.remat(scan_step), in_axes=(nnx.Carry, 0), unroll=4)
        _, (all_z_seq, all_halts, all_temp_loss) = scan_fn((z_seq, z_shared, z_output), all_time_embeds)
        
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain

        last_step_halt_prob = all_halts[-1]
        remaining_prob = p_remain[-1] * (1.0 - last_step_halt_prob)
        step_weights = step_weights.at[-1].add(remaining_prob)

        weighted_z = jnp.einsum('sb,sbnd->bnd', step_weights, all_z_seq)
        
        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0)
        temporal_loss = jnp.sum(step_weights * all_temp_loss, axis=0)

        logits = weighted_z @ self.embed.embedding.value.T
        
        return logits, ponder_cost, temporal_loss

    def infer(self, tokens, max_steps=MAX_STEPS_LIMIT, threshold=0.5):
        batch_size, seq_len = tokens.shape
        seq_pos, shared_pos, output_pos = self._get_positions(seq_len)
        
        z_seq = self.embed(tokens)
        
        mods1, mods2 = self._get_hyper_mods(z_seq)
        reason_mask, know_mask = self._get_sliding_divider_masks(z_seq)
        
        h_seq, _ = self.processor1(z_seq, mask=None, q_pos=seq_pos, kv_pos=seq_pos, hyper_mods=mods1)
        z_seq, _ = self.processor2(h_seq, mask=None, q_pos=seq_pos, kv_pos=seq_pos, hyper_mods=mods2)

        z_shared = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        z_output = jnp.tile(self.output_token.value, (batch_size, 1, 1))
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        def scan_step(state, t_signal):
            step, curr_seq, curr_shared, curr_output, halt_prob = state
            
            z_reason_in = curr_shared + t_signal[None, None, :]
            shared_ctx = jnp.concatenate([curr_seq, curr_shared], axis=1)
            shared_kv_pos = jnp.concatenate([seq_pos, shared_pos])
            
            h_reason, _ = self.processor1(z_reason_in, context=shared_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods1)
            new_reason_raw, _ = self.processor2(h_reason, context=shared_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods2)
            shared_after_reason = new_reason_raw * reason_mask + curr_shared * (1.0 - reason_mask)
            
            z_know_in = shared_after_reason + t_signal[None, None, :]
            know_ctx = jnp.concatenate([curr_seq, shared_after_reason], axis=1)
            
            h_know, _ = self.processor1(z_know_in, context=know_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods1)
            new_know_raw, _ = self.processor2(h_know, context=know_ctx, q_pos=shared_pos, kv_pos=shared_kv_pos, hyper_mods=mods2)
            new_shared = new_know_raw * know_mask + shared_after_reason * (1.0 - know_mask)
            
            z_output_in = curr_output + t_signal[None, None, :]
            output_ctx = jnp.concatenate([curr_seq, new_shared], axis=1)
            output_kv_pos = jnp.concatenate([seq_pos, shared_pos])
            
            h_output, _ = self.processor1(z_output_in, context=output_ctx, q_pos=output_pos, kv_pos=output_kv_pos, hyper_mods=mods1)
            new_output, _ = self.processor2(h_output, context=output_ctx, q_pos=output_pos, kv_pos=output_kv_pos, hyper_mods=mods2)
            
            seq_kv_pos = output_pos
            h_proposed, _ = self.processor1(curr_seq, context=new_output, q_pos=seq_pos, kv_pos=seq_kv_pos, hyper_mods=mods1)
            proposed_updates, _ = self.processor2(h_proposed, context=new_output, q_pos=seq_pos, kv_pos=seq_kv_pos, hyper_mods=mods2)
            
            salience_logits = self.salience_head(curr_seq)
            salience = jax.nn.sigmoid(salience_logits)
            new_seq = curr_seq + salience * (proposed_updates - curr_seq)
            
            mean_salience = jnp.mean(salience, axis=(1, 2))
            latent_shift = jnp.mean(jnp.abs(new_shared - curr_shared), axis=(1, 2))
            halt_logits = self.halt_head(new_shared).mean(axis=(1, 2)) - latent_shift - mean_salience
            new_halt_prob = jax.nn.sigmoid(halt_logits * HALT_TEMP)
            
            has_halted = halt_prob >= threshold
            
            final_seq    = jnp.where(has_halted[:, None, None], curr_seq,    new_seq)
            final_shared = jnp.where(has_halted[:, None, None], curr_shared, new_shared)
            final_output = jnp.where(has_halted[:, None, None], curr_output, new_output)
            final_halt_prob = jnp.where(has_halted, halt_prob, new_halt_prob)
            
            return (step + 1, final_seq, final_shared, final_output, final_halt_prob), None

        init_state = (0, z_seq, z_shared, z_output, jnp.zeros((batch_size,)))
        scan_fn = nnx.scan(scan_step, in_axes=(nnx.Carry, 0))
        (final_step, final_seq, _, _, _), _ = scan_fn(init_state, all_time_embeds)
        
        logits = final_seq @ self.embed.embedding.value.T
        return logits

model = UniversalReasoner(LATENT_DIM, rngs=nnx.Rngs(0))

@nnx.jit
def train_step(m, opt, batch_tokens):
    def loss_fn(model_instance):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        
        preds, ponder_cost, temporal_cost = model_instance(inputs, training=True)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
        token_loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-8)
        
        temporal_cost_clipped = jnp.clip(jnp.mean(temporal_cost), a_max=10.0)
        total_loss = (
            token_loss
            + PONDER_LAMBDA * jnp.mean(ponder_cost)
            + TEMP_LAMBDA * temporal_cost_clipped
        )
        return total_loss, (token_loss, jnp.mean(ponder_cost), jnp.mean(temporal_cost))

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
    opt.update(m, grads)
    return loss, aux

schedule = optax.warmup_cosine_decay_schedule(1e-6, 8e-5, 1000, 100000, 1e-6)
base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), 
    optax.adamw(learning_rate=schedule)
)
optimizer_chain = optax.MultiSteps(base_optimizer, every_k_schedule=ACCUMULATION_STEPS)

optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)