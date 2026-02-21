import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import pickle
import csv
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

BATCH_SIZE = 2
MAX_STEPS_LIMIT = 4
ACCUMULATION_STEPS = 64
SCRATCH_SLOTS = 256
LATENT_DIM = 384
MAX_SEQ_LEN = 512
VOCAB_SIZE = 50257
PAD_TOKEN_ID = 50256
LOSS_SCALE = 128.0
CHECKPOINT_INTERVAL = 100
PONDER_LAMBDA = 0.005
TEMP_LAMBDA = 0.01

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
        t = jnp.arange(MAX_SEQ_LEN + SCRATCH_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)
        self.k_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.v_proj = nnx.Linear(in_features, self.num_groups * self.head_dim, rngs=rngs, dtype=jnp.float32)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float32)

    def __call__(self, x, mask=None):
        b, s, d = x.shape
        
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_groups, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_groups, self.head_dim)

        sin = self.sin_cached[:s, None, :]
        cos = self.cos_cached[:s, None, :]

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        repeats = self.num_heads // self.num_groups
        k = jnp.repeat(k, repeats, axis=2)
        v = jnp.repeat(v, repeats, axis=2)

        q = q * self.scale
        out = jax.nn.dot_product_attention(q, k, v, mask=mask)

        out = out.reshape(b, s, d)
        return self.o_proj(out)

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, num_groups=2, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.mlp = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim * 4, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Linear(latent_dim * 4, latent_dim, rngs=rngs, dtype=dtype)
        )

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float32):
        self.latent_dim = latent_dim
        self.num_scratch = SCRATCH_SLOTS
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)).astype(jnp.float32) * 0.02)
        self.processor = nnx.remat(lambda: StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype))()

        self.salience_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.salience_head.bias = jnp.full((1,), 1.0)
        
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.halt_head.bias = jnp.full((1,), -3.0)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=jnp.float32, rngs=rngs)

    def get_mask(self, seq_len):
        total_len = seq_len + self.num_scratch
        causal_text = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        mask = jnp.ones((total_len, total_len), dtype=bool)
        mask = mask.at[:seq_len, :seq_len].set(causal_text)
        
        mask = mask.at[:seq_len, seq_len:].set(False)
        
        return mask[None, None, :, :]

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, key=None):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        z_scratch = jnp.tile(self.scratch_token[...], (batch_size, 1, 1))
        
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = self.get_mask(seq_len)

        def scan_step(carry, i):
            curr_z = carry

            t_signal = self.time_embed(i)[None, None, :]
            z_scratch_with_time = curr_z[:, seq_len:, :] + t_signal
            
            z_input = jnp.concatenate([curr_z[:, :seq_len, :], z_scratch_with_time], axis=1)
            
            new_z_raw = self.processor(z_input, mask)

            curr_seq = curr_z[:, :seq_len, :]
            new_seq_raw = new_z_raw[:, :seq_len, :]

            salience_logists = self.salience_head(curr_seq)
            salience = nnx.sigmoid(salience_logists)

            new_seq = curr_seq + salience * (new_seq_raw - curr_seq)
            new_scratch = new_z_raw[:, seq_len:, :]
            
            new_z = jnp.concatenate([new_seq, new_scratch], axis=1)

            step_temp_loss = jnp.mean((1.0 - salience) * jnp.square(new_seq_raw - curr_seq), axis=(1, 2))

            latent_shift = jnp.mean(jnp.abs(new_seq - curr_seq), axis=(1, 2))
            base_halt_logit = self.halt_head(new_scratch).mean(axis=(1, 2))
            
            halt_prob = nnx.sigmoid(base_halt_logit - latent_shift)
            
            return new_z, (new_z, halt_prob, step_temp_loss)

        scan_fn = nnx.remat(nnx.scan(scan_step))
        _, (all_z, all_halts, all_temp_loss) = scan_fn(z_combined, jnp.arange(max_steps))
        
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain

        last_step_halt_prob = all_halts[-1]
        remaining_prob = p_remain[-1] * (1.0 - last_step_halt_prob)
        step_weights = step_weights.at[-1].add(remaining_prob)

        all_z_seq = all_z[:, :, :seq_len, :]
        weighted_z = jnp.einsum('sb,sbnd->bnd', step_weights, all_z_seq)
        
        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0)

        temporal_loss = jnp.sum(step_weights * all_temp_loss, axis=0)

        return self.decoder(weighted_z), ponder_cost, temporal_loss

    def infer(self, tokens, max_steps=MAX_STEPS_LIMIT, threshold=0.5):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        z_scratch = jnp.tile(self.scratch_token[...], (batch_size, 1, 1))
        
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = self.get_mask(seq_len)

        def cond_fn(state):
            step, _, halt_prob = state
            return jnp.logical_and(step < max_steps, jnp.all(halt_prob < threshold))

        def body_fn(state):
            step, curr_z, _ = state
            
            t_signal = self.time_embed(step)[None, None, :]
            z_scratch_with_time = curr_z[:, seq_len:, :] + t_signal
            z_input = jnp.concatenate([curr_z[:, :seq_len, :], z_scratch_with_time], axis=1)
            
            new_z_raw = self.processor(z_input, mask)

            curr_seq = curr_z[:, :seq_len, :]
            new_seq_raw = new_z_raw[:, :seq_len, :]

            # Apply salience gating
            salience_logists = self.salience_head(curr_seq)
            salience = nnx.sigmoid(salience_logists)
            new_seq = curr_seq + salience * (new_seq_raw - curr_seq)
            new_scratch = new_z_raw[:, seq_len:, :]
            
            new_z = jnp.concatenate([new_seq, new_scratch], axis=1)

            # Check latent convergence
            latent_shift = jnp.mean(jnp.abs(new_seq - curr_seq), axis=(1, 2))
            base_halt_logit = self.halt_head(new_scratch).mean(axis=(1, 2))
            halt_prob = nnx.sigmoid(base_halt_logit - latent_shift)
            
            return (step + 1, new_z, halt_prob)

        init_state = (0, z_combined, jnp.zeros((batch_size,)))
        final_step, final_z, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        final_z_seq = final_z[:, :seq_len, :]
        return self.decoder(final_z_seq)


schedule = optax.warmup_cosine_decay_schedule(1e-6, 8e-5, 1000, 100000, 1e-6)
optimizer_chain = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule))

@jax.jit
def compute_gradients(graphdef, state, batch_tokens, key):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        preds, ponder_cost, temporal_cost = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
        token_loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-8)
        
        ponder_loss = PONDER_LAMBDA * jnp.mean(ponder_cost)

        temporal_loss = TEMP_LAMBDA * jnp.mean(temporal_cost)

        total_loss = token_loss + ponder_loss + temporal_loss
        
        return total_loss, (token_loss, jnp.mean(ponder_cost))

    (loss, (raw_ce, avg_p)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state)
    
    return grads, loss, raw_ce, avg_p
