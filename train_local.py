import jax
import jax.numpy as jnp
from flax import nnx
import optax

BATCH_SIZE = 16
MAX_STEPS_LIMIT = 16
ACCUMULATION_STEPS = 8
SCRATCH_SLOTS = 1024
LATENT_DIM = 768
MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100277
PAD_TOKEN_ID = 100257
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

    def __call__(self, x, context=None, mask=None, cache=None):
        b, s, d = x.shape
        
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        
        kv_input = context if context is not None else x
        s_kv = kv_input.shape[1]
        
        k = self.k_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)
        v = self.v_proj(kv_input).reshape(b, s_kv, self.num_groups, self.head_dim)

        # Apply RoPE
        sin = self.sin_cached[:s, None, :]
        cos = self.cos_cached[:s, None, :]
        q = (q * cos) + (rotate_half(q) * sin)
        
        sin_kv = self.sin_cached[:s_kv, None, :]
        cos_kv = self.cos_cached[:s_kv, None, :]
        k = (k * cos_kv) + (rotate_half(k) * sin_kv)

        if cache is not None:
            prev_k, prev_v = cache
            k = jnp.concatenate([prev_k, k], axis=1)
            v = jnp.concatenate([prev_v, v], axis=1)
        
        new_cache = (k, v) if cache is not None else None

        # Multi-Query/Grouped-Query Expansion
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
        self.mlp = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim * 4, rngs=rngs, dtype=dtype),
            jax.nn.gelu,
            nnx.Linear(latent_dim * 4, latent_dim, rngs=rngs, dtype=dtype)
        )

    def __call__(self, x, context=None, mask=None, cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), context=context, mask=mask, cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.bfloat16):
        self.latent_dim = latent_dim
        self.num_scratch = SCRATCH_SLOTS
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)).astype(jnp.float32) * 0.02)
        self.processor1 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        self.processor2 = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)

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

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        h_seq, _ = self.processor1(z_seq, mask=None)
        z_seq, _ = self.processor2(h_seq, mask=None)
        
        z_scratch = jnp.tile(self.scratch_token.value, (batch_size, 1, 1))
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        def scan_step(carry, t_signal):
            curr_seq, curr_scratch = carry
            
            z_scratch_in = curr_scratch + t_signal[None, None, :]
            h_scratch, _ = self.processor1(z_scratch_in, context=curr_seq)
            new_scratch, _ = self.processor2(h_scratch, context=curr_seq)
            
            h_proposed, _ = self.processor1(curr_seq, context=new_scratch)
            proposed_updates, _ = self.processor2(h_proposed, context=new_scratch)
            
            salience_logits = self.salience_head(curr_seq)
            salience = jax.nn.sigmoid(salience_logits)
            
            new_seq = curr_seq + salience * (proposed_updates - curr_seq)
            
            step_temp_loss = jnp.mean((1.0 - salience) * jnp.square(proposed_updates - curr_seq), axis=(1, 2))
            
            latent_shift = jnp.mean(jnp.abs(new_scratch - curr_scratch), axis=(1, 2))
            halt_prob = jax.nn.sigmoid(self.halt_head(new_scratch).mean(axis=(1, 2)) - latent_shift)
            
            return (new_seq, new_scratch), (new_seq, halt_prob, step_temp_loss)

        scan_fn = nnx.scan(nnx.remat(scan_step), in_axes=(nnx.Carry, 0), unroll=4)
        _, (all_z_seq, all_halts, all_temp_loss) = scan_fn((z_seq, z_scratch), all_time_embeds)
        
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain

        last_step_halt_prob = all_halts[-1]
        remaining_prob = p_remain[-1] * (1.0 - last_step_halt_prob)
        step_weights = step_weights.at[-1].add(remaining_prob)

        weighted_z = jnp.einsum('sb,sbnd->bnd', step_weights, all_z_seq)
        
        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0)
        temporal_loss = jnp.sum(step_weights * all_temp_loss, axis=0)

        return self.decoder(weighted_z), ponder_cost, temporal_loss

    def infer(self, tokens, max_steps=MAX_STEPS_LIMIT, threshold=0.5):
        batch_size, seq_len = tokens.shape
        
        z_seq = self.embed(tokens)
        h_seq, _ = self.processor1(z_seq, mask=None)
        z_seq, _ = self.processor2(h_seq, mask=None)
        
        z_scratch = jnp.tile(self.scratch_token.value, (batch_size, 1, 1))
        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        def scan_step(state, t_signal):
            step, curr_seq, curr_scratch, halt_prob = state
            
            z_scratch_in = curr_scratch + t_signal[None, None, :]
            h_scratch, _ = self.processor1(z_scratch_in, context=curr_seq)
            new_scratch, _ = self.processor2(h_scratch, context=curr_seq)
            
            h_proposed, _ = self.processor1(curr_seq, context=new_scratch)
            proposed_updates, _ = self.processor2(h_proposed, context=new_scratch)
            
            salience_logits = self.salience_head(curr_seq)
            salience = jax.nn.sigmoid(salience_logits)
            
            new_seq = curr_seq + salience * (proposed_updates - curr_seq)
            
            latent_shift = jnp.mean(jnp.abs(new_scratch - curr_scratch), axis=(1, 2))
            new_halt_prob = jax.nn.sigmoid(self.halt_head(new_scratch).mean(axis=(1, 2)) - latent_shift)
            
            has_halted = halt_prob >= threshold
            
            final_seq = jnp.where(has_halted[:, None, None], curr_seq, new_seq)
            final_scratch = jnp.where(has_halted[:, None, None], curr_scratch, new_scratch)
            final_halt_prob = jnp.where(has_halted, halt_prob, new_halt_prob)
            
            return (step + 1, final_seq, final_scratch, final_halt_prob), None

        init_state = (0, z_seq, z_scratch, jnp.zeros((batch_size,)))
        scan_fn = nnx.scan(scan_step, in_axes=(nnx.Carry, 0))
        (final_step, final_seq, _, _), _ = scan_fn(init_state, all_time_embeds)
        
        return self.decoder(final_seq)


@nnx.jit
def train_step(model, optimizer, batch_tokens):
    def loss_fn(m):
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        preds, ponder_cost, temporal_cost = m(inputs, training=True) 
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
        token_loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-8)
        
        total_loss = token_loss + (PONDER_LAMBDA * jnp.mean(ponder_cost)) + (TEMP_LAMBDA * jnp.mean(temporal_cost))
        return total_loss, (token_loss, jnp.mean(ponder_cost))

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    optimizer.update(grads) 
    
    return loss, aux


schedule = optax.warmup_cosine_decay_schedule(1e-6, 8e-5, 1000, 100000, 1e-6)
base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), 
    optax.adamw(learning_rate=schedule)
)
optimizer_chain = optax.MultiSteps(base_optimizer, every_k_schedule=ACCUMULATION_STEPS)
