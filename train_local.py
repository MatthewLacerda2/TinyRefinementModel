import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset

LATENT_DIM = 512
BATCH_SIZE = 4
MAX_STEPS_LIMIT = 8
MAX_SEQ_LEN = 128
SCRATCH_SLOTS = 64 
VOCAB_SIZE = 100277

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SCRATCH_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)

    def rotate(self, x_in, sin, cos):
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(x_in.shape[-3], self.head_dim)[None, :, None, :]
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(x_in.shape[-3], self.head_dim)[None, :, None, :]
        x_even, x_odd = x_in[..., 0::2], x_in[..., 1::2]
        x_rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x_in.shape)
        return x_in * cos_ext + x_rot * sin_ext

    def __call__(self, x, mask=None, cache_k=None, cache_v=None):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        
        if cache_k is not None and cache_v is not None:
            prefix_len = cache_k.shape[1]
            x_scratch = x[:, prefix_len:, :]
            k_scratch = self.k_proj(x_scratch).reshape(b, -1, self.num_heads, self.head_dim)
            v_scratch = self.v_proj(x_scratch).reshape(b, -1, self.num_heads, self.head_dim)
            
            sin = self.sin_cached
            cos = self.cos_cached
            q = self.rotate(q, sin[:s, :], cos[:s, :])
            k_scratch = self.rotate(k_scratch, sin[prefix_len:s, :], cos[prefix_len:s, :])
            
            k = jnp.concatenate([cache_k, k_scratch], axis=1)
            v = jnp.concatenate([cache_v, v_scratch], axis=1)
        else:
            k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
            v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
            sin = self.sin_cached[:s, :]
            cos = self.cos_cached[:s, :]
            q = self.rotate(q, sin, cos)
            k = self.rotate(k, sin, cos)

        logits = jnp.einsum('bshd,bthd->bhst', q, k) * self.scale
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhst,bthd->bshd', weights, v)
        return self.o_proj(out.reshape(b, s, d))

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, rngs, dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.mlp = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim * 4, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Linear(latent_dim * 4, latent_dim, rngs=rngs, dtype=dtype)
        )

    def __call__(self, x, mask, cache_k=None, cache_v=None):
        x = x + self.attn(self.norm1(x), mask=mask, cache_k=cache_k, cache_v=cache_v)
        x = x + self.mlp(self.norm2(x))
        return x

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.num_scratch = SCRATCH_SLOTS 
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=jnp.float16, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=jnp.float16, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)) * 0.02)
        self.processor = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs)
        
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=jnp.float32, rngs=rngs)

    def get_mask(self, seq_len):
        total_len = seq_len + self.num_scratch
        causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))
        mask = causal.at[seq_len:, :].set(True)
        return mask[None, None, :, :]

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, key=None):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        
        # Precompute prompt KV
        z_prompt_norm = self.processor.norm1(z_seq)
        k_prompt = self.processor.attn.k_proj(z_prompt_norm).reshape(batch_size, seq_len, self.processor.attn.num_heads, self.processor.attn.head_dim)
        v_prompt = self.processor.attn.v_proj(z_prompt_norm).reshape(batch_size, seq_len, self.processor.attn.num_heads, self.processor.attn.head_dim)
        
        sin_p = self.processor.attn.sin_cached[:seq_len, :]
        cos_p = self.processor.attn.cos_cached[:seq_len, :]
        k_prompt = self.processor.attn.rotate(k_prompt, sin_p, cos_p)

        # FIXED: Use [...] instead of .value to avoid warning
        z_scratch = jnp.tile(self.scratch_token[...], (batch_size, 1, 1))
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = self.get_mask(seq_len)

        # We always scan the full MAX_STEPS to keep the graph static
        scan_steps = jnp.arange(max_steps)

        def scan_step(carry, i):
            curr_z = carry
            t_signal = self.time_embed(i)[None, None, :]
            # Apply time only to scratchpad
            z_scratch_with_time = curr_z[:, seq_len:, :] + t_signal
            z_with_time = jnp.concatenate([curr_z[:, :seq_len, :], z_scratch_with_time], axis=1)
                        
            new_z = self.processor(z_with_time, mask, cache_k=k_prompt, cache_v=v_prompt)
            
            halt_logit = self.halt_head(new_z[:, seq_len:, :]).mean(axis=(1, 2))
            halt_score = nnx.sigmoid(halt_logit)
            
            # Carry is new_z
            # Output stack is (new_z, halt_score)
            return new_z, (new_z, halt_score)

        # FIXED: Use lax.scan instead of fori_loop
        # Returns: final_z, (stacked_all_zs, stacked_halt_scores)
        _, (all_z, all_halts) = jax.lax.scan(scan_step, z_combined, scan_steps)
        
        if training and key is not None:
            # Pick a random depth to train on
            # 1-based index (1..MAX) -> 0-based array index (0..MAX-1)
            active_steps = jax.random.randint(key, (), 1, max_steps + 1)
            idx = active_steps - 1
            final_z = all_z[idx]
        else:
            # Inference: Use the final result
            final_z = all_z[-1]

        final_seq = final_z[:, :seq_len, :]
        return self.decoder(final_seq), all_halts

class TextDataGenerator:
    def __init__(self):
        self.dataset = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True)
        self.iterator = iter(self.dataset)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def get_batch(self, batch_size):
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                item = next(self.iterator)
            text = item['text']
            tokens = self.enc.encode(text)
            if len(tokens) < MAX_SEQ_LEN:
                tokens = tokens + [100257] * (MAX_SEQ_LEN - len(tokens))
            else:
                tokens = tokens[:MAX_SEQ_LEN]
            batch_ids.append(tokens)
        return jnp.array(batch_ids, dtype=jnp.int32)

optimizer_chain = optax.adamw(3e-4)

@jax.jit
def pure_train_step(graphdef, state, opt_state, batch_tokens, key):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        
        preds, halt_scores = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != 100257)
        token_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
        
        step_indices = jnp.arange(MAX_STEPS_LIMIT)[:, None]
        target_halt = jnp.minimum(1.0, step_indices / 4.0)
        
        # halt_scores is [Steps, Batch]
        halt_loss = jnp.mean((halt_scores - target_halt) ** 2)
        
        total_loss = token_loss + 0.1 * halt_loss
        return total_loss, (token_loss, halt_loss)

    (loss, (raw_ce, h_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state)
    updates, new_opt_state = optimizer_chain.update(grads, opt_state, state)
    new_state = optax.apply_updates(state, updates)
    return new_state, new_opt_state, loss, raw_ce, h_loss

if __name__ == "__main__":
    print(f"🚀 Initializing Standard Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    
    graphdef, state = nnx.split(model)
    opt_state = optimizer_chain.init(state)
    
    data_gen = TextDataGenerator()
    key = jax.random.key(0)
    print("Starting training loop...")
    for step in range(1, 10000):
        key, subkey = jax.random.split(key)
        batch = data_gen.get_batch(BATCH_SIZE)
        
        state, opt_state, loss, raw_ce, h_loss = pure_train_step(
            graphdef, state, opt_state, batch, subkey
        )
        if step % 100 == 0:
            print(f"Step {step:04d} | Loss: {loss:.4f} | CE: {raw_ce:.4f} | Halt Loss: {h_loss:.4f}")
            print("\n--- INFERENCE CHECK ---")
            
            model = nnx.merge(graphdef, state)
            prompt = "Write a brief educational tip suited for college students"
            gen_tokens = data_gen.enc.encode(prompt)
            
            for _ in range(15):
                curr_input = jnp.array([gen_tokens], dtype=jnp.int32)
                logits, _ = model(curr_input, training=False)
                next_tok = jnp.argmax(logits[0, -1]).item()
                gen_tokens.append(next_tok)
                if next_tok == 100257: 
                    break
            
            print(f"Input: {prompt}")
            print(f"Generated: {data_gen.enc.decode(gen_tokens)}")
            print("-----------------------\n")