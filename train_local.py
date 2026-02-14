import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset # pip install datasets

LATENT_DIM = 384      # Reduced from 512 to save VRAM
BATCH_SIZE = 8        # Keep small
ACCUM_STEPS = 16      # Increased to simulate larger batch size
MAX_STEPS_LIMIT = 8   # "Reasoning depth" (recurrence). 8 is enough for PoC.
MAX_SEQ_LEN = 128     # Increased for text (64 is too short for stories)
VOCAB_SIZE = 50304    # Standard GPT-2/3 vocab size (rounded for efficiency)

# --- 1. Fix Attention (Add Causal Mask) ---
class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        
        # RoPE Setup
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        
        # RoPE Application
        sin = self.sin_cached[:s, :]
        cos = self.cos_cached[:s, :]
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        
        def rotate(x_in):
            x_even, x_odd = x_in[..., 0::2], x_in[..., 1::2]
            x_rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x_in.shape)
            return x_in * cos_ext + x_rot * sin_ext

        q, k = rotate(q), rotate(k)
        
        # Transpose: (B, H, S, D)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        
        # --- CAUSAL MASKING (Critical for Text) ---
        # Mask out future tokens (upper triangle)
        mask = jnp.tril(jnp.ones((s, s), dtype=bool))
        mask = mask[None, None, :, :] # Broadcast over Batch and Heads
        logits = jnp.where(mask, logits, -1e9)
        
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, s, d)
        return self.o_proj(out)

class LatentReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads=num_heads, in_features=latent_dim, rngs=rngs, dtype=dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.fc = nnx.Linear(latent_dim, latent_dim, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)

    def __call__(self, z):
        z = z + self.attn(self.norm1(z))
        z = z + nnx.gelu(self.fc(self.norm2(z)))
        return z

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        dtype = jnp.float32
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=dtype, rngs=rngs)
        
        self.processor = LatentReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        
        self.norm = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)

    def __call__(self, tokens, max_steps=16, training=False, key=None):
        # tokens: (batch, seq_len)
        z = self.embed(tokens)
        
        # Positions are now handled by RoPE within the LatentReasoningBlock
        
        batch_size, seq_len, _ = z.shape
        
        # Predict complexity based on the whole sequence
        predicted_steps = nnx.sigmoid(jnp.mean(self.complexity_head(z), axis=1)) * max_steps
        
        if training and key is not None:
            is_key_batched = (key.ndim > 0) and (key.shape[0] == batch_size)
            if is_key_batched:
                step_keys = jax.vmap(lambda k: jax.random.split(k, max_steps))(key)
                step_keys = jnp.swapaxes(step_keys, 0, 1)
            else:
                step_keys = jax.random.split(key, max_steps)
        else:
            step_keys = jnp.zeros((max_steps, batch_size, 2) if z.ndim > 1 else (max_steps, 2), dtype=jnp.uint32)

        def refine_step(carry, step_key_input):
            curr_z, step_idx, run_prob, w_z = carry
            
            next_z_raw = self.processor(curr_z)
            
            if training:
                if step_key_input.ndim > 0:
                    noise = jax.vmap(lambda k: jax.random.normal(k, curr_z.shape[1:], dtype=curr_z.dtype))(step_key_input)
                else:
                    noise = jax.random.normal(step_key_input, curr_z.shape, dtype=curr_z.dtype)
                next_z_raw = next_z_raw + (noise * 0.01)
            
            next_z = self.norm(next_z_raw)
            
            # Mean pooling for halt signal
            halt = nnx.sigmoid(jnp.mean(self.halt_fc(next_z), axis=1))
            p = halt * (1.0 - run_prob)
            
            new_z = w_z + (p[:, :, None] * next_z)
            
            return (next_z, step_idx + 1, run_prob + p, new_z), p

        init_carry = (
            z,
            0,
            jnp.zeros((batch_size, 1), dtype=jnp.float32),
            jnp.zeros((batch_size, seq_len, self.latent_dim), dtype=jnp.float32)
        )
        
        (final_z, _, final_prob, w_z), step_probs = jax.lax.scan(
            refine_step, init_carry, step_keys, length=max_steps
        )
        
        step_probs = jnp.swapaxes(step_probs, 0, 1)
        
        rem = 1.0 - final_prob
        final_w_z = w_z + (rem[:, :, None] * final_z)
        
        w_out = self.decoder(final_w_z)
        
        # Output is (batch, seq_len, vocab_size)
        return w_out.astype(jnp.float32), step_probs, predicted_steps

# --- 2. HuggingFace Data Streamer ---
class TextDataGenerator:
    def __init__(self):
        # Streaming = True prevents downloading the whole dataset (saves disk/ram)
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        self.iterator = iter(self.dataset)
        self.enc = tiktoken.get_encoding("cl100k_base")
        
    def get_batch(self, batch_size):
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset) # Restart if end reached
                item = next(self.iterator)
                
            text = item['text']
            tokens = self.enc.encode(text)
            
            # Simple truncation/padding to MAX_SEQ_LEN
            if len(tokens) < MAX_SEQ_LEN:
                tokens = tokens + [50256] * (MAX_SEQ_LEN - len(tokens)) # Pad (using EOS-like ID)
            else:
                tokens = tokens[:MAX_SEQ_LEN]
                
            batch_ids.append(tokens)
            
        return jnp.array(batch_ids, dtype=jnp.int32)

# --- 3. Update Training Step for Next-Token Prediction ---
@nnx.jit
def train_step_text(model, optimizer, batch_tokens, noise_keys):
    loss_scale = 1000.0
    
    # Standard Causal Language Modeling setup:
    # Input:  [A, B, C, D]
    # Target: [B, C, D, E]
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(carry, loop_inputs):
            # Accumulation loop
            (inp_slice, tgt_slice, key_slice) = loop_inputs
            
            m = nnx.merge(graphdef, state)
            # Run model (it now has causal mask!)
            preds, step_probs, pred_steps = m(inp_slice, MAX_STEPS_LIMIT, True, key_slice)
            
            # Cross Entropy
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=tgt_slice)
            token_loss = jnp.mean(ce_loss)
            
            # Latent Reasoning Penalty (keep your unique regularization)
            actual_steps = jnp.sum(step_probs * jnp.arange(step_probs.shape[1])[None, :, None], axis=1)
            avg_steps = jnp.mean(actual_steps)
            
            loss = token_loss + (avg_steps * 0.001) # Small penalty for thinking too long
            
            return carry, (loss * loss_scale, token_loss)

        # Reshape for accumulation
        # (ACCUM, BATCH, LEN)
        inputs_reshaped = inputs.reshape(ACCUM_STEPS, BATCH_SIZE, -1)
        targets_reshaped = targets.reshape(ACCUM_STEPS, BATCH_SIZE, -1)
        
        _, (scaled_losses, raw_losses) = jax.lax.scan(
            scan_body, None, (inputs_reshaped, targets_reshaped, noise_keys)
        )
        
        return jnp.mean(scaled_losses), jnp.mean(raw_losses)

    (loss_s, raw_loss), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    
    return raw_loss

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🚀 Initializing Text Reasoner...")
    # Re-instantiate model with new VOCAB_SIZE and Latent Dim
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    # Make sure your UniversalReasoner calls the NEW RotaryAttention defined above!

    optimizer = nnx.Optimizer(model, optax.adamw(3e-4, weight_decay=0.1), wrt=nnx.Param)
    data_gen = TextDataGenerator()
    key = jax.random.key(0)

    start_time = time.time()

    # 5 Hours = 18000 seconds
    # Estimate step time ~ 0.5s -> ~36k steps
    for step in range(1, 10000): 
        key, subkey = jax.random.split(key)
        
        # Get large batch for accumulation
        batch_data = data_gen.get_batch(BATCH_SIZE * ACCUM_STEPS)
        noise_keys = jax.random.split(subkey, ACCUM_STEPS)
        
        loss = train_step_text(model, optimizer, batch_data, noise_keys)
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")
            
            if elapsed > (5 * 3600): # 5 Hours
                print("🛑 Time limit reached.")
                break

        # Simple text generation test every 500 steps
        if step % 500 == 0:
            print("\n--- GENERATION TEST ---")
            prompt = "Once upon a time"
            enc = tiktoken.get_encoding("cl100k_base")
            prompt_ids = jnp.array([enc.encode(prompt)], dtype=jnp.int32)
            
            # Very hacky greedy generation for PoC
            curr = prompt_ids
            for _ in range(20):
                preds, _, _ = model(curr, MAX_STEPS_LIMIT, False)
                next_token = jnp.argmax(preds[0, -1, :])
                curr = jnp.concatenate([curr, next_token[None, None]], axis=1)
            
            print(enc.decode(np.array(curr[0])))
            print("-----------------------\n")
