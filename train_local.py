import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset, interleave_datasets
import pickle
import csv
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LATENT_DIM = 384
BATCH_SIZE = 8
MAX_STEPS_LIMIT = 8
MAX_SEQ_LEN = 512
SCRATCH_SLOTS = 64 
VOCAB_SIZE = 50257
PAD_TOKEN_ID = 50256

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
        dtype = x_in.dtype
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(x_in.shape[-3], self.head_dim)[None, :, None, :].astype(dtype)
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(x_in.shape[-3], self.head_dim)[None, :, None, :].astype(dtype)
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
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float16):
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
    def __init__(self, latent_dim, rngs, dtype=jnp.float16):
        self.latent_dim = latent_dim
        self.num_scratch = SCRATCH_SLOTS 
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)) * 0.02)
        self.processor = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=jnp.float32, rngs=rngs)

    def get_mask(self, seq_len):
        total_len = seq_len + self.num_scratch
        causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))
        mask = causal.at[seq_len:, :].set(True)
        return mask[None, None, :, :]

    def generate(self, tokens, gen_len, key, temperature=0.7, top_p=0.9):
        # Static-shape generation to prevent OOM
        batch_size, start_len = tokens.shape
        
        def body_fn(i, carry):
            current_tokens, loop_key = carry
            step_key, next_loop_key = jax.random.split(loop_key)
            
            logits, _ = self(current_tokens, training=False)
            
            valid_idx = start_len + i - 1
            next_logits = logits[:, valid_idx, :] / temperature
            
            # Top-P (Nucleus) Filtering
            sorted_indices = jnp.argsort(next_logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(next_logits, sorted_indices, axis=-1)
            
            probs = jax.nn.softmax(sorted_logits, axis=-1)
            cum_probs = jnp.cumsum(probs, axis=-1)
            
            # Create a mask for tokens to exclude (those exceeding cumulative top_p)
            # We shift the mask to ensure at least one token (the one that crosses the threshold) is kept
            mask = cum_probs > top_p
            mask = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=bool), mask[:, :-1]], axis=-1)
            
            filtered_logits = jnp.where(mask, -1e9, sorted_logits)
            
            # Sample from the filtered distribution
            sample_indices = jax.random.categorical(step_key, filtered_logits)
            next_token = jnp.take_along_axis(sorted_indices, sample_indices[:, None], axis=-1)
            
            new_tokens = jax.lax.dynamic_update_slice(current_tokens, next_token, (0, start_len + i))
            return (new_tokens, next_loop_key)

        # Pad tokens to a fixed max length
        padded_tokens = jnp.pad(tokens, ((0, 0), (0, gen_len)), constant_values=PAD_TOKEN_ID)
        
        (final_tokens, _) = jax.lax.fori_loop(0, gen_len, body_fn, (padded_tokens, key))
        return final_tokens

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

        z_scratch = jnp.tile(self.scratch_token[...], (batch_size, 1, 1))
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = self.get_mask(seq_len)

        scan_steps = jnp.arange(max_steps)

        @jax.checkpoint
        def scan_step(carry, i):
            curr_z = carry
            t_signal = self.time_embed(i)[None, None, :]
            z_scratch_with_time = curr_z[:, seq_len:, :] + t_signal
            z_with_time = jnp.concatenate([curr_z[:, :seq_len, :], z_scratch_with_time], axis=1)
                        
            new_z = self.processor(z_with_time, mask, cache_k=k_prompt, cache_v=v_prompt)
            
            halt_logit = self.halt_head(new_z[:, seq_len:, :]).mean(axis=(1, 2))
            halt_score = nnx.sigmoid(halt_logit)
            
            return new_z, (new_z, halt_score)

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
    def __init__(self, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        
        # All 8 splits of Cosmopedia-v2
        configs = [
            "web_samples_v1", "web_samples_v2", "stanford", 
            "stories", "wikihow", "openstax", 
            "khanacademy", "automathtext"
        ]
        
        print(f"mixing {len(configs)} dataset splits...")
        ds_list = [
            load_dataset("HuggingFaceTB/cosmopedia-v2", c, split="train", streaming=True) 
            for c in configs
        ]
        
        # This creates a single stream that pulls from all datasets 
        self.dataset = interleave_datasets(ds_list, stopping_strategy="all_exhausted")
        self.iterator = iter(self.dataset)
        self.exhausted = False

    def get_batch(self, batch_size):
        if self.exhausted: return None
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
                tokens = self.enc.encode(item['text'])
                # Pad/Truncate logic
                if len(tokens) < self.max_seq_len:
                    tokens = tokens + [PAD_TOKEN_ID] * (self.max_seq_len - len(tokens))
                else:
                    tokens = tokens[:self.max_seq_len]
                batch_ids.append(tokens)
            except StopIteration:
                self.exhausted = True
                break
        return jnp.array(batch_ids, dtype=jnp.int32)

class LossMonitor:
    def __init__(self, patience=50000, window=5000):
        self.patience = patience
        self.window = window
        self.history = []
        self.best_loss = float('inf')
        self.last_improvement_step = 0

    def push(self, step, loss):
        self.history.append(loss)
        if len(self.history) > self.window:
            self.history.pop(0)
        
        avg_loss = sum(self.history) / len(self.history)
        if avg_loss < (self.best_loss - 0.01):
            self.best_loss = avg_loss
            self.last_improvement_step = step
            return False
        
        if (step - self.last_improvement_step) > self.patience:
            return True # Converged
        return False

schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=1e-4,
    warmup_steps=500,
    decay_steps=50000,
    end_value=1e-6
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule)
)

@jax.jit
def pure_train_step(graphdef, state, opt_state, batch_tokens, key):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        
        preds, halt_scores = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
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
    
    start_step = 1
    if os.path.exists("checkpoint.pkl"):
        print("🔄 Found checkpoint! Loading weights...")
        with open("checkpoint.pkl", "rb") as f:
            ckpt = pickle.load(f)
            state = ckpt['state']
            opt_state = ckpt['opt_state']
            start_step = ckpt.get('step', 0) + 1
        print(f"✅ Resuming from step {start_step}")

    data_gen = TextDataGenerator(MAX_SEQ_LEN)
    key = jax.random.key(start_step)

    history_file = "training_history.csv"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                reader = csv.DictReader(f)
                history_steps = [int(row['step']) for row in reader]
                if history_steps:
                    print(f"📊 Found training history up to step {max(history_steps)}")
        except Exception as e:
            print(f"⚠️ Could not read history: {e}")

    import time
    monitor = LossMonitor()
    step = start_step
    
    while True:
        t0 = time.time()
        key, subkey = jax.random.split(key)
        batch = data_gen.get_batch(BATCH_SIZE)
        
        if batch is None:
            print("🎉 Dataset finished! Training complete.")
            break
            
        t_data = time.time() - t0
        
        t1 = time.time()
        state, opt_state, loss, raw_ce, h_loss = pure_train_step(
            graphdef, state, opt_state, batch, subkey
        )
        loss.block_until_ready()
        t_train = time.time() - t1

        if monitor.push(step, float(loss)):
            print(f"📉 Model converged (no improvement in {monitor.patience} steps). Stopping.")
            break

        if step % 500 == 0:
            if hasattr(jax, "clear_caches"): jax.clear_caches()
            
            with open("checkpoint.pkl", "wb") as f:
                pickle.dump({"state": state, "opt_state": opt_state, "step": step}, f)
            
            print("--- INFERENCE CHECK ---")
            print(f"Step {step:04d} | Loss: {loss:.4f} | CE: {raw_ce:.4f} | Halt: {h_loss:.4f} | Train: {t_train:.2f}s | Data: {t_data:.2f}s")
            model_eval = nnx.merge(graphdef, state)
            prompt = "What do you know about?"
            tokens_in = jnp.array([data_gen.enc.encode(prompt)], dtype=jnp.int32)
            
            # Use the entropy of the current step for sampling
            gen_key = jax.random.key(step)
            gen_tokens = model_eval.generate(tokens_in, gen_len=64, key=gen_key)
            
            decoded = data_gen.enc.decode(gen_tokens[0, tokens_in.shape[1]:].tolist())
            print(f"Prompt: {prompt}")
            print(f"Output: {decoded}")
            print("-----------------------\n")
            
            file_exists = os.path.exists(history_file)
            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "loss", "ce", "halt_loss", "t_train", "t_data"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "step": int(step),
                    "loss": f"{float(loss):.4f}",
                    "ce": f"{float(raw_ce):.4f}",
                    "halt_loss": f"{float(h_loss):.4f}",
                    "t_train": f"{float(t_train):.2f}",
                    "t_data": f"{float(t_data):.2f}"
                })
                
            del model_eval

        step += 1
