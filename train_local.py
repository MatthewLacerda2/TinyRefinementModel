import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset, interleave_datasets
import pickle
import csv
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LATENT_DIM = 384
BATCH_SIZE = 8
MAX_STEPS_LIMIT = 8
MAX_SEQ_LEN = 512
SCRATCH_SLOTS = 64
VOCAB_SIZE = 50257
PAD_TOKEN_ID = 50256
PONDER_LAMBDA = 0.01
CHECKPOINT_INTERVAL = 500

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

        @jax.checkpoint
        def scan_step(carry, i):
            curr_z = carry
            t_signal = self.time_embed(i)[None, None, :]
            z_scratch_with_time = curr_z[:, seq_len:, :] + t_signal
            z_with_time = jnp.concatenate([curr_z[:, :seq_len, :], z_scratch_with_time], axis=1)
            new_z = self.processor(z_with_time, mask, cache_k=k_prompt, cache_v=v_prompt)
            # Halting signal from scratchpad tokens
            halt_logit = self.halt_head(new_z[:, seq_len:, :]).mean(axis=(1, 2))
            halt_prob = nnx.sigmoid(halt_logit)
            return new_z, (new_z, halt_prob)

        _, (all_z, all_halts) = jax.lax.scan(scan_step, z_combined, jnp.arange(max_steps))
        
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain
        
        last_weight = step_weights[-1] + p_remain[-1] * (1.0 - all_halts[-1])
        step_weights = step_weights.at[-1].set(last_weight)

        # Average the embeddings of the sequence tokens
        all_z_seq = all_z[:, :, :seq_len, :]
        weighted_z = jnp.einsum('sb,sbsd->bsd', step_weights, all_z_seq)
        
        # Ponder cost = sum of (weight * step_index)
        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0) # [Batch]

        return self.decoder(weighted_z), ponder_cost

class TextDataGenerator:
    def __init__(self, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        print("🚀 Preparing SmolLM-Corpus mix...")
        ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True).select_columns(["text"])
        ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True).select_columns(["text"])
        self.dataset = interleave_datasets([ds_cosmo, ds_fineweb], stopping_strategy="all_exhausted")
        self.iterator = iter(self.dataset)
        self.exhausted = False

    def get_batch(self, batch_size):
        if self.exhausted: return None
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
                tokens = self.enc.encode(item['text'])
                if len(tokens) < self.max_seq_len:
                    tokens = tokens + [PAD_TOKEN_ID] * (self.max_seq_len - len(tokens))
                else:
                    tokens = tokens[:self.max_seq_len]
                batch_ids.append(tokens)
            except StopIteration: self.exhausted = True; break
            except Exception as e: continue
        return jnp.array(batch_ids, dtype=jnp.int32)

class LossMonitor:
    def __init__(self, patience=50000, window=5000):
        self.patience, self.window = patience, window
        self.history, self.best_loss, self.last_improvement_step = [], float('inf'), 0
    def push(self, step, loss):
        self.history.append(loss)
        if len(self.history) > self.window: self.history.pop(0)
        avg_loss = sum(self.history) / len(self.history)
        if avg_loss < (self.best_loss - 0.01):
            self.best_loss, self.last_improvement_step = avg_loss, step
            return False
        return (step - self.last_improvement_step) > self.patience

schedule = optax.warmup_cosine_decay_schedule(1e-6, 8e-5, 1000, 100000, 1e-6)
optimizer_chain = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule))

@jax.jit
def pure_train_step(graphdef, state, opt_state, batch_tokens, key):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        preds, ponder_cost = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
        token_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
        
        # ACT Ponder Loss (encourage efficiency)
        avg_ponder = jnp.mean(ponder_cost)
        ponder_loss = PONDER_LAMBDA * avg_ponder
        
        return token_loss + ponder_loss, (token_loss, avg_ponder)

    (loss, (raw_ce, avg_p)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state)
    updates, new_opt_state = optimizer_chain.update(grads, opt_state, state)
    new_state = optax.apply_updates(state, updates)
    return new_state, new_opt_state, loss, raw_ce, avg_p

if __name__ == "__main__":
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    graphdef, state = nnx.split(model)
    opt_state = optimizer_chain.init(state)
    
    start_step = 1
    data_gen = TextDataGenerator(MAX_SEQ_LEN)
    key = jax.random.key(start_step)
    history_file = "training_history.csv"
    monitor = LossMonitor()
    step = start_step
    
    while True:
        t0 = time.time()
        key, subkey = jax.random.split(key)
        batch = data_gen.get_batch(BATCH_SIZE)
        if batch is None: break
        
        t1 = time.time()
        state, opt_state, loss, raw_ce, avg_p = pure_train_step(graphdef, state, opt_state, batch, subkey)
        loss.block_until_ready()
        t_train, t_data = time.time() - t1, t1 - t0

        if monitor.push(step, float(loss)): break

        if step % CHECKPOINT_INTERVAL == 0:
            with open("checkpoint.pkl", "wb") as f:
                pickle.dump({"state": state, "opt_state": opt_state, "step": step}, f)
            
            print(f"Step {step:04d} | Loss: {loss:.4f} | CE: {raw_ce:.4f} | Avg Steps: {avg_p:.2f}")
            
            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "loss", "ce", "avg_ponder", "t_train", "t_data"])
                if not os.path.exists(history_file) or os.stat(history_file).st_size == 0: writer.writeheader()
                writer.writerow({"step": int(step), "loss": f"{float(loss):.4f}", "ce": f"{float(raw_ce):.4f}", 
                                 "avg_ponder": f"{float(avg_p):.4f}", "t_train": f"{t_train:.2f}", "t_data": f"{t_data:.2f}"})
        step += 1