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
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

LATENT_DIM = 384
BATCH_SIZE = 4
ACCUMULATION_STEPS = 16
MAX_STEPS_LIMIT = 4
MAX_SEQ_LEN = 512
SCRATCH_SLOTS = 64
VOCAB_SIZE = 50257
PAD_TOKEN_ID = 50256
PONDER_LAMBDA = 0.005
CHECKPOINT_INTERVAL = 100
LOSS_SCALE = 128.0

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SCRATCH_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)

    def __call__(self, x, mask=None):
        b, s, d = x.shape
        x = x.astype(jnp.float16)
        
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        sin = self.sin_cached[:s, None, :].astype(jnp.float32)
        cos = self.cos_cached[:s, None, :].astype(jnp.float32)
        
        q = (q.astype(jnp.float32) * cos) + (rotate_half(q.astype(jnp.float32)) * sin)
        k = (k.astype(jnp.float32) * cos) + (rotate_half(k.astype(jnp.float32)) * sin)
        v = v.astype(jnp.float32)

        q = q * jnp.array(self.scale, dtype=jnp.float32)

        q, k, v = [t.transpose(0, 2, 1, 3) for t in (q, k, v)]

        out = jax.nn.dot_product_attention(q, k, v, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.o_proj(out.astype(jnp.float16))

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

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, dtype=jnp.float16):
        self.latent_dim = latent_dim
        self.num_scratch = SCRATCH_SLOTS 
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)).astype(jnp.float16) * 0.02)
        self.processor = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        
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
            
            new_z = self.processor(z_input, mask) 
            
            halt_logit = self.halt_head(new_z[:, seq_len:, :]).mean(axis=(1, 2))
            halt_prob = nnx.sigmoid(halt_logit)
            
            return new_z, (new_z, halt_prob)

        scan_fn = nnx.remat(nnx.scan(scan_step))
        _, (all_z, all_halts) = scan_fn(z_combined, jnp.arange(max_steps))
        
        p_remain = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - all_halts, axis=0)[:-1]], axis=0)
        step_weights = all_halts * p_remain

        last_step_halt_prob = all_halts[-1]
        remaining_prob = p_remain[-1] * (1.0 - last_step_halt_prob)
        step_weights = step_weights.at[-1].add(remaining_prob)

        # Average the embeddings of the sequence tokens
        all_z_seq = all_z[:, :, :seq_len, :]
        weighted_z = jnp.einsum('sb,sbnd->bnd', step_weights, all_z_seq)
        
        step_indices = jnp.arange(1, max_steps + 1)[:, None]
        ponder_cost = jnp.sum(step_weights * step_indices, axis=0)

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
def compute_gradients(graphdef, state, batch_tokens, key):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        preds, ponder_cost = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != PAD_TOKEN_ID)
        token_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
        
        avg_ponder = jnp.mean(ponder_cost)
        ponder_loss = PONDER_LAMBDA * avg_ponder
        
        return (token_loss + ponder_loss) * LOSS_SCALE, (token_loss, avg_ponder)

    (scaled_loss, (raw_ce, avg_p)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state)
    
    grads = jax.tree_util.tree_map(lambda x: x / LOSS_SCALE, grads)
    
    return grads, scaled_loss / LOSS_SCALE, raw_ce, avg_p

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
    accumulated_grads = None
    accum_loss, accum_ce, accum_p = 0.0, 0.0, 0.0

    while True:
        t0 = time.time()
        
        for i in range(ACCUMULATION_STEPS):
            key, subkey = jax.random.split(key)
            batch = data_gen.get_batch(BATCH_SIZE)
            if batch is None: break
            
            grads, loss, raw_ce, avg_p = compute_gradients(graphdef, state, batch, subkey)
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, grads)
            
            accum_loss += loss / ACCUMULATION_STEPS
            accum_ce += raw_ce / ACCUMULATION_STEPS
            accum_p += avg_p / ACCUMULATION_STEPS

        if batch is None: break

        accumulated_grads = jax.tree_util.tree_map(lambda x: x / ACCUMULATION_STEPS, accumulated_grads)

        updates, opt_state = optimizer_chain.update(accumulated_grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        
        accumulated_grads = None
        t_total = time.time() - t0

        if monitor.push(step, float(accum_loss)): break
        if step % CHECKPOINT_INTERVAL == 0:
            with open("checkpoint.pkl", "wb") as f:
                pickle.dump({"state": state, "opt_state": opt_state, "step": step}, f)
            
            print(f"Step {step:04d} | Agg Loss: {accum_loss:.4f} | Avg Steps: {accum_p:.2f} | Time: {t_total:.2f}s")
            
            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "loss", "ce", "avg_ponder", "t_total"])
                if not os.path.exists(history_file) or os.stat(history_file).st_size == 0: writer.writeheader()
                writer.writerow({
                    "step": int(step), 
                    "loss": f"{float(accum_loss):.4f}", 
                    "ce": f"{float(accum_ce):.4f}", 
                    "avg_ponder": f"{float(accum_p):.4f}", 
                    "t_total": f"{t_total:.2f}"
                })
            
        accum_loss, accum_ce, accum_p = 0.0, 0.0, 0.0
        step += 1