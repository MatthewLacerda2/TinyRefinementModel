import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

LATENT_DIM = 512     # raises memory and compute quadratically
BATCH_SIZE = 8
ACCUM_STEPS = 8
MAX_STEPS_LIMIT = 16 # raises memory linearly
MAX_SEQ_LEN = 64    # raises memory linearly and compute quadratically

#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Tiktoken integration
import tiktoken
ENCODING = tiktoken.get_encoding("cl100k_base")

# Define special tokens extending the tiktoken vocab
SOS_ID = ENCODING.n_vocab
EOS_ID = ENCODING.n_vocab + 1
PAD_ID = ENCODING.n_vocab + 2
VOCAB_SIZE = ENCODING.n_vocab + 3

class UniversalTaskWorld:
    """
    Generates INFINITE procedural arithmetic problems.
    Forces the model to learn ALGORITHMS, not just memorize answers.
    """
    # We remove the static "RAW_DATA" entirely.
    
    @staticmethod
    def get_input_dim():
        return MAX_SEQ_LEN

    @staticmethod
    def get_output_dim():
        return MAX_SEQ_LEN

    @staticmethod
    def generate_batch(key, batch_size, difficulty):
        # Handle tuple batch_size from your loop
        total_samples = batch_size[0] * batch_size[1] if isinstance(batch_size, tuple) else batch_size
        
        batch_q = []
        batch_a = []
        
        # Scale complexity with difficulty
        # Diff 0.1 -> 3+5
        # Diff 0.5 -> 23+89
        # Diff 0.9 -> 123*4
        max_val = int(10 + (difficulty * 990)) 
        
        # Use a localized random generator to avoid JAX/Numpy conflicts
        # (We use the JAX key to seed it for reproducibility)
        seed = int(jax.random.randint(key, (), 0, 1000000))
        rng = np.random.default_rng(seed)
        
        ops = ['+', '-', '*']
        
        for _ in range(total_samples):
            op = rng.choice(ops)
            a = rng.integers(0, max_val)
            b = rng.integers(0, max_val)
            
            # Curriculum: Start with simple addition
            if difficulty < 0.3: op = '+'
            elif difficulty < 0.6 and op == '*': op = '-'
            
            if op == '+': ans = a + b
            elif op == '-': ans = a - b
            elif op == '*': 
                # Keep multiplication inputs smaller to avoid huge numbers
                b = rng.integers(0, 10 + int(difficulty * 20)) 
                ans = a * b
                
            q_str = f"{a}{op}{b}="
            a_str = str(ans)
            
            # Encode using your TikToken setup (or character map if you switched)
            # Since you are using TikToken 'cl100k_base', we use it here:
            q_ids = ENCODING.encode(q_str)
            a_ids = ENCODING.encode(a_str)
            
            q_toks = [SOS_ID] + q_ids + [EOS_ID]
            a_toks = [SOS_ID] + a_ids + [EOS_ID]
            
            # Padding
            q_toks += [PAD_ID] * (MAX_SEQ_LEN - len(q_toks))
            a_toks += [PAD_ID] * (MAX_SEQ_LEN - len(a_toks))
            
            batch_q.append(q_toks[:MAX_SEQ_LEN])
            batch_a.append(a_toks[:MAX_SEQ_LEN])
            
        # Convert to JAX arrays
        q_arr = jnp.array(batch_q, dtype=jnp.int32)
        a_arr = jnp.array(batch_a, dtype=jnp.int32)
        
        # Reshape back to (ACCUM, BATCH, LEN)
        if isinstance(batch_size, tuple):
            q_arr = q_arr.reshape(batch_size[0], batch_size[1], MAX_SEQ_LEN)
            a_arr = a_arr.reshape(batch_size[0], batch_size[1], MAX_SEQ_LEN)
            
        return q_arr, a_arr


def apply_rotary_emb(q, k, sin, cos):
    # q, k: (batch, seq, heads, head_dim)
    # sin, cos: (1, seq, 1, head_dim)
    
    def rotate(x):
        # x_rot = [-x_odd, x_even] interleaved
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x.shape)
        return x * cos + x_rot * sin

    return rotate(q), rotate(k)

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        
        # --- RoPE Cache ---
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN)
        freqs = jnp.outer(t, inv_freq) # (MAX_SEQ_LEN, head_dim//2)
        
        # Cache sin/cos as fixed constants
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        b, s, d = x.shape
        # Project and reshape to (B, S, H, D)
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        
        # Apply RoPE
        # Slice the cache to the current sequence length
        sin = self.sin_cached[:s, :]
        cos = self.cos_cached[:s, :]
        
        # Pre-stack to match head_dim and broadcast for heads: (1, S, 1, head_dim)
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        
        q, k = apply_rotary_emb(q, k, sin_ext, cos_ext)
        
        # Transpose for attention: (B, H, S, D)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        weights = jax.nn.softmax(logits, axis=-1)
        
        out = jnp.matmul(weights, v)
        # Reshape to (B, S, D)
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


@nnx.jit
def train_step(model, optimizer, batch_q, batch_a, noise_keys, difficulty, baseline_error):
    # batch_q shape: (ACCUM_STEPS, BATCH_SIZE, SEQ_LEN)
    # noise_keys shape: (ACCUM_STEPS, 2)
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(current_baseline, inputs):
            (q_in, a_in, noise_key) = inputs
            
            m = nnx.merge(graphdef, state)
            preds, step_probs, pred_steps = m(q_in, MAX_STEPS_LIMIT, True, noise_key)
            
            # --- Masking Logic ---
            mask = (a_in != PAD_ID).astype(jnp.float32)
            
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=a_in)
            
            # Apply mask: Only learn from real words
            token_loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-6)
            
            # Accuracy only on non-padding tokens
            correct = (jnp.argmax(preds, axis=-1) == a_in)
            accuracy = jnp.sum(correct * mask) / (jnp.sum(mask) + 1e-6)
            
            # Latent Reasoning Penalty (encourage efficiency)
            steps_range = jnp.arange(step_probs.shape[1], dtype=jnp.float32)[None, :, None]
            actual_steps = jnp.sum(step_probs * steps_range, axis=1) # (batch, 1)
            planner_err = jnp.mean((pred_steps - actual_steps)**2)
            
            avg_steps = jnp.mean(actual_steps)
            
            loss = token_loss + (planner_err * 0.1) + (avg_steps * 0.005)

            metrics = {
                'loss': token_loss,
                'accuracy': accuracy,
                'planner_err': jnp.mean(jnp.abs(pred_steps - actual_steps)),
                'avg_steps': avg_steps,
            }
            return current_baseline, (loss * loss_scale, metrics)

        # Scan over batches and THEIR RESPECTIVE noise keys
        _, (losses_with_scale, all_metrics) = jax.lax.scan(scan_body, baseline_error, (batch_q, batch_a, noise_keys))
        
        avg_metrics = jax.tree.map(jnp.mean, all_metrics)
        return jnp.mean(losses_with_scale), (avg_metrics, baseline_error)

    (loss_s, (metrics, new_baseline)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    
    return loss_s / loss_scale, metrics, new_baseline


print(f"🚀 Initializing Universal Reasoner (Vocab Size={VOCAB_SIZE})...")
model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

key = jax.random.key(0)
accuracy_history = []
difficulty = 0.0
start_time = time.time()
step = 0

integral_error = 0.0
prev_error = 0.0
kp, ki, kd = 0.005, 0.0001, 0.01

baseline_error = 1.0

ckpt_path = "universal_reasoner.pkl"
if os.path.exists(ckpt_path):
    print(f"📂 Loading checkpoint from {ckpt_path}...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        difficulty = ckpt.get('difficulty', 0.0)
    print(f"✅ Loaded checkpoint weights")

log_file = "training_log.csv"
if os.path.exists(log_file):
    os.remove(log_file) 

print("🔥 Compiling Kernels (This may take 30s)...")
while True:
    step += 1
    key, subkey, noise_key = jax.random.split(key, 3)
    # Generate data
    accum_q, accum_a = UniversalTaskWorld.generate_batch(subkey, (ACCUM_STEPS, BATCH_SIZE), difficulty)
    # Generate unique noise keys for each accumulation step
    accum_noise_keys = jax.random.split(noise_key, ACCUM_STEPS)
    
    loss_val, step_metrics, baseline_error = train_step(model, optimizer, accum_q, accum_a, accum_noise_keys, difficulty, baseline_error)
    
    acc = float(step_metrics['accuracy'])
    accuracy_history.append(acc)
    if len(accuracy_history) > 50: accuracy_history.pop(0)
    avg_acc = sum(accuracy_history) / len(accuracy_history)
    
    planner_err = float(step_metrics['planner_err'])
    avg_steps = float(step_metrics['avg_steps'])

    # PID difficulty adjustment based on accuracy
    # Target 95% accuracy
    target_acc = 0.95
    error = target_acc - acc

    P = kp * error
    I = ki * integral_error
    D = kd * (error - prev_error)
    adjustment = P + I + D

    difficulty += adjustment
    difficulty = max(0.0, difficulty)

    integral_error += error
    prev_error = error
    
    if abs(error) > 0.1: integral_error = 0.0

    if step % 50 == 0:
        sps = 50 / (time.time() - start_time + 1e-6)
        
        print(f"Step {step} | Diff: {difficulty:.3f} | Acc: {avg_acc:.4f} | Loss: {float(loss_val):.4f} | Steps: {avg_steps:.2f} (PlnErr: {planner_err:.2f}) | {sps:.1f} steps/s")
        start_time = time.time()
        
        with open(log_file, "a") as f:
            f.write(f"{step},{difficulty:.4f},{float(loss_val):.4f},{avg_acc:.4f},{sps:.1f},{planner_err:.4f},{avg_steps:.2f}\n")
            
        if step % 1000 == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)

    if avg_acc > 0.98:
        print(f"\n🧠 AGENT ACHIEVED MASTERY at Step {step}!")
        print(f"   - Final Accuracy: {avg_acc:.4f}")
        print(f"   - Difficulty: {difficulty:.2f}")
        
        with open("mastered_reasoner.pkl", "wb") as f:
            pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)
        break

