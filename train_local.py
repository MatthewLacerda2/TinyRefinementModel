import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

MAX_N = 64
LATENT_DIM = 512
BATCH_SIZE = 128
ACCUM_STEPS = 2

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

MAX_SEQ_LEN = 32

# Tiktoken integration
import tiktoken
ENCODING = tiktoken.get_encoding("cl100k_base")

# Define special tokens extending the tiktoken vocab
SOS_ID = ENCODING.n_vocab
EOS_ID = ENCODING.n_vocab + 1
PAD_ID = ENCODING.n_vocab + 2
VOCAB_SIZE = ENCODING.n_vocab + 3

# Precompute valid ids for random generation (digits and symbols)
# We select single-token representations for 0-9 and operators
_DIGITS = [str(i) for i in range(10)]
_SYMBOLS = list("+-*()= ")
_VALID_TOKENS_STR = _DIGITS + _SYMBOLS
# Ensure they map to single tokens (cl100k usually does for these)
VALID_TOKEN_IDS_LIST = []
for t in _VALID_TOKENS_STR:
    ids = ENCODING.encode(t)
    if len(ids) == 1:
        VALID_TOKEN_IDS_LIST.append(ids[0])
    else:
        # Fallback or just take first? For digits/ops in cl100k they are single.
        VALID_TOKEN_IDS_LIST.append(ids[0])

VALID_TOKEN_IDS = jnp.array(VALID_TOKEN_IDS_LIST, dtype=jnp.int32)


class UniversalTaskWorld:
    # A tiny "Physics Textbook" for the model to memorize
    RAW_DATA = [
        ("F = m * a", "Force equals mass times acceleration"),
        ("E = m * c^2", "Energy equals mass times light speed squared"),
        ("v = d / t", "Velocity is distance divided by time"),
        ("What is gravity?", "Gravity is a force of attraction"),
        ("Newton's First Law", "Objects in motion stay in motion"),
    ]
    
    # --- PRE-COMPUTE DATA (The Fix) ---
    # We tokenize EVERYTHING once at startup using Python
    # This creates static numpy arrays that JAX can easily handle
    
    print("📚 Pre-tokenizing Physics Textbook...")
    _QUESTIONS = []
    _ANSWERS = []
    
    for q, a in RAW_DATA:
        # Tokenize Question
        q_ids = ENCODING.encode(q)
        q_tokens = [SOS_ID] + q_ids + [EOS_ID]
        # Pad Question
        if len(q_tokens) < MAX_SEQ_LEN:
            q_tokens += [PAD_ID] * (MAX_SEQ_LEN - len(q_tokens))
        else:
            q_tokens = q_tokens[:MAX_SEQ_LEN]
            
        # Tokenize Answer
        a_ids = ENCODING.encode(a)
        a_tokens = [SOS_ID] + a_ids + [EOS_ID]
        # Pad Answer
        if len(a_tokens) < MAX_SEQ_LEN:
            a_tokens += [PAD_ID] * (MAX_SEQ_LEN - len(a_tokens))
        else:
            a_tokens = a_tokens[:MAX_SEQ_LEN]
            
        _QUESTIONS.append(q_tokens)
        _ANSWERS.append(a_tokens)
        
    # Convert to JAX Arrays
    QUESTIONS_DB = jnp.array(_QUESTIONS, dtype=jnp.int32)
    ANSWERS_DB = jnp.array(_ANSWERS, dtype=jnp.int32)
    print("✅ Textbook Loaded into VRAM.")

    @staticmethod
    def get_input_dim():
        return MAX_SEQ_LEN

    @staticmethod
    def get_output_dim():
        return MAX_SEQ_LEN

    @staticmethod
    def generate_batch(key, batch_size, difficulty, steps=None):
        # NOW this is 100% JAX-compatible!
        # It's just picking rows from a matrix.
        indices = jax.random.randint(key, (batch_size,), 0, len(UniversalTaskWorld.RAW_DATA))
        
        batch_q = UniversalTaskWorld.QUESTIONS_DB[indices]
        batch_a = UniversalTaskWorld.ANSWERS_DB[indices]
        
        return batch_q, batch_a


class LatentReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=latent_dim, dtype=dtype, rngs=rngs)
        self.norm1 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.fc = nnx.Linear(latent_dim, latent_dim, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)

    def __call__(self, z):
        z_norm = self.norm1(z)
        z_attn = self.attn(z_norm, z_norm, decode=False)
        z = z + z_attn
        
        z_next = self.fc(self.norm2(z))
        z = z + nnx.gelu(z_next)
        return z

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        dtype = jnp.bfloat16
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=dtype, rngs=rngs)
        self.pos_embed = nnx.Linear(1, latent_dim, dtype=dtype, rngs=rngs)
        
        self.processor = LatentReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        
        self.norm = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)

    def __call__(self, tokens, max_steps=16, training=False, key=None):
        # tokens: (batch, seq_len)
        z = self.embed(tokens)
        
        # Add simple positional encoding
        seq_len = tokens.shape[1]
        pos = jnp.arange(seq_len, dtype=jnp.float32)[:, None] / seq_len
        z = z + self.pos_embed(pos[None, :, :])
        
        batch_size = z.shape[0]
        
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
            curr_z, step_idx, run_prob, w_out, w_z = carry
            
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
            
            new_out = w_out + (p[:, :, None] * self.decoder(next_z))
            new_z = w_z + (p[:, :, None] * next_z)
            
            return (next_z, step_idx + 1, run_prob + p, new_out, new_z), p

        init_carry = (
            z,
            0,
            jnp.zeros((batch_size, 1), dtype=jnp.bfloat16),
            jnp.zeros((batch_size, seq_len, VOCAB_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((batch_size, seq_len, self.latent_dim), dtype=jnp.bfloat16)
        )
        
        (final_z, _, final_prob, w_out, w_z), step_probs = jax.lax.scan(
            refine_step, init_carry, step_keys, length=max_steps
        )
        
        step_probs = jnp.swapaxes(step_probs, 0, 1)
        rem = 1.0 - final_prob
        w_out = w_out + (rem[:, :, None] * self.decoder(final_z))
        
        # Output is (batch, seq_len, vocab_size)
        return w_out.astype(jnp.float32), step_probs, predicted_steps


@nnx.jit(static_argnums=(3, 4))
def train_step(model, optimizer, subkeys, micro_batch, thinking_steps, difficulty, baseline_error):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(current_baseline, key):
            data_key, noise_key = jax.random.split(key)
            
            m = nnx.merge(graphdef, state)
            inputs, targets = jax.lax.stop_gradient(
                UniversalTaskWorld.generate_batch(data_key, micro_batch, difficulty)
            )
            
            preds, step_probs, pred_steps = m(inputs, thinking_steps, True, noise_key)
            
            # --- NEW: Masking Logic ---
            # Create a mask where Valid Tokens = 1.0, Padding = 0.0
            mask = (targets != PAD_ID).astype(jnp.float32)
            
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
            
            # Apply mask: Only learn from real words
            token_loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-6)
            
            # Accuracy only on non-padding tokens
            correct = (jnp.argmax(preds, axis=-1) == targets)
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

        _, (losses_with_scale, all_metrics) = jax.lax.scan(scan_body, baseline_error, subkeys)
        
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
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, ACCUM_STEPS)
    
    thinking_steps = int(5 + (difficulty * 15))
    
    loss_val, step_metrics, baseline_error = train_step(model, optimizer, subkeys, BATCH_SIZE, thinking_steps, difficulty, baseline_error)
    
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

