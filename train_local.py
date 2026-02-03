

import os
import pickle
import json
import time
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- 1. MODEL WITH RECOGNITION CIRCUIT ---
class AdaptiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # --- The New Predictor Sub-Model ---
        # Guesses total steps needed (scalar) based on the initial problem state
        self.predictor_fc = nnx.Sequential(
            nnx.Linear(latent_dim * 2, latent_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(latent_dim // 2, 1, rngs=rngs)
        )
        
        # --- Main Refinement Circuit ---
        # Added +1 to input dim to accommodate the 'current_step' counter
        self.fc1 = nnx.Linear(latent_dim * 2 + 1, latent_dim * 2, rngs=rngs)
        self.refine_fc = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim * 2, 3, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def predict_complexity(self, z, target):
        """Initial guess of how many steps this problem requires."""
        combined = jnp.concatenate([z, target], axis=-1)
        return jax.nn.softplus(self.predictor_fc(combined)) # Ensure positive steps

    def predict_recognition(self, z, target):
        z = z.astype(jnp.float16)
        target = target.astype(jnp.float16)
        combined = jnp.concatenate([z, target], axis=-1)
        return self.recog_fc(combined)

    def __call__(self, z, target, step_count):
        z = z.astype(jnp.float16)
        target = target.astype(jnp.float16)
        
        # Inject the step counter as a feature
        # This tells the model: "You are currently on step X"
        step_feat = jnp.array([step_count], dtype=jnp.float16).reshape(-1, 1)
        # If batching, broadcast step_feat to (batch_size, 1)
        step_feat = jnp.broadcast_to(step_feat, (z.shape[0], 1))
        
        combined = jnp.concatenate([z, target, step_feat], axis=-1)
        
        logits = self.recog_fc(combined[:, :-1]).astype(jnp.float16)
        h = jax.nn.gelu(self.fc1(combined)).astype(jnp.float16)
        delta = self.refine_fc(h).astype(jnp.float16)
        
        next_z = self.norm(z + 0.1 * delta).astype(jnp.float16)
        p = jax.nn.sigmoid(self.halt_fc(next_z)).astype(jnp.float16)
        
        return next_z, p, logits

# --- 2. MODEL LOSS FUNCTION ---
def run_model_loss(model, batch, key):
    loss_scale = 32768.0
    batch_size = batch['input'].shape[0]
    z_init = (jax.random.normal(key, batch['input'].shape) * 0.02 + 0.01).astype(jnp.float16)
    
    # 1. PLAN: Predict complexity before starting
    # This acts as the 'prior' or 'guess'
    predicted_steps = model.predict_complexity(z_init, batch["target"])

    # 2. RECOGNIZE: Predict problem level/difficulty (Moved outside loop)
    initial_logits = model.predict_recognition(z_init, batch["target"])
    recog_loss = optax.softmax_cross_entropy_with_integer_labels(
        initial_logits.astype(jnp.float32),
        batch['level']
    ).mean()
    
    graphdef, state = nnx.split(model)

    initial_carry = (
        z_init, 
        jnp.ones((batch_size,), dtype=bool), 
        jnp.zeros((batch_size,), dtype=jnp.float16),
        0    # loop counter
    )

    def body_fn(carry):
        z, active_mask, step_counts, step = carry
        m = nnx.merge(graphdef, state)
        
        # Pass the current loop 'step' into the model
        next_z_raw, p_halt, _ = m(z, batch["target"], step)
        
        # If the model is an expert, p_halt will trigger early.
        # If the model is struggling, it knows it's at 'step' N and can adjust.
        new_halt_decision = (p_halt.squeeze(axis=-1) > 0.5)
        still_active = active_mask & (~new_halt_decision)
        
        z_updated = jnp.where(active_mask[:, None], next_z_raw, z).astype(jnp.float16)
        new_step_counts = step_counts + active_mask.astype(jnp.float16)

        return (z_updated, still_active, new_step_counts, step + 1)

    final_z, _, per_sample_steps, total_steps = jax.lax.while_loop(
        lambda c: jnp.any(c[1]) & (c[3] < 512), body_fn, initial_carry
    )
    
    # --- New Loss Components ---
    recon_loss = jnp.mean((final_z - batch["target"]) ** 2)
    
    # Penalty 1: Standard step penalty
    step_penalty = jnp.mean(per_sample_steps) * 0.001
    
    # Penalty 2: Prediction Accuracy
    # Trains the predictor to actually match the real steps taken
    prediction_loss = jnp.mean((predicted_steps - per_sample_steps)**2) * 0.001
    
    return (recon_loss + step_penalty + prediction_loss + 0.1 * recog_loss) * loss_scale

# --- 3. UPDATED INFRASTRUCTURE ---
@jax.jit(static_argnums=(1, 2, 4)) 
def generate_complex_math(key, batch_size, latent_dim, step, num_ops):
    op_limit = min(2 + (num_ops - 2) // 3, 4)
    level_label = jnp.minimum(step // 1000, 2).astype(jnp.int32)
    x = jax.random.normal(key, (batch_size, latent_dim), dtype=jnp.float16)
    target = x
    key_ops, key_vals = jax.random.split(key)
    ops = jax.random.randint(key_ops, (num_ops,), 0, op_limit)
    vals = jax.random.uniform(key_vals, (num_ops,), minval=0.5, maxval=1.5)
    for i in range(num_ops):
        target = jax.lax.switch(ops[i], [
            lambda v: target + v, lambda v: target - v,
            lambda v: target * v, lambda v: target / (v + 1e-3)
        ], vals[i])
    return x, target, level_label

micro_batch = 128
accum_steps = 16 
latent_dim = 768
model = AdaptiveRefiner(latent_dim, nnx.Rngs(42))

# --- 1. Define Schedules with Internal Type Casting ---
# Using optax.cosine_decay (the raw function) is faster and less strict
def muon_lr_sched(step):
    # This cast happens inside the schedule, satisfying Chex immediately
    return optax.cosine_decay(init_value=0.02, decay_steps=20000)(step.astype(jnp.int32))

def adam_lr_sched(step):
    return optax.cosine_decay(init_value=3e-4, decay_steps=20000)(step.astype(jnp.int32))

muon_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.contrib.muon(learning_rate=muon_lr_sched))
adam_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=adam_lr_sched))

# Create a partition function to separate 2D weights from 1D params
def label_fn(path, params):
    path_names = [str(p.key if hasattr(p, "key") else p) for p in path]

    if "kernel" in path_names and getattr(params, "ndim", 0) == 2:
        return "muon"
    return "adam"

def param_labels(params):
    return jax.tree_util.tree_map_with_path(label_fn, params)

# --- 3. Optimized train_step ---
# Keep step dynamic (remove 5 from static_argnums) to avoid constant re-compilation
@nnx.jit(static_argnums=(3, 4, 6))
def train_step(model, optimizer, subkeys, micro_batch, latent_dim, step, current_num_ops):
    loss_scale = 32768.0
    
    # Cast here for use in generate_complex_math
    step_int = step.astype(jnp.int32)
    
    # We pass the model directly; nnx.value_and_grad handles the state tracing
    def loss_fn(model):
        # 1. Split state so it can be safely closed over by the scan
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            # 2. Re-merge inside the scan to stay within the current trace level
            m = nnx.merge(graphdef, state)
            x, target, level_label = generate_complex_math(key, micro_batch, latent_dim, step_int, current_num_ops)
            batch = {'input': x, 'target': target, 'level': level_label}
            # run_model_loss now handles its own internal split/merge
            loss = run_model_loss(m, batch, key) 
            return None, loss

        _, losses = jax.lax.scan(scan_body, None, subkeys)
        return jnp.mean(losses)

    # Differentiate the entire accumulation in one go
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Unscale the gradients and loss before optimizer update
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    return loss / loss_scale

# --- 4. TRAINING LOOP ---
history_path = "training_history.json"
ckpt_path = "model_ckpt.pkl"
start_step = 0
current_num_ops = 2
loss_buffer = []
history = []

# --- 2. Robust Loading with Auto-Reset ---
if os.path.exists(ckpt_path):
    try:
        with open(ckpt_path, 'rb') as f:
            cp = pickle.load(f)
        
        # Update weights
        nnx.update(model, cp['model'])
        
        # Inject state directly into a new optimizer instance
        optimizer = nnx.Optimizer(
            model, 
            optax.multi_transform({'muon': muon_tx, 'adam': adam_tx}, param_labels),
            wrt=nnx.Param,
            state=cp['optimizer']
        )
        
        start_step = cp['step'] + 1
        current_num_ops = cp.get('num_ops', 2)
        print(f"✅ Successfully resumed at step {start_step}")
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        
    except (ValueError, KeyError, TypeError) as e:
        print(f"⚠️ Checkpoint incompatible or corrupt: {e}")
        print(f"🗑️ Deleting old checkpoint and starting fresh...")
        os.remove(ckpt_path)
        # Initialize fresh optimizer if load fails
        optimizer = nnx.Optimizer(
            model, 
            optax.multi_transform({'muon': muon_tx, 'adam': adam_tx}, param_labels),
            wrt=nnx.Param
        )
else:
    # Standard fresh start
    optimizer = nnx.Optimizer(
        model, 
        optax.multi_transform({'muon': muon_tx, 'adam': adam_tx}, param_labels),
        wrt=nnx.Param
    )

try:
    key = jax.random.key(start_step)
    last_print_time = time.time()
    for step in range(start_step, 20001):
        # Pre-split all keys for the accumulation steps at once
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, accum_steps)

        # Optimized JIT-compiled training step
        total_loss = train_step(model, optimizer, subkeys, micro_batch, latent_dim, step, current_num_ops)

        # Add current loss to buffer
        loss_buffer.append(float(total_loss))
        if len(loss_buffer) > 100: loss_buffer.pop(0)

        # CHECK FOR PLATEAU: If we've trained at least 100 steps at this level
        # and the average loss is low enough (e.g., < 2.5)
        if len(loss_buffer) == 100 and sum(loss_buffer)/100 < 2.5:
            current_num_ops += 1
            loss_buffer = [] # Reset buffer for the new level
            print(f"--- LEVEL UP! Mastery reached. Now using {current_num_ops} ops ---")

        if step % 100 == 0:
            current_time = time.time()
            elapsed = current_time - last_print_time
            print(f"Step {step} | Ops: {current_num_ops} | Loss: {total_loss:.4f} | Time: {elapsed:.2f}s")
            last_print_time = current_time
            
            # Save History
            history.append({"step": int(step), "loss": float(total_loss)})
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            # Save Checkpoint
            state = {'model': nnx.state(model), 'optimizer': nnx.state(optimizer), 'step': step, 'num_ops': current_num_ops}
            with open(ckpt_path, 'wb') as f:
                pickle.dump(state, f)
except Exception as e:
    print(f"Halted: {e}")