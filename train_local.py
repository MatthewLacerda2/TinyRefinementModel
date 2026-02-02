import os
import pickle
import json
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
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs)
        self.refine_fc = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim * 2, 3, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z, target):
        z = z.astype(jnp.float16)
        target = target.astype(jnp.float16)
        
        combined = jnp.concatenate([z, target], axis=-1)
        
        logits = self.recog_fc(combined).astype(jnp.float16)
        h = jax.nn.gelu(self.fc1(combined)).astype(jnp.float16)
        delta = self.refine_fc(h).astype(jnp.float16)
        
        next_z = self.norm(z + 0.1 * delta).astype(jnp.float16)
        p = jax.nn.sigmoid(self.halt_fc(next_z)).astype(jnp.float16)
        
        return next_z, p, logits

# --- 2. UPDATED DIFFERENTIABLE SCAN ---
@nnx.jit
def compute_grads(model, batch):
    # Scale factor (e.g., 2^15) to prevent gradients from becoming zero
    loss_scale = 32768.0 
    
    def loss_fn(model):
        z_init = jnp.ones_like(batch['input'], dtype=jnp.float16) * 0.01
        batch_size = batch['input'].shape[0]
        active_mask_init = jnp.ones((batch_size,), dtype=bool)
        step_counts_init = jnp.zeros((batch_size,), dtype=jnp.float16)
        
        def scan_fn(carry, _):
            z, active_mask, step_counts = carry
            next_z_raw, p_halt, logits = model(z, batch["target"])
            
            # Ensure the comparison doesn't change carry types
            new_halt_decision = (p_halt.squeeze(axis=-1) > 0.5)
            still_active = active_mask & (~new_halt_decision)
            
            # This where clause is where the error usually triggers
            z_updated = jnp.where(active_mask[:, None], next_z_raw, z).astype(jnp.float16)
            new_step_counts = (step_counts + active_mask.astype(jnp.float16)).astype(jnp.float16)
            
            return (z_updated, still_active, new_step_counts), logits

        (final_z, _, per_sample_steps), all_logits = jax.lax.scan(
            scan_fn, (z_init, active_mask_init, step_counts_init), None, length=16
        )
        
        # Reconstruction Loss
        recon_loss = jnp.mean((final_z - batch["target"]) ** 2)
        
        labels = jnp.full((batch_size,), batch['level'], dtype=jnp.int32)
        # Better supervision for the recognition circuit
        recog_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(all_logits, labels[None, :]))
        
        step_penalty = jnp.mean(per_sample_steps) * 0.001
        
        return (recon_loss + recog_loss + step_penalty) * loss_scale
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    # Unscale the gradients before the optimizer update
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    return loss / loss_scale, grads

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

micro_batch = 64
accum_steps = 32 
latent_dim = 768
model = AdaptiveRefiner(latent_dim, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)), wrt=nnx.Param)

# Create a JIT-compiled function to handle the entire accumulation block
@nnx.jit(static_argnums=(3, 4, 6))
def train_step(model, optimizer, subkeys, micro_batch, latent_dim, step, current_num_ops):
    def accum_loss_fn(model):
        def scan_body(carry, key):
            x, target, level_label = generate_complex_math(key, micro_batch, latent_dim, step, current_num_ops)
            batch = {'input': x, 'target': target, 'level': level_label}
            loss, grads = compute_grads(model, batch)
            return carry, (loss, grads)

        # Use jax.lax.scan to keep the accumulation loop on-device
        _, (losses, grads) = jax.lax.scan(scan_body, None, subkeys)
        avg_loss = jnp.mean(losses)
        avg_grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)
        return avg_loss, avg_grads

    loss, grads = accum_loss_fn(model)
    optimizer.update(model, grads)
    return loss

# --- 4. TRAINING LOOP ---
history_path = "training_history.json"
ckpt_path = "model_ckpt.pkl"
start_step = 0
current_num_ops = 2
loss_buffer = []
history = []

if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        cp = pickle.load(f)
        nnx.update(model, cp['model'])
        nnx.update(optimizer, cp['optimizer'])
        start_step = cp['step'] + 1
        current_num_ops = cp.get('num_ops', cp.get('level', 0) * 3 + 2)
    print(f"Resumed at step {start_step} | Ops: {current_num_ops}")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

try:
    key = jax.random.key(start_step)
    for step in range(start_step, 10001):
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

        if step % 25 == 0 or (step > 9975 and step % 5 == 0):
            print(f"Step {step} | Ops: {current_num_ops} | Loss: {total_loss:.4f}")
            
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