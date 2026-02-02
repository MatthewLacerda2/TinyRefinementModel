import os
import pickle
import json
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.70' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- 1. MODEL WITH RECOGNITION CIRCUIT ---
class AdaptiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # Weights are float32 by default in NNX
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs)
        self.refine_fc = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim * 2, 3, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z, target):
        # Manually cast inputs to fp16 for the 'compute' phase
        z = z.astype(jnp.float16)
        target = target.astype(jnp.float16)
        
        combined = jnp.concatenate([z, target], axis=-1)
        
        # NNX layers will automatically cast weights to match the input 
        # (or you can wrap the call in a jax.jit with a dtype)
        logits = self.recog_fc(combined)
        h = jax.nn.gelu(self.fc1(combined))
        delta = self.refine_fc(h)
        next_z = self.norm(z + 0.1 * delta)
        p = jax.nn.sigmoid(self.halt_fc(next_z))
        
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
            
            new_halt_decision = (p_halt.squeeze(axis=-1) > 0.5)
            still_active = active_mask & (~new_halt_decision)
            z_updated = jnp.where(active_mask[:, None], next_z_raw, z)
            new_step_counts = step_counts + active_mask.astype(jnp.float16)
            
            return (z_updated, still_active, new_step_counts), logits

        (final_z, _, per_sample_steps), all_logits = jax.lax.scan(
            scan_fn, (z_init, active_mask_init, step_counts_init), None, length=16
        )
        
        # Reconstruction Loss
        recon_loss = jnp.mean((final_z - batch["target"]) ** 2)
        
        # Recognition Loss: Grade the first step's identification
        labels = jnp.full((batch_size,), batch['level'], dtype=jnp.int32)
        recog_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(all_logits[0], labels))
        
        step_penalty = jnp.mean(per_sample_steps) * 0.001
        
        return (recon_loss + recog_loss + step_penalty) * loss_scale
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    # Unscale the gradients before the optimizer update
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    return loss / loss_scale, grads

# --- 3. UPDATED INFRASTRUCTURE ---
def generate_complex_math(key, batch_size, latent_dim, step):
    level = min(step // 1000, 2)
    num_ops = 2 + (level * 3)
    op_limit = [2, 3, 4][level]
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
    return x, target, level

micro_batch = 64
accum_steps = 32 
latent_dim = 768
model = AdaptiveRefiner(latent_dim, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)), wrt=nnx.Param)

# --- 4. TRAINING LOOP ---
history_path = "training_history.json"
ckpt_path = "model_ckpt.pkl"
start_step = 0
history = []

if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        cp = pickle.load(f)
        nnx.update(model, cp['model'])
        nnx.update(optimizer, cp['optimizer'])
        start_step = cp['step'] + 1
    print(f"Resumed at step {start_step}")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

try:
    key = jax.random.key(start_step)
    for step in range(start_step, 10001):
        total_loss = 0
        accum_grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
        for _ in range(accum_steps):
            key, subkey = jax.random.split(key)
            x, target, level = generate_complex_math(subkey, micro_batch, latent_dim, step)
            batch = {'input': x, 'target': target, 'level': level}
            loss, grads = compute_grads(model, batch)
            accum_grads = jax.tree.map(lambda a, g: a + g / accum_steps, accum_grads, grads)
            total_loss += loss / accum_steps
        optimizer.update(model, accum_grads)
        if step % 100 == 0:
            print(f"Step {step} | Level: {level} | Loss: {total_loss:.4f}")
            
            # Save History
            history.append({"step": int(step), "loss": float(total_loss)})
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            # Save Checkpoint
            state = {'model': nnx.state(model), 'optimizer': nnx.state(optimizer), 'step': step}
            with open(ckpt_path, 'wb') as f:
                pickle.dump(state, f)
except Exception as e:
    print(f"Halted: {e}")