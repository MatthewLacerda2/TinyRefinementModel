import os
import pickle
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- 1. MODEL ---
class AdaptiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs, dtype=jnp.float16)
        self.refine_fc = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs, dtype=jnp.float16)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs, dtype=jnp.float16)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=jnp.float16)

    def __call__(self, z, target):
        combined = jnp.concatenate([z, target], axis=-1)
        h = jax.nn.gelu(self.fc1(combined))
        
        delta = self.refine_fc(h)
        next_z = self.norm(z + 0.1 * delta)
        
        p = jax.nn.sigmoid(self.halt_fc(next_z))
        
        return next_z, p

# --- 2. TRUE ACCUMULATION STEP ---
@nnx.jit
def compute_grads(model, batch):
    def loss_fn(model):
        # Initial thought state
        z_init = jnp.ones_like(batch['input'], dtype=jnp.float16) * 0.01
        batch_size = batch['input'].shape[0]
        active_mask_init = jnp.ones((batch_size,), dtype=bool)
        step_counts_init = jnp.zeros((batch_size,), dtype=jnp.float16)
        
        def cond_fun(state):
            z, step, active_mask, step_counts = state
            return (step < 32) & jnp.any(active_mask)

        def body_fun(state):
            z, step, active_mask, step_counts = state
            next_z_raw, p_halt = model(z, batch["target"])
            
            new_halt_decision = (p_halt.squeeze(axis=-1) > 0.5)
            still_active = active_mask & (~new_halt_decision)
            
            z_updated = jnp.where(active_mask[:, None], next_z_raw, z)
            
            # Increment step counts for those who were active
            new_step_counts = step_counts + active_mask.astype(jnp.float16)
            
            return z_updated, step + 1, still_active, new_step_counts

        # Run the actual adaptive thought process
        final_z, total_steps, _, per_sample_steps = jax.lax.while_loop(
            cond_fun, body_fun, (z_init, 0, active_mask_init, step_counts_init)
        )
        
        # 1. Reconstruction Loss: Did it reach the truth?
        recon_loss = jnp.mean((final_z - batch["target"]) ** 2)
        
        # 2. Efficiency Penalty: Encourage solving in fewer steps (per sample)
        step_penalty = jnp.mean(per_sample_steps) * 0.001
        
        return recon_loss + step_penalty
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads


# --- 3. DATA GENERATION ---
def generate_complex_math(key, batch_size, latent_dim, num_ops=8):
    x = jax.random.normal(key, (batch_size, latent_dim), dtype=jnp.float16)
    target = x
    
    key_ops, key_vals = jax.random.split(key)
    ops = jax.random.randint(key_ops, (num_ops,), 0, 4)
    # Keep constants near 1.0 for numerical stability
    vals = jax.random.uniform(key_vals, (num_ops,), minval=0.8, maxval=1.2)
    
    for i in range(num_ops):
        target = jax.lax.switch(ops[i], [
            lambda v: target + v,
            lambda v: target - v,
            lambda v: target * v,
            lambda v: target / (v + 1e-4) # Safety epsilon
        ], vals[i])
        
    return x, target


# --- 4. EXECUTION ---
micro_batch = 128
accum_steps = 4 
latent_dim = 768
rngs = nnx.Rngs(42)
model = AdaptiveRefiner(latent_dim, rngs)
optimizer = nnx.Optimizer(model, optax.adam(5e-4), wrt=nnx.Param)

ckpt_path = "model_ckpt.pkl"
start_step = 0

# Check & Resume
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        cp = pickle.load(f)
        nnx.update(model, cp['model'])
        nnx.update(optimizer, cp['optimizer'])
        start_step = cp['step'] + 1
    print(f"Resumed at step {start_step}")

try:
    key = jax.random.key(start_step)
    for step in range(start_step, 5001):
        total_loss = 0
        accum_grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
        
        for _ in range(accum_steps):
            key, subkey = jax.random.split(key)
            x, target = generate_complex_math(subkey, micro_batch, latent_dim)
            batch = {'input': x, 'target': target}
            
            loss, grads = compute_grads(model, batch)
            accum_grads = jax.tree.map(lambda a, g: a + g / accum_steps, accum_grads, grads)
            total_loss += loss / accum_steps

        optimizer.update(model, accum_grads)

        if step % 20 == 0:
            print(f"Step {step} | Loss: {total_loss:.6f}")
            # Save Checkpoint
            state = {'model': nnx.state(model), 'optimizer': nnx.state(optimizer), 'step': step}
            with open(ckpt_path, 'wb') as f:
                pickle.dump(state, f)
            
except Exception as e:
    print(f"Halted: {e}")