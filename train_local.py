import os
import pickle
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- 1. MODEL (Mixed Precision Enabled) ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # Set dtype to float16 for memory savings
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs, dtype=jnp.float16)
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs, dtype=jnp.float16)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=jnp.float16)

    def __call__(self, z, target):
        combined = jnp.concatenate([z, target], axis=-1)
        h = jax.nn.gelu(self.fc1(combined))
        delta = self.fc2(h) 
        return self.norm(z + 0.1 * delta)

# --- 2. TRAINING STEP (With Accumulation) ---
@nnx.jit
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        # Ensure initial state matches precision
        z = jnp.ones_like(batch['input'], dtype=jnp.float16) * 0.01
        def step_fn(current_z, _):
            next_z = model(current_z, batch["target"])
            return next_z, next_z
        zs, _ = jax.lax.scan(step_fn, z, None, length=16)
        return jnp.mean((zs[-1] - batch["target"]) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(model, grads)
    metrics.loss.update(loss=loss)
    return loss

# --- 3. CHECKPOINT SYSTEM ---
def save_ckpt(model, optimizer, step, path="model_ckpt.pkl"):
    state = {'model': nnx.state(model), 'optimizer': nnx.state(optimizer), 'step': step}
    with open(path, 'wb') as f:
        pickle.dump(state, f)

# --- 4. EXECUTION ---
micro_batch = 128 
accum_steps = 4 # 128 * 4 = 512 effective batch
latent_dim = 768
rngs = nnx.Rngs(42)
model = RecursiveRefiner(latent_dim, rngs)

# Prevent dead zones
model.fc1.kernel[...] *= 0.01 
model.fc2.kernel[...] *= 0.01

optimizer = nnx.Optimizer(model, optax.adam(5e-4), wrt=nnx.Param)
metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))

ckpt_path = "model_ckpt.pkl"
start_step = 0
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        cp = pickle.load(f)
        nnx.update(model, cp['model'])
        nnx.update(optimizer, cp['optimizer'])
        start_step = cp['step'] + 1
    print(f"Resumed at {start_step}")

try:
    key = jax.random.key(start_step)
    for step in range(start_step, 5001):
        # Accumulation loop to simulate 512 batch
        for _ in range(accum_steps):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (micro_batch, latent_dim), dtype=jnp.float16)
            batch = {'input': x, 'target': x * 2.0}
            loss = train_step(model, optimizer, metrics, batch)

        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss:.6f}")
            save_ckpt(model, optimizer, step, ckpt_path)
            metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))
            
except Exception as e:
    print(f"Halted: {e}")