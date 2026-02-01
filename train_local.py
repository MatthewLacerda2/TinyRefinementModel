import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- 1. THE RECURSIVE BRAIN (Velocity Version) ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z, target):
        combined = jnp.concatenate([z, target], axis=-1)
        h = jax.nn.gelu(self.fc1(combined))
        
        # Latent Velocity calculation (the delta)
        delta = self.fc2(h) 
        
        return self.norm(z + 0.1 * delta)

# --- 2. THE TRAINING LOOP ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        z = jnp.ones_like(batch['input']) * 0.01

        def step_fn(current_z, _):
            next_z = model(current_z, batch["target"])
            return next_z, next_z

        zs, _ = jax.lax.scan(step_fn, z, None, length=16)

        loss = jnp.mean((zs[-1] - batch["target"]) ** 2)
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(model, grads)
    metrics.loss.update(loss=loss)
    return loss

# --- 3. EXECUTION: SATURATING THE RTX 2060 ---
batch_size = 512
latent_dim = 768
rngs = nnx.Rngs(42)
model = RecursiveRefiner(latent_dim, rngs)

model.fc1.kernel[...] *= 0.01 
model.fc2.kernel[...] *= 0.01

optimizer = nnx.Optimizer(model, optax.adam(5e-4), wrt=nnx.Param)
metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))

print(f"RefineMath V1: Starting Algebraic Discovery (Batch: {batch_size})")
try:
    key = jax.random.key(0)
    for step in range(5001):
        key, subkey = jax.random.split(key)
        
        x = jax.random.normal(subkey, (batch_size, latent_dim))
        y = x * 2.0  
        batch = {'input': x, 'target': y}

        loss = train_step(model, optimizer, metrics, batch)

        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss:.6f}")
            metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))
            
except Exception as e:
    print(f"Halted on Arch: {e}")