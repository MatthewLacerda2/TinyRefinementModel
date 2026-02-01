import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time

# --- 1. THE RECURSIVE BRAIN (Velocity Version) ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # We concatenate [current_state, target] so input is latent_dim * 2
        self.fc1 = nnx.Linear(latent_dim * 2, latent_dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z, target):
        # The 'Holistic Glance': model sees where it is vs where it needs to be
        combined = jnp.concatenate([z, target], axis=-1)
        h = jax.nn.gelu(self.fc1(combined))
        
        # This is the 'Latent Velocity' (delta)
        delta = self.fc2(h) 
        
        # Update z by a fraction of the velocity (Euler integration step)
        return self.norm(z + 0.1 * delta)

# --- 2. THE TRAINING LOOP ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        # Start at a fixed 'noisy' state (the 'concept' starts here)
        z = jnp.ones_like(batch['input']) * 0.01

        def step_fn(current_z, _):
            next_z = model(current_z, batch["target"])
            return next_z, next_z

        # Recursive Loop: 16 steps of 'thinking'
        zs, _ = jax.lax.scan(step_fn, z, None, length=16)

        # We penalize based on the final 'converged' thought
        loss = jnp.mean((zs[-1] - batch["target"]) ** 2)
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(model, grads)
    metrics.loss.update(loss=loss)
    return loss

# --- 3. EXECUTION: PROCEDURAL MATH DATA ---
latent_dim = 768
rngs = nnx.Rngs(42)
model = RecursiveRefiner(latent_dim, rngs)

# Muon-style initialization hint: keep weights small for stability
model.fc1.kernel[...] *= 0.01 
model.fc2.kernel[...] *= 0.01

optimizer = nnx.Optimizer(model, optax.adam(5e-4), wrt=nnx.Param)
metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))

print("Starting RefineMath: Latent Algebraic Discovery...")
try:
    key = jax.random.key(0)
    for step in range(2001):
        key, subkey = jax.random.split(key)
        
        # --- PROCEDURAL GENERATION ---
        # Instead of noise, we generate a 'Problem' (x) and a 'Truth' (2x)
        x = jax.random.normal(subkey, (8, latent_dim))
        y = x * 2.0  # The algebraic rule to discover
        batch = {'input': x, 'target': y}

        loss = train_step(model, optimizer, metrics, batch)

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss:.6f}")
            # Reset metrics to prevent the memory bloat/OOM we saw earlier
            metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))
            
except Exception as e:
    print(f"Halted: {e}")