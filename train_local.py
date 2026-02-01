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
        # The "Halting" head: outputs a single value between 0 and 1
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs, dtype=jnp.float16)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=jnp.float16)

    def __call__(self, z, target):
        combined = jnp.concatenate([z, target], axis=-1)
        # Use a more complex hidden state for the decision
        h = jax.nn.gelu(self.fc1(combined))
        
        delta = self.refine_fc(h)
        next_z = self.norm(z + 0.1 * delta)
        
        # Calculate probability of stopping now
        p = jax.nn.sigmoid(self.halt_fc(next_z))
        
        return next_z, p

# --- 2. TRUE ACCUMULATION STEP ---
@nnx.jit
def compute_grads(model, batch):
    def loss_fn(model):
        z = jnp.ones_like(batch['input'], dtype=jnp.float16) * 0.01
        def step_fn(current_z, _):
            next_z, p = model(current_z, batch["target"])
            return next_z, (next_z, p)
        (last_z, _), (zs, ps) = jax.lax.scan(step_fn, z, None, length=16)
        
        # Base loss: final reconstruction
        recon_loss = jnp.mean((last_z - batch["target"]) ** 2)
        
        # Adaptive loss: we want to halt eventually (simple penalty for now)
        # Or just return recon_loss to match previous behavior
        return recon_loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads

# --- 3. EXECUTION ---
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
        # Initialize zero gradients for accumulation
        accum_grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
        
        for _ in range(accum_steps):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (micro_batch, latent_dim), dtype=jnp.float16)
            batch = {'input': x, 'target': x * 2.0}
            
            loss, grads = compute_grads(model, batch)
            # Accumulate
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