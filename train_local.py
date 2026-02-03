import os, pickle, json, time
import jax, jax.numpy as jnp
import optax
from flax import nnx

# Optimized Environment for JAX
os.environ.update({
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
    'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'
})

class AdaptiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # Predictor: Initial guess of depth
        self.predictor = nnx.Sequential(
            nnx.Linear(latent_dim * 2, latent_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(latent_dim // 2, 1, rngs=rngs)
        )
        # Refiner: Core logic
        self.fc1 = nnx.Linear(latent_dim * 2 + 1, latent_dim * 2, rngs=rngs)
        self.refine_fc = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def predict_complexity(self, z, target):
        return jax.nn.softplus(self.predictor(jnp.concatenate([z, target], axis=-1)))

    def __call__(self, z, target, step_count):
        # Inject step feature
        step_feat = jnp.full((z.shape[0], 1), step_count, dtype=z.dtype)
        combined = jnp.concatenate([z, target, step_feat], axis=-1)
        
        h = jax.nn.gelu(self.fc1(combined))
        delta = self.refine_fc(h)
        
        next_z = self.norm(z + 0.1 * delta)
        p_halt = jax.nn.sigmoid(self.halt_fc(next_z))
        return next_z, p_halt

def run_model_loss(model, batch, key):
    z_init = (jax.random.normal(key, batch['input'].shape) * 0.02 + 0.01).astype(jnp.float16)
    target = batch['target'].astype(jnp.float16)
    predicted_steps = model.predict_complexity(z_init, target)

    def body_fn(carry):
        z, active_mask, step_counts, step = carry
        next_z, p_halt = model(z, target, step)
        
        # Update logic
        new_halt = (p_halt.squeeze() > 0.5)
        still_active = active_mask & (~new_halt)
        
        z_updated = jnp.where(active_mask[:, None], next_z, z)
        return (z_updated, still_active, step_counts + active_mask.astype(jnp.float16), step + 1)

    initial_carry = (z_init, jnp.ones(z_init.shape[0], bool), jnp.zeros(z_init.shape[0]), 0)
    final_z, _, actual_steps, _ = jax.lax.while_loop(
        lambda c: jnp.any(c[1]) & (c[3] < 128), body_fn, initial_carry
    )
    
    # Losses
    recon_loss = jnp.mean((final_z - target) ** 2)
    step_penalty = jnp.mean(actual_steps) * 0.001
    pred_loss = jnp.mean((predicted_steps - actual_steps)**2) * 0.001
    return recon_loss + step_penalty + pred_loss

@nnx.jit
def train_step(model, optimizer, subkeys, current_ops, step):
    def loss_fn(model):
        def scan_body(_, key):
            # Dynamic problem generation
            x = jax.random.normal(key, (128, 768), dtype=jnp.float16)
            # Simplified target logic for brevity; keep your generate_complex_math here
            batch = {'input': x, 'target': x * 1.5} 
            return None, run_model_loss(model, batch, key)

        _, losses = jax.lax.scan(scan_body, None, subkeys)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads / 32768.0) # Unscale
    return loss / 32768.0

# --- Optimizer Setup ---
def get_optimizer(model):
    def label_fn(path, _):
        return "muon" if "kernel" in [str(p) for p in path] else "adam"
    
    muon = optax.chain(optax.clip_by_global_norm(1.0), optax.contrib.muon(0.02))
    adam = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    
    return nnx.Optimizer(model, optax.multi_transform(
        {'muon': muon, 'adam': adam}, 
        lambda p: jax.tree_util.tree_map_with_path(label_fn, p)
    ))

# --- Main Initialization ---
model = AdaptiveRefiner(768, nnx.Rngs(42))
optimizer = get_optimizer(model)
# ... Load logic remains similar but cleaner ...