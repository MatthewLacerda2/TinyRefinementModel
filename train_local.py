import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time

# --- 1. THE RECURSIVE BRAIN ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.refine_layer = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)
    
    def __call__(self, z):
        delta = jax.nn.gelu(self.refine_layer(z))
        return self.norm(z + 0.01 * delta)

# --- 2. MUON TRANSFORM (The "2060" Optimizer) ---
def muon_transform(lr=0.02):
    """
    Wraps your Newton-Schulz logic into an Optax-compatible transformation.
    """
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        def newton_schulz(G):
            for _ in range(5):
                G = 1.5 * G - 0.5 * G @ G.T @ G
            return G
        
        new_updates = jax.tree_map(
            lambda u: newton_schulz(u) * lr if u.ndim == 2 else u * lr, 
            updates
        )
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

# --- 3. TRAINING LOOP WITH NAN GUARD ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        z_initial = jnp.zeros((8, 512)) 
        def think_loop(z, _):
            return model(z), None
        z_final, _ = jax.lax.scan(think_loop, z_initial, None, length=16)
        return jnp.mean((z_final - batch['target'])**2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)

    if not jnp.isfinite(loss):
        print(f"\n!!! NAN DETECTED !!! | Grad Norm: {optax.global_norm(grads)}")
        raise FloatingPointError("Stopping training.")

    optimizer.update(model, grads) 
    metrics.update(loss=loss)
    return loss

# --- 4. EXECUTION ---
rngs = nnx.Rngs(42)
model = RecursiveRefiner(512, rngs)

model.refine_layer.kernel[...] *= 0.001 
model.norm.scale[...] = 0.0 

tx = optax.chain(
    optax.clip_by_global_norm(1.0), 
    muon_transform(lr=0.0001)
)

optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))

print("Starting Local Proof-of-Concept with Muon...")
try:
    for step in range(101):
        start = time.time()
        batch = {'target': jax.random.normal(jax.random.key(step), (8, 512))}
        loss = train_step(model, optimizer, metrics, batch)
        
        if step % 5 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Time: {time.time()-start:.3f}s")
except FloatingPointError:
    pass