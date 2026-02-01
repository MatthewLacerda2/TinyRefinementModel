import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time

# --- 1. THE RECURSIVE BRAIN ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # Expansion: Map 768 -> 1536 (higher capacity)
        self.fc1 = nnx.Linear(latent_dim, latent_dim * 2, rngs=rngs)
        # Compression: Map 1536 -> 768
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z):
        # The "Internal Brain" step
        h = jax.nn.gelu(self.fc1(z))
        delta = self.fc2(h) 
        
        # Residual connection + Norm
        return self.norm(z + 0.1 * delta)

def run_until_converged(model, z0, max_steps=32, eps=1e-3):
    z = z0
    for _ in range(max_steps):
        z_next = model(z)
        if jnp.linalg.norm(z_next - z) < eps:
            break
        z = z_next
    return z

# --- 2. THE TRAINING LOOP WITH NAN GUARD ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        # initial state z0 (Option B robustness)
        z0 = jnp.ones((8, 768)) * 0.01

        def step_fn(z, _):
            z_next = model(z)
            return z_next, z_next

        # unroll fixed number of steps for training
        zs, _ = jax.lax.scan(step_fn, z0, None, length=16)

        # ----- Option B: per-step loss -----
        T = zs.shape[0]
        weights = jnp.linspace(0.1, 1.0, T)[:, None, None]
        loss_B = jnp.mean(weights * (zs - batch["target"]) ** 2)

        # ----- Option A: final-step loss -----
        loss_A = jnp.mean((zs[-1] - batch["target"]) ** 2)

        # combine (A is light)
        return loss_B + 0.1 * loss_A

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)

    optimizer.update(model, grads)

    metrics.loss.update(value=loss)
    return loss

# --- 3. EXECUTION ---
rngs = nnx.Rngs(42)
model = RecursiveRefiner(768, rngs)

model.fc1.kernel[...] *= 0.001
model.fc2.kernel[...] *= 0.001
model.norm.scale[...] = 0.1

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-4)
)

optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
metrics = nnx.metrics.MultiMetric(loss=nnx.metrics.Average('loss'))

print("Starting Local Proof-of-Concept...")
try:
    # Move this OUTSIDE the for-loop to make the target constant
    key = jax.random.key(0)
    static_target = jax.random.normal(key, (8, 768))
    for step in range(5000):
        start = time.time()
        
        key, subkey = jax.random.split(key)
        batch = {'target': jax.random.normal(subkey, (8, 768))}

        loss = train_step(model, optimizer, metrics, batch)

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Time: {time.time()-start:.3f}s")
except FloatingPointError as e:
    print(f"Training halted: {e}")