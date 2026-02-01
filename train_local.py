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
        return self.norm(z + 0.1 * delta)

# --- 2. THE TRAINING LOOP WITH NAN GUARD ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        z_initial = jnp.ones((8, 512)) * 0.01
        def think_loop(z, _):
            z_next = model(z)
            return z_next, z_next
        zs, _ = jax.lax.scan(think_loop, z_initial, None, length=16)
        return jnp.mean((zs - batch['target'])**2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)

    if not jnp.isfinite(loss):
        print("\n!!! NAN DETECTED !!!")

        global_grad_norm = optax.global_norm(grads)
        print(f"Loss: {loss}")
        print(f"Global Gradient Norm: {global_grad_norm}")

        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        max_grad = max([jnp.max(jnp.abs(g)) for g in flat_grads])
        print(f"Max Absolute Gradient Value: {max_grad}")

        raise FloatingPointError("Stopping training due to NaN")

    optimizer.update(model, grads)

    kernel_norm = jnp.linalg.norm(model.refine_layer.kernel)
    print(f"Kernel norm: {kernel_norm:.6f}")
    print("Δkernel norm:", jnp.linalg.norm(grads.refine_layer.kernel))

    metrics.update(loss=loss)
    return loss

# --- 3. EXECUTION ---
rngs = nnx.Rngs(42)
model = RecursiveRefiner(512, rngs)

model.refine_layer.kernel[...] *= 0.001
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
    static_target = jax.random.normal(key, (8, 512))
    for step in range(100):
        start = time.time()
        batch = {'target': static_target} # Now it's a fixed goal

        loss = train_step(model, optimizer, metrics, batch)

        if step % 5 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Time: {time.time()-start:.3f}s")
except FloatingPointError as e:
    print(f"Training halted: {e}")