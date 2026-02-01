import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time

# --- 1. THE RECURSIVE BRAIN ---
class RecursiveRefiner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        # Using a simple GRU-like update for the 'thought' state
        self.refine_layer = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, z):
        # One 'step' of thinking
        delta = jax.nn.gelu(self.refine_layer(z))
        return self.norm(z + delta)

# --- 2. MUON OPTIMIZER (Simplified for 2060) ---
def muon_update(params, updates, lr=0.02):
    # Newton-Schulz iteration to orthogonalize updates
    def newton_schulz(G):
        for _ in range(5):
            G = 1.5 * G - 0.5 * G @ G.T @ G
        return G
    
    return jax.tree_map(lambda p, u: u * lr, params, updates)

# --- 3. THE TRAINING LOOP (Updated for Flax 0.11.0+) ---
def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        # 8 parallel 'thoughts' for GRPO
        z_initial = jnp.zeros((8, 512)) 

        # Recursive loop (Thinking for 16 steps)
        def think_loop(z, _):
            return model(z), None
        z_final, _ = jax.lax.scan(think_loop, z_initial, None, length=16)
        return jnp.mean((z_final - batch['target'])**2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    
    # NEW: Must pass 'model' here too
    optimizer.update(model, grads) 
    
    metrics.update(loss=loss)
    return loss

# --- 4. EXECUTION (Updated for Flax 0.11.0+) ---
rngs = nnx.Rngs(42)
model = RecursiveRefiner(512, rngs)

# NEW: Added 'wrt=nnx.Param' to tell the optimizer to target the weights
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

metrics = nnx.metrics.MultiMetric(
    loss=nnx.metrics.Average()
)

print("Starting Local Proof-of-Concept...")
for step in range(100):
    start = time.time()
    # Dummy data: trying to map zeros to a target 'thought' vector
    batch = {'target': jax.random.normal(jax.random.key(step), (8, 512))}
    
    loss = train_step(model, optimizer, metrics, batch)
    
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss:.4f} | Time: {time.time()-start:.3f}s")