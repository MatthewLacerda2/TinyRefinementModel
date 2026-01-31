import jax
import jax.numpy as jnp
from flax import nnx
from .muon import apply_muon_stabilization

class RefineMath(nnx.Module):
    def __init__(self, latent_dim, hidden_dim, rngs: nnx.Rngs):
        self.latent_dim = latent_dim
        # The 'Thought Generator' - projects input into latent space
        self.input_proj = nnx.Linear(1, latent_dim, rngs=rngs)
        # The 'Refiner' - the weights reused in every recursive step
        self.refine_layer = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, x, max_iters=32, threshold=1e-5):
        # Initial 'Noisy' Thought
        z = jax.nn.tanh(self.input_proj(x))
        
        def scan_body(carry, _):
            z_prev, done = carry
            
            # 1. Recursive Update
            z_next = self.refine_layer(z_prev)
            z_next = jax.nn.gelu(self.norm(z_next))
            
            # 2. Muon Stabilization (Internal Orthogonality)
            z_next = apply_muon_stabilization(z_next)
            
            # 3. Check for Convergence (Latent Velocity)
            velocity = jnp.linalg.norm(z_next - z_prev)
            is_done = velocity < threshold
            
            # If done, keep the old state (effectively halting)
            z_final = jnp.where(done | is_done, z_prev, z_next)
            return (z_final, done | is_done), velocity

        # jax.lax.scan fuses the loop into one GPU kernel
        (z_final, _), velocities = jax.lax.scan(
            scan_body, (z, False), jnp.arange(max_iters)
        )
        
        return z_final, velocities