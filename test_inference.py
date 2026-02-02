import jax
import jax.numpy as jnp
from flax import nnx
import pickle
import os
from train_local import AdaptiveRefiner, generate_complex_math

# Setup
latent_dim = 768
ckpt_path = "model_ckpt.pkl"

def run_test():
    # 1. Load Model
    rngs = nnx.Rngs(0)
    model = AdaptiveRefiner(latent_dim, rngs)
    
    if not os.path.exists(ckpt_path):
        print("No checkpoint found! Train for a bit first.")
        return

    with open(ckpt_path, 'rb') as f:
        cp = pickle.load(f)
        nnx.update(model, cp['model'])
        current_step = cp['step']
    
    print(f"--- Testing Model from Step {current_step} ---")

    # 2. Generate one specific problem
    key = jax.random.key(42)
    x, target = generate_complex_math(key, 1, latent_dim, step=current_step)
    
    # 3. Manually run the loop to see the "Thinking" steps
    z = jnp.ones_like(x) * 0.01
    print(f"Initial Error: {jnp.mean((z - target)**2):.6f}")
    
    for i in range(32):
        z_next, p_halt = model(z, target)
        
        # Calculate how much the "thought" changed
        velocity = jnp.linalg.norm(z_next - z)
        error = jnp.mean((z_next - target)**2)
        
        print(f"Step {i+1:02d} | Halt Prob: {p_halt[0,0]:.4f} | Error: {error:.6f} | Velocity: {velocity:.4f}")
        
        z = z_next
        if p_halt > 0.5:
            print(f">>> Model decided to HALT at step {i+1}")
            break

    print(f"Final Error: {jnp.mean((z - target)**2):.6f}")

if __name__ == "__main__":
    run_test()