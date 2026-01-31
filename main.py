import os
from flax import nnx
import optax
import jax

# Import our custom modules
from model.refiner import RefineMath
from training.train import train_step
from data.procedural_gen import MathGym

def main():
    # 1. Setup Environment & Hardware
    # Force JAX to be polite to your 6GB VRAM
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"Devices found: {jax.devices()}")

    # 2. Initialize the Beast (10M Parameters)
    rngs = nnx.Rngs(42)
    model = RefineMath(latent_dim=512, hidden_dim=1024, rngs=rngs)
    
    # 3. Setup Optimizer (Muon-compatible learning rate)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4))
    
    # 4. Data Generator
    gym = MathGym()

    # 5. Training Loop
    print("Starting training...")
    for step in range(100):
        batch = gym.generate_batch(batch_size=16)
        # We wrap the train_step in JIT here for maximum speed
        loss = train_step(model, optimizer, batch)
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss:.4f}")

    print("Model trained and ready for GCP scaling.")

if __name__ == "__main__":
    main()