import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from flax import nnx
import jax.numpy as jnp

# 1. Load the model
model = RefineMathPhysics(LATENT_DIM, nnx.Rngs(0))
with open("physics_ckpt.pkl", "rb") as f:
    ckpt = pickle.load(f)
    nnx.update(model, ckpt['state'])

# 2. Setup Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
scatter = ax.scatter([], [], s=50)

# 3. Animation Function
def update(frame):
    # Get model prediction for the current frame
    # (Assuming you modify the model to return a sequence or run iterative steps)
    preds, _, _, _, _ = model(test_input, training=False)
    # Reshape predictions to (N, 2) for plotting
    pos = preds[0].reshape(-1, 2)
    scatter.set_offsets(pos)
    return scatter,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
plt.show()