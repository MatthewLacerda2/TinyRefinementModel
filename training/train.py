import jax
import jax.numpy as jnp
from flax import nnx
import optax
from model.refiner import RefineMath
from training.grpo import compute_grpo_loss

def train_step(model: RefineMath, optimizer: nnx.Optimizer, batch):
    """
    A single high-speed update. 
    Using nnx.value_and_grad to track state and math simultaneously.
    """
    def loss_fn(model):
        # 1. Forward Pass (The 'Thinking' Loop)
        # We vmap this across the group_size for GRPO
        latent_out, velocities = model(batch['input'])
        
        # 2. Mock Reward (Logic Check)
        # In a real run, this is where your reward function grades the 'thought'
        rewards = jnp.where(velocities[-1] < 1e-5, 1.0, 0.0)
        
        # 3. Compute GRPO Loss
        # We compare trajectories within the group to find the advantage
        loss = compute_grpo_loss(latent_out, batch['actions'], rewards, batch['old_probs'])
        return loss

    # JAX calculates gradients through the entire recursive scan
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    
    return loss

@jax.jit
def main_training_loop(model, optimizer, data_iterator, steps=1000):
    """The 'Hot Loop' that stays on the TPU/GPU."""
    for _ in range(steps):
        batch = next(data_iterator)
        loss = train_step(model, optimizer, batch)
    return model, loss