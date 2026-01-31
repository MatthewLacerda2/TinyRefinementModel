import jax
import jax.numpy as jnp
import optax

def compute_grpo_loss(logits, actions, rewards, old_log_probs, epsilon=0.2):
    """
    GRPO Loss: Policy-only RL using Group-Relative Advantage.
    No Critic model needed—saves 50% VRAM.
    """
    # 1. Calculate Group-Relative Advantage
    # rewards shape: (batch_size, group_size)
    mean_reward = jnp.mean(rewards, axis=1, keepdims=True)
    std_reward = jnp.std(rewards, axis=1, keepdims=True) + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # 2. Policy Ratio: (new_prob / old_prob)
    log_probs = jax.nn.log_softmax(logits)
    # Gather log_probs for the specific actions taken
    current_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    
    ratio = jnp.exp(current_log_probs - old_log_probs)

    # 3. Clipped Objective (Standard PPO logic)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    loss = -jnp.mean(jnp.minimum(surr1, surr2))
    return loss

def python_code_reward(generated_code, target_output):
    """
    The 'Execution-Feedback' Reward Function.
    This would typically run in a JAX-external sandbox.
    """
    # Placeholder: In practice, use a subprocess or Pyodide
    # Returns 1.0 if code passes tests, 0.0 otherwise.
    pass