import jax
import jax.numpy as jnp
from flax import nnx

from train_local import MAX_STEPS_LIMIT, UniversalReasoner, HUNCH_REFRESH_EVERY


def run_model_inference(
    model: UniversalReasoner,
    tokens: jnp.ndarray,
    max_steps: int = MAX_STEPS_LIMIT,
    should_refresh: bool = True,
) -> jnp.ndarray:
    logits, ponder_cost, forget_loss, halt_diag, expected_shared = model(
        tokens, max_steps=max_steps, training=False, should_refresh=should_refresh
    )
    return logits


def generate_text(
    model: UniversalReasoner,
    prompt_tokens: jnp.ndarray,
    max_new_tokens: int,
    rng: jax.Array,
    max_steps: int = MAX_STEPS_LIMIT,
    temperature: float = 1.0,
) -> jnp.ndarray:
    current_tokens = prompt_tokens
    
    for i in range(max_new_tokens):
        should_refresh = (i % HUNCH_REFRESH_EVERY == 0)
        logits = run_model_inference(model, current_tokens, max_steps=max_steps, should_refresh=should_refresh)
        next_token_logits = logits[:, -1, :]
        
        if temperature > 0.0:
            rng, key = jax.random.split(rng)
            next_token_logits = next_token_logits / temperature
            next_token = jax.random.categorical(key, next_token_logits)[:, None]
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)[:, None]
            
        current_tokens = jnp.concatenate([current_tokens, next_token], axis=1)
        
    return current_tokens
