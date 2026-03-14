import jax
import jax.numpy as jnp
from flax import nnx
from train_local import UniversalReasoner

rngs = nnx.Rngs(42)
model = UniversalReasoner(512, rngs)

# Normal initialization creates params with std ~1/sqrt(D). 
# We'll run 32 steps and see if it explodes.
batch_tokens = jnp.full((1, 128), 100, dtype=jnp.int32)
logits, ponder_cost, forget_cost, halt_diag, expected_shared = model(batch_tokens, training=True, max_steps=32)

print("Max expected_shared:", jnp.max(jnp.abs(expected_shared)))
print("Max logits:", jnp.max(jnp.abs(logits)))
print("Halt Diag NaNs:", any(jnp.isnan(v).any() for v in halt_diag.values() if isinstance(v, jnp.ndarray)))
