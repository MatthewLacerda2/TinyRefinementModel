import jax
import jax.numpy as jnp
from flax import nnx
from train_local import UniversalReasoner, train_step, base_optimizer, optimizer_chain, PAD_TOKEN_ID, FORGET_LAMBDA

print("Testing NaN * 0:", jnp.nan * 0)
print("Testing NaN * False:", jnp.nan * False)
print("Testing jnp.sum of NaN * False:", jnp.sum(jnp.nan * False))

rngs = nnx.Rngs(42)
model = UniversalReasoner(512, rngs)
opt = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

# Case 1: All PAD
batch_tokens = jnp.full((1, 1024), PAD_TOKEN_ID, dtype=jnp.int32)

loss, metrics, expected_shared = train_step(model, opt, batch_tokens, step=1, f_lambda=FORGET_LAMBDA)
print("Loss with ALL PAD:", loss)
"""
# Case 2: One PAD
batch_tokens_2 = jnp.full((1, 1024), PAD_TOKEN_ID, dtype=jnp.int32)
batch_tokens_2 = batch_tokens_2.at[:, 0].set(100)
loss2, metrics2, _ = train_step(model, opt, batch_tokens_2, step=2, f_lambda=0.0)
print("Loss with ONE VALID:", loss2)
"""
