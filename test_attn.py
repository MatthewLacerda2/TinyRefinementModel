import jax
import jax.numpy as jnp

q = jnp.ones((1, 1, 1, 1))
k = jnp.ones((1, 1, 1, 1))
v = jnp.ones((1, 1, 1, 1))
m = jnp.zeros((1, 1, 1, 1), dtype=jnp.bool_)

out = jax.nn.dot_product_attention(q, k, v, mask=m)
print("Out with all false mask:", out)
