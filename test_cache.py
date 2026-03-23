import jax
import jax.numpy as jnp
from flax import nnx

class M(nnx.Module):
    def __init__(self, rngs):
        self.p = nnx.Param(jnp.ones(10))
        self.cache = nnx.Cache(None)
    def __call__(self, x):
        return x

@nnx.vmap(in_axes=0, out_axes=0)
def create(x): return M(nnx.Rngs(0))

m = create(jnp.arange(4))

@nnx.jit
def f(m, x):
    m.cache.value = None
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def scan_fn(c, block):
        return block(c), None
    return scan_fn(x, m)[0]

print(f(m, jnp.ones(10)))
