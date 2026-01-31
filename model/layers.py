import jax
import jax.numpy as jnp
from flax import nnx

class RefinerBlock(nnx.Module):
    """The core recursive unit: reused for N 'thinking' steps."""
    def __init__(self, dim, num_heads, rngs: nnx.Rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False, # We process the latent, not a sequence
            rngs=rngs
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(dim, dim * 4, rngs=rngs),
            jax.nn.gelu,
            nnx.Linear(dim * 4, dim, rngs=rngs)
        )
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x):
        # Multi-head self-attention on the latent thought
        x = x + self.attn(self.norm1(x))
        # Feed-forward refinement
        x = x + self.ffn(self.norm2(x))
        return x