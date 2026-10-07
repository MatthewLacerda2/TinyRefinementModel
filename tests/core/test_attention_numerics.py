"""Reference-numerics test for the model's attention (CausalAttention).

Recomputes attention with an independent, naive implementation of the score
math (scale once, mask, softmax, weighted sum) using the module's own
projections, norms, and RoPE tables — so any disagreement is in the attention
math itself. This is the test that would have caught the double-scaling bug
(q pre-scaled on top of dot_product_attention's internal scaling) on day one.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.model.plain import CausalAttention
from trm.model.rope import apply_rope

HEADS, DIM = 4, 64
HEAD_DIM = DIM // HEADS
POSITIONS = 32


def _reference_attention(attn, x, pad_bias=None):
    b, s, _ = x.shape

    q = attn.q_norm(attn.q(x).reshape(b, s, HEADS, HEAD_DIM))
    k = attn.k_norm(attn.k(x).reshape(b, s, HEADS, HEAD_DIM))
    v = attn.v(x).reshape(b, s, HEADS, HEAD_DIM)
    q = apply_rope(q, attn.cos[:s, None, :], attn.sin[:s, None, :])
    k = apply_rope(k, attn.cos[:s, None, :], attn.sin[:s, None, :])

    # The attention math, written once, naively, in f32: exactly one scaling.
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = scores * (HEAD_DIM ** -0.5)
    if pad_bias is not None:
        scores = scores + pad_bias
    pos = jnp.arange(s)
    causal = pos[:, None] >= pos[None, :]
    scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32))
    return attn.o(out.astype(x.dtype).reshape(b, s, DIM))


def _attn():
    return CausalAttention(DIM, HEADS, POSITIONS, rngs=nnx.Rngs(0))


def test_causal_self_attention_matches_naive_reference():
    attn = _attn()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, DIM), dtype=jnp.float32)

    np.testing.assert_allclose(
        np.asarray(attn(x), dtype=np.float32),
        np.asarray(_reference_attention(attn, x), dtype=np.float32),
        rtol=2e-2, atol=2e-2,
        err_msg="CausalAttention disagrees with the naive reference — check for "
                "double/missing score scaling or mask handling.",
    )


def test_pad_bias_matches_naive_reference():
    """The key-padding bias the model builds (`[b, 1, 1, s]`, -1e9 on a pad key) must
    remove exactly those keys, on top of the causal mask."""
    attn = _attn()
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 10, DIM), dtype=jnp.float32)
    pad_bias = jnp.zeros((2, 1, 1, 10), dtype=jnp.float32).at[1, :, :, 6:8].set(-1e9)

    np.testing.assert_allclose(
        np.asarray(attn(x, pad_bias), dtype=np.float32),
        np.asarray(_reference_attention(attn, x, pad_bias), dtype=np.float32),
        rtol=2e-2, atol=2e-2,
    )
