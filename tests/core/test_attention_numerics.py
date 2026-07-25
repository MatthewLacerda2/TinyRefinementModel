"""Reference-numerics test for RotaryAttention.

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

from config import COMPUTE_DTYPE
from layers import RotaryAttention, apply_rope

HEADS, GROUPS, DIM = 4, 2, 64
HEAD_DIM = DIM // HEADS


def _reference_attention(attn, x, context=None, bias=None, q_pos=None, kv_pos=None, is_causal=True):
    b, s, _ = x.shape
    kv_input = context if context is not None else x
    s_kv = kv_input.shape[1]

    q = attn.q_proj(x).reshape(b, s, HEADS, HEAD_DIM)
    k = attn.k_proj(kv_input).reshape(b, s_kv, GROUPS, HEAD_DIM)
    v = attn.v_proj(kv_input).reshape(b, s_kv, GROUPS, HEAD_DIM)
    q, k = attn.q_norm(q), attn.k_norm(k)

    if q_pos is None:
        q_pos = jnp.arange(s)
    if kv_pos is None:
        kv_pos = jnp.arange(s_kv)
    q = apply_rope(q, attn.cos_cached[q_pos, None, :], attn.sin_cached[q_pos, None, :])
    k = apply_rope(k, attn.cos_cached[kv_pos, None, :], attn.sin_cached[kv_pos, None, :])

    k = jnp.repeat(k, HEADS // GROUPS, axis=2)
    v = jnp.repeat(v, HEADS // GROUPS, axis=2)

    # The attention math, written once, naively, in f32: exactly one scaling.
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = scores * (HEAD_DIM ** -0.5)
    if bias is not None:
        scores = scores + bias
    if is_causal:
        causal = q_pos[:, None] >= kv_pos[None, :]
        scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32))
    return attn.o_proj(out.astype(COMPUTE_DTYPE).reshape(b, s, DIM))


def test_causal_self_attention_matches_naive_reference():
    attn = RotaryAttention(HEADS, DIM, num_groups=GROUPS, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, DIM), dtype=jnp.float32)

    module_out = attn(x)
    ref_out = _reference_attention(attn, x)

    np.testing.assert_allclose(
        np.asarray(module_out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        rtol=2e-2, atol=2e-2,
        err_msg="RotaryAttention disagrees with the naive reference — check for "
                "double/missing score scaling or mask handling.",
    )


def test_cross_attention_with_bias_matches_naive_reference():
    """Mirrors the reasoning-loop usage: non-causal cross-attention with a float
    bias mask and explicit slot-style positions."""
    attn = RotaryAttention(HEADS, DIM, num_groups=GROUPS, rngs=nnx.Rngs(0))
    slots = jax.random.normal(jax.random.PRNGKey(2), (2, 4, DIM), dtype=jnp.float32)
    ctx = jax.random.normal(jax.random.PRNGKey(3), (2, 12, DIM), dtype=jnp.float32)

    q_pos = jnp.arange(100, 104)
    kv_pos = jnp.arange(12)
    bias = jnp.zeros((2, 1, 1, 12), dtype=jnp.float32).at[:, :, :, -2:].set(-1e9)

    module_out = attn(slots, context=ctx, mask=bias, q_pos=q_pos, kv_pos=kv_pos, is_causal=False)
    ref_out = _reference_attention(attn, slots, context=ctx, bias=bias, q_pos=q_pos, kv_pos=kv_pos, is_causal=False)

    np.testing.assert_allclose(
        np.asarray(module_out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        rtol=2e-2, atol=2e-2,
    )
