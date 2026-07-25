"""Chunked attention (#66) parity guards: the blockwise path must equal stock
dot_product_attention in VALUE and GRADIENTS — including causal masking, key
padding, and query lengths that don't divide the block size — and the chunked
CausalRefiner must preserve the no-future-leak invariant."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from attention import chunked_causal_attention
from plan_a_model import CausalRefiner

B, H, D = 1, 4, 8


def _stock(q, k, v, pad_cols):
    s = q.shape[1]
    pos = jnp.arange(s)
    causal = pos[:, None] >= pos[None, :]
    bias = jnp.where(causal, 0.0, -1e9)[None, None, :, :] + pad_cols[:, None, None, :]
    return jax.nn.dot_product_attention(q, k, v, bias=bias)


def _rand_qkv(seed, s):
    rng = np.random.default_rng(seed)
    mk = lambda: jnp.asarray(rng.normal(size=(B, s, H, D)), jnp.float32)
    q, k, v = mk(), mk(), mk()
    pad_cols = jnp.where(jnp.arange(s) < s - 3, 0.0, -1e9)[None, :].astype(jnp.float32)
    pad_cols = jnp.broadcast_to(pad_cols, (B, s))
    return q, k, v, pad_cols


def test_value_and_grad_parity_dividing_and_ragged():
    for s, block_q in ((32, 16), (24, 16)):  # dividing and non-dividing
        q, k, v, pad_cols = _rand_qkv(s, s)

        out_c = chunked_causal_attention(q, k, v, pad_cols, block_q)
        out_s = _stock(q, k, v, pad_cols)
        np.testing.assert_allclose(np.asarray(out_c), np.asarray(out_s), rtol=1e-5, atol=1e-6)

        # Gradient parity through a scalar loss that weights every element.
        w = jnp.asarray(np.random.default_rng(7).normal(size=out_s.shape), jnp.float32)
        g_c = jax.grad(lambda q, k, v: jnp.sum(chunked_causal_attention(q, k, v, pad_cols, block_q) * w),
                       argnums=(0, 1, 2))(q, k, v)
        g_s = jax.grad(lambda q, k, v: jnp.sum(_stock(q, k, v, pad_cols) * w),
                       argnums=(0, 1, 2))(q, k, v)
        for gc, gs, name in zip(g_c, g_s, "qkv"):
            np.testing.assert_allclose(np.asarray(gc), np.asarray(gs), rtol=1e-4, atol=1e-6,
                                       err_msg=f"grad w.r.t. {name} drifted (s={s})")


def test_chunked_refiner_matches_stock_refiner():
    """Full-model parity: same weights, flag on vs off — logits and a training
    gradient must agree."""
    kw = dict(dim=32, vocab_size=17, num_heads=4, num_encoder_layers=2,
              max_depth=3, max_seq_len=40)
    stock = CausalRefiner(**kw, chunked_attention=False, rngs=nnx.Rngs(0))
    chunked = CausalRefiner(**kw, chunked_attention=True, rngs=nnx.Rngs(0))

    tokens = jnp.asarray(np.random.default_rng(1).integers(0, 17, size=(2, 40)), jnp.int32)
    np.testing.assert_allclose(np.asarray(chunked(tokens, depth=3)),
                               np.asarray(stock(tokens, depth=3)), rtol=1e-4, atol=1e-5)

    def loss(m):
        return jnp.mean(m(tokens, depth=3) ** 2)

    gs = jax.tree_util.tree_leaves(nnx.grad(loss)(stock))
    gc = jax.tree_util.tree_leaves(nnx.grad(loss)(chunked))
    for a, b in zip(gc, gs):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-6)


def test_chunked_refiner_multiblock_with_padding():
    """Model-level parity where the chunked path actually crosses a block
    boundary (seq 160 > DEFAULT_BLOCK_Q=128) AND threads a real pad mask —
    the two conditions the small model tests above never exercise together."""
    kw = dict(dim=32, vocab_size=17, num_heads=4, num_encoder_layers=1,
              max_depth=2, max_seq_len=160)
    stock = CausalRefiner(**kw, chunked_attention=False, rngs=nnx.Rngs(3))
    chunked = CausalRefiner(**kw, chunked_attention=True, rngs=nnx.Rngs(3))

    rng = np.random.default_rng(11)
    tokens = jnp.asarray(rng.integers(0, 17, size=(2, 160)), jnp.int32)
    pad_mask = jnp.asarray(np.arange(160) < 149)[None, :]          # last 11 padded
    pad_mask = jnp.broadcast_to(pad_mask, (2, 160))

    out_s = stock(tokens, depth=2, pad_mask=pad_mask)
    out_c = chunked(tokens, depth=2, pad_mask=pad_mask)
    np.testing.assert_allclose(np.asarray(out_c), np.asarray(out_s), rtol=1e-4, atol=1e-5)

    def loss(m):
        return jnp.mean(m(tokens, depth=2, pad_mask=pad_mask) ** 2)

    for a, b in zip(jax.tree_util.tree_leaves(nnx.grad(loss)(chunked)),
                    jax.tree_util.tree_leaves(nnx.grad(loss)(stock))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-6)


@pytest.mark.gpu
def test_chunked_attention_f16_parity_gpu():
    """Pin the PRODUCTION dtype path: on the GPU lane (RUN_TESTS_ON_GPU=1,
    f16 compute) chunked and stock attention must agree within f16 headroom.
    CPU CI skips this — the CPU lane forces f32 (see conftest)."""
    s, block_q = 160, 128
    rng = np.random.default_rng(23)
    mk = lambda: jnp.asarray(rng.normal(size=(B, s, H, D)), jnp.float16)
    q, k, v = mk(), mk(), mk()
    pad_cols = jnp.broadcast_to(
        jnp.where(jnp.arange(s) < s - 5, 0.0, -1e9)[None, :].astype(jnp.float32), (B, s))

    out_c = chunked_causal_attention(q, k, v, pad_cols, block_q)
    out_s = _stock(q, k, v, pad_cols)
    assert np.isfinite(np.asarray(out_c)).all()
    np.testing.assert_allclose(np.asarray(out_c), np.asarray(out_s), rtol=2e-2, atol=2e-3)


def test_chunked_path_has_no_future_leak():
    """Perturbing position j must leave logits at positions < j unchanged at
    every depth — the same causality bar the stock refiner passes."""
    model = CausalRefiner(dim=32, vocab_size=17, num_heads=4, num_encoder_layers=2,
                          max_depth=4, max_seq_len=24, chunked_attention=True,
                          rngs=nnx.Rngs(2))
    rng = np.random.default_rng(5)
    tokens = jnp.asarray(rng.integers(0, 17, size=(1, 24)), jnp.int32)
    j = 15
    perturbed = tokens.at[0, j].set((int(tokens[0, j]) + 1) % 17)
    for depth in (1, 2, 4):
        base = model(tokens, depth=depth)
        pert = model(perturbed, depth=depth)
        np.testing.assert_array_equal(
            np.asarray(base[:, :j]), np.asarray(pert[:, :j]),
            err_msg=f"future-token leak through the chunked path at depth {depth}",
        )
