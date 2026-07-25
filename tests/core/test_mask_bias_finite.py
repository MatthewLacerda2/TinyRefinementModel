"""Mask-bias overflow guard (#84): fully-masked rows must stay finite in f16.

The additive mask constant is -1e9; cast to f16 (max ~65504) it overflows to
-inf, and any fully-masked row — e.g. a pad query position whose only visible
keys are pad — turns its softmax row into NaN. The fix keeps the bias in f32
into dot_product_attention (which adds it to f32 logits, so the f16
tensor-core QK path is untouched), matching the contract the chunked path
already made.

These tests build genuinely fully-masked rows in explicit f16 and require
finite output from BOTH attention paths. The tiny shapes lower on the CPU
backend too, so they run on both lanes; RUN_TESTS_ON_GPU=1 exercises the
production f16 compile on the real device.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import layers
from layers import RotaryAttention
from plan_a_model import CausalAttention, CausalRefiner


@pytest.mark.parametrize("chunked", [False, True], ids=["stock", "chunked"])
def test_fully_masked_rows_stay_finite_in_f16(chunked):
    """Batch element 1 has every key masked, so all its query rows are fully
    masked — the exact configuration that softmaxes to NaN with an -inf bias."""
    attn = CausalAttention(dim=32, num_heads=4, max_pos=16, rngs=nnx.Rngs(0),
                           dtype=jnp.float16, chunked=chunked)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 16, 32)), jnp.float16)
    pad_bias = jnp.zeros((2, 1, 1, 16), jnp.float32).at[1].set(-1e9)

    out = np.asarray(attn(x, pad_bias))
    assert np.isfinite(out).all(), (
        f"non-finite attention output on fully-masked rows ({'chunked' if chunked else 'stock'} path)"
    )


@pytest.mark.parametrize("chunked", [False, True], ids=["stock", "chunked"])
def test_refiner_leading_pad_stays_finite_in_f16(chunked):
    """End-to-end: leading pad makes position 0's row see only pad keys (causal
    mask + key padding), a fully-masked row arising from a plain pad_mask."""
    model = CausalRefiner(dim=32, vocab_size=17, num_heads=4, num_encoder_layers=1,
                          max_depth=2, max_seq_len=16, chunked_attention=chunked,
                          rngs=nnx.Rngs(1), dtype=jnp.float16)
    tokens = jnp.asarray(np.random.default_rng(1).integers(0, 17, size=(1, 16)), jnp.int32)
    pad_mask = jnp.asarray(np.arange(16) >= 2)[None, :]  # first two positions are pad

    logits = np.asarray(model(tokens, depth=2, pad_mask=pad_mask))
    assert np.isfinite(logits).all(), "non-finite logits from a leading-pad batch in f16"


def test_rotary_attention_additive_mask_stays_finite_in_f16(monkeypatch):
    """#99: the reasoner path's RotaryAttention has the same hazard — its
    additive float masks carry -1e9 entries (slot masks, model.py constants),
    and the pre-fix cast to COMPUTE_DTYPE overflowed them to -inf in f16.
    COMPUTE_DTYPE is read at call time, so patching it forces the production
    f16 path on the CPU lane too; batch element 1 masks every key, giving
    genuinely fully-masked rows."""
    monkeypatch.setattr(layers, "COMPUTE_DTYPE", jnp.float16)
    attn = RotaryAttention(4, 32, num_groups=4, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(2).normal(size=(2, 8, 32)), jnp.float32)
    additive_mask = jnp.zeros((2, 1, 8, 8), jnp.float32).at[1].add(-1e9)

    out = np.asarray(attn(x, mask=additive_mask))
    assert np.isfinite(out).all(), \
        "non-finite RotaryAttention output on fully-masked rows (additive-bias branch)"


def test_rotary_attention_boolean_mask_stays_finite_in_f16(monkeypatch):
    """Contrast branch: a fully-masked BOOLEAN row goes through
    dot_product_attention's mask argument, whose masked path uses its own
    finite large-negative constant — pinned here so the two branches' contracts
    stay documented together."""
    monkeypatch.setattr(layers, "COMPUTE_DTYPE", jnp.float16)
    attn = RotaryAttention(4, 32, num_groups=4, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 8, 32)), jnp.float32)
    bool_mask = jnp.ones((2, 1, 8, 8), bool).at[1].set(False)

    out = np.asarray(attn(x, mask=bool_mask))
    assert np.isfinite(out).all(), \
        "non-finite RotaryAttention output on fully-masked rows (boolean-mask branch)"
