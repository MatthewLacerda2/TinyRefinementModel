"""Mask-bias overflow guard (#84): fully-masked rows must stay finite in f16.

The additive mask constant is -1e9; cast to f16 (max ~65504) it overflows to
-inf, and any fully-masked row — e.g. a pad query position whose only visible
keys are pad — turns its softmax row into NaN. The fix keeps the bias in f32
into dot_product_attention (which adds it to f32 logits, so the f16
tensor-core QK path is untouched).

These tests build genuinely fully-masked rows in explicit f16 and require
finite output, from the attention alone and through the whole model. The tiny shapes lower on the CPU
backend too, so they run on both lanes; RUN_TESTS_ON_GPU=1 exercises the
production f16 compile on the real device.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.model.plain import CausalAttention, PlainTransformer
from trm.settings import CONFIG

PAD = 0


def test_fully_masked_rows_stay_finite_in_f16():
    """Batch element 1 has every key masked, so all its query rows are fully
    masked — the exact configuration that softmaxes to NaN with an -inf bias."""
    attn = CausalAttention(dim=32, num_heads=4, max_pos=16, rngs=nnx.Rngs(0),
                           dtype=jnp.float16)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 16, 32)), jnp.float16)
    pad_bias = jnp.zeros((2, 1, 1, 16), jnp.float32).at[1].set(-1e9)

    out = np.asarray(attn(x, pad_bias))
    assert np.isfinite(out).all(), (
        "non-finite attention output on fully-masked rows "
    )


def test_leading_pad_stays_finite_in_f16():
    """End-to-end: leading pad makes position 0's row see only pad keys (causal
    mask + key padding), a fully-masked row arising from ordinary padding."""
    model = PlainTransformer(32, nnx.Rngs(1), CONFIG, vocab_size=17, num_heads=4,
                             num_layers=1, max_seq_len=16, pad_token_id=PAD,
                             dtype=jnp.float16)
    tokens = jnp.asarray(np.random.default_rng(1).integers(1, 17, size=(1, 16)), jnp.int32)
    tokens = tokens.at[0, :2].set(PAD)  # first two positions are pad

    logits = np.asarray(model(tokens, training=False).logits)
    assert np.isfinite(logits).all(), "non-finite logits from a leading-pad batch in f16"
