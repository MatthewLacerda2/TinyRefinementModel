"""Whole-model invariants: padding correctness and causality."""

import jax.numpy as jnp
import numpy as np

from trm.config import COMPUTE_DTYPE
from trm.settings import CONFIG


def _logits(model, tokens_np):
    out = model(jnp.asarray(tokens_np), training=False)
    return np.asarray(out.logits, dtype=np.float32)


def test_pad_run_length_does_not_change_the_real_tokens_after_it(tiny_model, make_tiny_model,
                                                                 token_batch):
    """PAD must be invisible to the real tokens that could attend to it.

    Until #322 the pads went at the TAIL, after the real tokens. On a causal-only
    model that is vacuous: nothing before a pad can attend to it, so deleting the
    pad mask entirely moved plain's logits by 1.9e-6, the same as the honest model.
    Here the pads come FIRST, where every real token after them could attend to
    them. With the mask, the run length is a pure position shift, and RoPE only
    sees relative distances between the (unmasked) real tokens, so the real-token
    logits must not depend on it."""
    real = token_batch[:, :48]
    short = np.concatenate([np.full((1, 8), CONFIG.PAD_TOKEN_ID, dtype=np.int32), real], axis=1)
    long = np.concatenate([np.full((1, 16), CONFIG.PAD_TOKEN_ID, dtype=np.int32), real], axis=1)

    logits_short = _logits(tiny_model, short)[:, 8:]
    logits_long = _logits(tiny_model, long)[:, 16:]

    # The floor is measured, not assumed (#121). In f32 a pure position shift is exact
    # to ~2e-6, so 1e-3 is loose. In f16 the rotations at shifted positions round
    # differently, and the shift moves the logits by about as much as f16 itself
    # does: measured on the card, 0.0079 against an f16-vs-f32 difference of 0.0068
    # on the same input, while a real leak (the same runs, unmasked) moves them by
    # up to 1.08. So on the f16 lane the tolerance is twice that f16-vs-f32 floor,
    # measured here on this model and this input.
    atol = 1e-3
    if COMPUTE_DTYPE == jnp.float16:
        f32_twin = make_tiny_model(seed=0, dtype=jnp.float32)
        atol = 2.0 * float(np.abs(logits_short - _logits(f32_twin, short)[:, 8:]).max())

    np.testing.assert_allclose(
        logits_short, logits_long, rtol=0.0, atol=atol,
        err_msg="Logits of real tokens depend on how many PAD tokens precede them — "
                "a padding mask is leaking somewhere.",
    )


def test_future_token_cannot_influence_past_predictions(tiny_model, token_batch):
    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    base = _logits(tiny_model, token_batch)[:, :40]
    after = _logits(tiny_model, perturbed)[:, :40]

    np.testing.assert_allclose(base, after, rtol=1e-3, atol=1e-3)

