"""Whole-model invariants: padding correctness and causality.

The padding test is a true invariant and must always pass. The causality test
documents a KNOWN VIOLATION: the reasoning slots read the entire sequence
bidirectionally (model.py: is_causal=False over z_seq) and the decoder exposes
those slots to every token position, so predicting token t can use information
about tokens after t. It is marked xfail(strict) — if it ever passes, the leak
was fixed and the marker must be removed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from config import PAD_TOKEN_ID


def _logits(model, tokens_np, max_steps=2):
    out = model(jnp.asarray(tokens_np), max_steps=max_steps, training=False, should_refresh=True)
    return np.asarray(out.logits, dtype=np.float32)


def test_pad_tail_length_does_not_change_real_token_logits(tiny_model, token_batch):
    real = token_batch[:, :48]
    short = np.concatenate([real, np.full((1, 8), PAD_TOKEN_ID, dtype=np.int32)], axis=1)
    long = np.concatenate([real, np.full((1, 16), PAD_TOKEN_ID, dtype=np.int32)], axis=1)

    logits_short = _logits(tiny_model, short)[:, :48]
    logits_long = _logits(tiny_model, long)[:, :48]

    np.testing.assert_allclose(
        logits_short, logits_long, rtol=1e-3, atol=1e-3,
        err_msg="Logits over real tokens depend on how much PAD follows them — "
                "a padding mask is leaking somewhere.",
    )


@pytest.mark.xfail(
    strict=True,
    reason="Known future-information leak: reasoning slots summarize the full "
           "window bidirectionally and the decoder attends to them from every "
           "position. Training CE and the depth-curve diagnostic are optimistic "
           "until the architecture closes this. Remove this marker when fixed.",
)
def test_future_token_cannot_influence_past_predictions(tiny_model, token_batch):
    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    base = _logits(tiny_model, token_batch)[:, :40]
    after = _logits(tiny_model, perturbed)[:, :40]

    np.testing.assert_allclose(base, after, rtol=1e-3, atol=1e-3)
