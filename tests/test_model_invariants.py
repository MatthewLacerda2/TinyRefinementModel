"""Whole-model invariants: padding correctness and causality.

Causality history: until 2026-06-11 the decoder read this window's reasoning
output, whose slots had seen the whole window bidirectionally — a future-token
leak (the slot-future-leak post-mortem in ROADMAP's Post-mortems section). Fixed by decoding against
the slots the window started with; the loop's output now only reaches the NEXT
window through the hunch cache. Both causality tests below guard that fix: the
fresh-slot path and the carried-hunch path (whose gate once peeked at the
current window's mean — the second leak).
"""

import jax.numpy as jnp
import numpy as np

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


def test_future_token_cannot_influence_past_predictions(tiny_model, token_batch):
    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    base = _logits(tiny_model, token_batch)[:, :40]
    after = _logits(tiny_model, perturbed)[:, :40]

    np.testing.assert_allclose(base, after, rtol=1e-3, atol=1e-3)


def test_causality_holds_with_carried_hunch(tiny_model, token_batch):
    """The riskier path: decode window B against the hunch carried from window A.
    A future token in B must still not influence B's earlier predictions.
    (Window A influencing all of B is legitimate — A is entirely in the past.)"""
    window_a = (token_batch + 17) % 5000 + 1

    def run(tokens_b):
        tiny_model(jnp.asarray(window_a), max_steps=2, training=False, should_refresh=True)
        out = tiny_model(jnp.asarray(tokens_b), max_steps=2, training=False, should_refresh=False)
        return np.asarray(out.logits, dtype=np.float32)[:, :40]

    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    np.testing.assert_allclose(run(token_batch), run(perturbed), rtol=1e-3, atol=1e-3)
