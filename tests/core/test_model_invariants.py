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

from trm.config import PAD_TOKEN_ID


def _logits(model, tokens_np, depth=2):
    out = model(jnp.asarray(tokens_np), depth=depth, training=False, new_document=True)
    return np.asarray(out.logits, dtype=np.float32)


def test_pad_run_length_does_not_change_the_real_tokens_after_it(tiny_model, token_batch):
    """PAD must be invisible to the real tokens that could attend to it.

    Until #322 the pads went at the TAIL, after the real tokens. On a causal-only
    model that is vacuous: nothing before a pad can attend to it, so deleting the
    pad mask entirely moved plain's logits by 1.9e-6, the same as the honest model.
    Here the pads come FIRST, where every real token after them could attend to
    them. With the mask, the run length is a pure position shift, and RoPE only
    sees relative distances between the (unmasked) real tokens, so the real-token
    logits must not depend on it."""
    real = token_batch[:, :48]
    short = np.concatenate([np.full((1, 8), PAD_TOKEN_ID, dtype=np.int32), real], axis=1)
    long = np.concatenate([np.full((1, 16), PAD_TOKEN_ID, dtype=np.int32), real], axis=1)

    logits_short = _logits(tiny_model, short)[:, 8:]
    logits_long = _logits(tiny_model, long)[:, 16:]

    np.testing.assert_allclose(
        logits_short, logits_long, rtol=1e-3, atol=1e-3,
        err_msg="Logits of real tokens depend on how many PAD tokens precede them — "
                "a padding mask is leaking somewhere.",
    )


def test_future_token_cannot_influence_past_predictions(tiny_model, token_batch):
    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    base = _logits(tiny_model, token_batch)[:, :40]
    after = _logits(tiny_model, perturbed)[:, :40]

    np.testing.assert_allclose(base, after, rtol=1e-3, atol=1e-3)


def test_causality_holds_with_carried_hunch(reasoner_model, token_batch):
    """The riskier path: decode window B against the hunch carried from window A.
    A future token in B must still not influence B's earlier predictions.
    (Window A influencing all of B is legitimate — A is entirely in the past.)"""
    window_a = (token_batch + 17) % 5000 + 1

    def run(tokens_b):
        reasoner_model(jnp.asarray(window_a), depth=2, training=False, new_document=True)
        out = reasoner_model(jnp.asarray(tokens_b), depth=2, training=False, new_document=False)
        return np.asarray(out.logits, dtype=np.float32)[:, :40]

    perturbed = token_batch.copy()
    perturbed[0, 40] = int(perturbed[0, 40]) + 1

    np.testing.assert_allclose(run(token_batch), run(perturbed), rtol=1e-3, atol=1e-3)
