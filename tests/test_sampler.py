"""Regression tests for issue #81: temperature must be applied to logits
before top-k/top-p truncation, not after — the nucleus/top-k cutoff has to
be computed on the distribution actually being sampled.
"""

import math

import jax.numpy as jnp
import numpy as np

from infer_local import _temperature_truncate

LOGITS = [2.0, 1.9, 1.8, 0.0, -1.0]


def _reference_nucleus_kept_indices(logits, temperature, top_p):
    """Independent (pure-Python) reference: temperature -> sort -> cumsum -> cutoff."""
    scaled = [logit / temperature for logit in logits]
    order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
    sorted_vals = [scaled[i] for i in order]
    m = max(sorted_vals)
    exps = [math.exp(v - m) for v in sorted_vals]
    total = sum(exps)
    probs = [e / total for e in exps]

    cum = 0.0
    kept = []
    for idx, p in zip(order, probs):
        if cum < top_p:
            kept.append(idx)
        cum += p
    return set(kept)


def _kept_indices(masked_logits):
    return {i for i, v in enumerate(np.asarray(masked_logits)) if np.isfinite(v)}


def test_temperature_one_matches_unscaled_nucleus():
    """At T=1 the kept token set must be unchanged vs the untempered nucleus."""
    logits = jnp.array(LOGITS)
    top_p = 0.7

    masked = _temperature_truncate(logits, temperature=1.0, top_k=0, top_p=top_p)
    kept = _kept_indices(masked)

    assert kept == _reference_nucleus_kept_indices(LOGITS, 1.0, top_p)


def test_temperature_scaling_changes_nucleus_and_matches_reference():
    """At T!=1 the kept set must match temperature-first truncation, and must
    differ from the untempered (pre-fix) nucleus on a vector built so the two
    orderings visibly disagree."""
    top_p = 0.7
    temperature = 0.3

    masked = _temperature_truncate(jnp.array(LOGITS), temperature=temperature, top_k=0, top_p=top_p)
    kept = _kept_indices(masked)

    reference = _reference_nucleus_kept_indices(LOGITS, temperature, top_p)
    assert kept == reference

    # Sharpening the distribution before truncation keeps a strictly smaller
    # nucleus than truncating on the raw (untempered) logits — this is the
    # case the pre-fix code (mask first, divide by T after) got wrong.
    untempered_kept = _reference_nucleus_kept_indices(LOGITS, 1.0, top_p)
    assert kept < untempered_kept


def test_top_k_selection_is_temperature_invariant():
    """top-k is order-preserving under any positive scale, so temperature
    should not change which tokens survive a top-k-only truncation."""
    logits = jnp.array(LOGITS)

    kept_t1 = _kept_indices(_temperature_truncate(logits, temperature=1.0, top_k=3, top_p=1.0))
    kept_t_sharp = _kept_indices(_temperature_truncate(logits, temperature=0.3, top_k=3, top_p=1.0))

    assert kept_t1 == kept_t_sharp == {0, 1, 2}
