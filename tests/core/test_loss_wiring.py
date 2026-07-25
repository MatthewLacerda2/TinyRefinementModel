"""Guards against the forget-cost incident: a loss component that is computed
and logged but never added to the total loss (commit e516132 fixed ~3900 opt
steps of training where the model never learned to forget). Every cost a model
reports must provably influence the total loss.

Post-#105 that wiring runs through the neutral contract — the model reports raw
terms in `LMOutput.aux`, weights them in its own `grade_aux`, and
`compute_total_loss` adds back whatever it gets — so the guard is generic over
architectures: it asks each arch which terms it reports and checks those, rather
than naming forget and diversity. A new architecture with a new regularizer is
covered the day it ships, and an arch that reports nothing has nothing to lose.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from grad_step import compute_total_loss
from schedules import sample_reasoning_depth
from config import MAX_STEPS_LIMIT

# Any step past warmup, so no schedule is legitimately sitting at zero.
GRADING_STEP = 1000


def _total(graded_aux, ce1=2.0, ce2=3.0):
    return float(compute_total_loss(jnp.array(ce1), jnp.array(ce2), graded_aux))


def test_every_graded_term_reaches_the_total():
    base = _total({"a": jnp.array(1.0), "b": jnp.array(1.0)})
    assert _total({"a": jnp.array(2.0), "b": jnp.array(1.0)}) != base, "term 'a' is not added"
    assert _total({"a": jnp.array(1.0), "b": jnp.array(2.0)}) != base, "term 'b' is not added"
    assert _total({}, ce1=4.0) != _total({}), "ce1 is not wired into the loss"
    assert _total({}, ce2=4.0) != _total({}), "ce2 is not wired into the loss"


def test_no_aux_means_plain_cross_entropy():
    # A model with no auxiliary objectives must pay exactly the CE — not the CE
    # plus a zero term it was obliged to invent (#105).
    assert _total({}) == pytest.approx(5.0)


def test_both_windows_weigh_equally():
    # No anchor/refinement asymmetry: raising either window's CE by the same
    # amount must change the loss by the same amount.
    base = _total({}, ce1=2.0, ce2=2.0)
    via_ce1 = _total({}, ce1=3.0, ce2=2.0)
    via_ce2 = _total({}, ce1=2.0, ce2=3.0)
    assert abs(via_ce1 - via_ce2) < 1e-6
    assert abs((via_ce1 - base) - 1.0) < 1e-6


def _graded(model, window_aux):
    return {k: float(v) for k, v in model.grade_aux(window_aux, GRADING_STEP).items()}


def test_each_reported_aux_term_survives_grading(tiny_model, token_batch):
    """Whatever the arch puts in `aux`, its own grade_aux must carry through to a
    graded term that responds to it — in BOTH windows. This is the forget-cost
    incident's actual shape: a cost computed, reported, and then dropped on the
    floor between the model and the loss."""
    reported = tiny_model(jnp.asarray(token_batch), depth=2, training=False).aux
    if not reported:
        pytest.skip("this architecture reports no auxiliary objectives")

    ones = {k: jnp.array(1.0) for k in reported}
    base = _graded(tiny_model, [ones, ones])

    for key in reported:
        for window in (0, 1):
            bumped = [dict(ones), dict(ones)]
            bumped[window][key] = jnp.array(2.0)
            after = _graded(tiny_model, bumped)
            assert after != base, (
                f"reported aux term '{key}' does not reach the loss from window {window}"
            )


def test_grading_scales_with_the_reported_cost(tiny_model, token_batch):
    """A graded term must be driven by the cost it is named for and nothing else:
    doubling one reported term moves its own graded value and leaves the others
    alone. Catches a schedule wired to the wrong key."""
    reported = tiny_model(jnp.asarray(token_batch), depth=2, training=False).aux
    if not reported:
        pytest.skip("this architecture reports no auxiliary objectives")

    ones = {k: jnp.array(1.0) for k in reported}
    base = _graded(tiny_model, [ones, ones])

    for key in reported:
        bumped = dict(ones, **{key: jnp.array(2.0)})
        after = _graded(tiny_model, [bumped, ones])
        assert after[key] != base[key], f"'{key}' grading ignores its own reported cost"
        for other in reported:
            if other != key:
                assert after[other] == base[other], (
                    f"bumping '{key}' moved the graded '{other}' term — crossed wiring"
                )


def test_depth_sampling_is_deterministic_and_covers_range():
    depths = [sample_reasoning_depth(step) for step in range(2048)]
    replay = [sample_reasoning_depth(step) for step in range(2048)]
    assert depths == replay, "depth sampling must replay identically on resume"
    assert min(depths) == 1 and max(depths) == MAX_STEPS_LIMIT
    assert set(depths) == set(range(1, MAX_STEPS_LIMIT + 1)), "all depths must be sampled"
