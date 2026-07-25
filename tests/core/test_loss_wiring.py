"""Guards against the forget-cost incident: a loss component that is computed
and logged but never added to the total loss (commit e516132 fixed ~3900 opt
steps of training where the model never learned to forget). Every cost the
model reports must provably influence the total loss."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax.numpy as jnp

from grad_step import compute_total_loss
from schedules import sample_reasoning_depth
from config import MAX_STEPS_LIMIT


def _out(diversity=1.0, forget=1.0):
    return SimpleNamespace(
        diversity_loss=jnp.array(diversity),
        forget_cost=jnp.array(forget),
    )


def _total(out1, out2, ce1=2.0, ce2=3.0, f_lambda=0.1, d_lambda=0.1):
    return float(compute_total_loss(
        jnp.array(ce1), jnp.array(ce2), out1, out2, f_lambda, d_lambda
    ))


def test_every_cost_component_is_wired():
    base = _total(_out(), _out())
    # Doubling any single component must change the total loss.
    assert _total(_out(forget=2.0), _out()) != base, "forget cost (segment 1) is not wired into the loss"
    assert _total(_out(), _out(forget=2.0)) != base, "forget cost (segment 2) is not wired into the loss"
    assert _total(_out(diversity=2.0), _out()) != base, "diversity loss is not wired into the loss"
    assert _total(_out(), _out(), ce1=4.0) != base, "ce1 is not wired into the loss"
    assert _total(_out(), _out(), ce2=4.0) != base, "ce2 is not wired into the loss"


def test_zero_lambda_disables_exactly_its_component():
    # With a lambda of zero, the corresponding component must stop mattering.
    assert _total(_out(forget=1.0), _out(), f_lambda=0.0) == _total(_out(forget=99.0), _out(), f_lambda=0.0)
    assert _total(_out(diversity=1.0), _out(), d_lambda=0.0) == _total(_out(diversity=99.0), _out(), d_lambda=0.0)


def test_both_segments_weigh_equally():
    # No anchor/refinement asymmetry: raising either segment's CE by the same
    # amount must change the loss by the same amount.
    out1, out2 = _out(), _out()
    base = _total(out1, out2, ce1=2.0, ce2=2.0)
    via_ce1 = _total(out1, out2, ce1=3.0, ce2=2.0)
    via_ce2 = _total(out1, out2, ce1=2.0, ce2=3.0)
    assert abs(via_ce1 - via_ce2) < 1e-6
    assert abs((via_ce1 - base) - 1.0) < 1e-6


def test_depth_sampling_is_deterministic_and_covers_range():
    depths = [sample_reasoning_depth(step) for step in range(2048)]
    replay = [sample_reasoning_depth(step) for step in range(2048)]
    assert depths == replay, "depth sampling must replay identically on resume"
    assert min(depths) == 1 and max(depths) == MAX_STEPS_LIMIT
    assert set(depths) == set(range(1, MAX_STEPS_LIMIT + 1)), "all depths must be sampled"
