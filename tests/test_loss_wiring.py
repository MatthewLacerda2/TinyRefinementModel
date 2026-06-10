"""Guards against the forget-cost incident: a loss component that is computed
and logged but never added to the total loss (commit e516132 fixed ~3900 opt
steps of training where the model never learned to forget). Every cost the
model reports must provably influence the total loss."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax.numpy as jnp

from grad_step import compute_total_loss
from schedules import REFINEMENT_LOSS_WEIGHT, ANCHOR_CE_WEIGHT


def _out(diversity=1.0, forget=1.0, ponder=1.0):
    return SimpleNamespace(
        diversity_loss=jnp.array(diversity),
        forget_cost=jnp.array(forget),
        ponder_cost=jnp.array(ponder),
    )


def _total(out1, out2, ce1=2.0, ce2=3.0, f_lambda=0.1, d_lambda=0.1, p_lambda=0.1):
    return float(compute_total_loss(
        jnp.array(ce1), jnp.array(ce2), out1, out2, f_lambda, d_lambda, p_lambda
    ))


def test_every_cost_component_is_wired():
    base = _total(_out(), _out())
    # Doubling any single component must change the total loss.
    assert _total(_out(forget=2.0), _out()) != base, "forget cost (segment 1) is not wired into the loss"
    assert _total(_out(), _out(forget=2.0)) != base, "forget cost (segment 2) is not wired into the loss"
    assert _total(_out(diversity=2.0), _out()) != base, "diversity loss is not wired into the loss"
    assert _total(_out(ponder=2.0), _out()) != base, "ponder cost is not wired into the loss"
    assert _total(_out(), _out(), ce1=4.0) != base, "ce1 is not wired into the loss"
    assert _total(_out(), _out(), ce2=4.0) != base, "ce2 is not wired into the loss"


def test_zero_lambda_disables_exactly_its_component():
    # With a lambda of zero, the corresponding component must stop mattering.
    assert _total(_out(forget=1.0), _out(), f_lambda=0.0) == _total(_out(forget=99.0), _out(), f_lambda=0.0)
    assert _total(_out(diversity=1.0), _out(), d_lambda=0.0) == _total(_out(diversity=99.0), _out(), d_lambda=0.0)
    assert _total(_out(ponder=1.0), _out(), p_lambda=0.0) == _total(_out(ponder=99.0), _out(), p_lambda=0.0)


def test_refinement_term_penalizes_ce2_above_ce1():
    out1, out2 = _out(), _out()
    gap = _total(out1, out2, ce1=2.0, ce2=3.0)
    flat = _total(out1, out2, ce1=2.0, ce2=2.0)
    # Raising ce2 by 1 adds its base CE plus the refinement penalty on the gap.
    expected = 1.0 + REFINEMENT_LOSS_WEIGHT * 1.0
    assert abs((gap - flat) - expected) < 1e-6


def test_anchor_weight_applies_to_ce1():
    out1, out2 = _out(), _out()
    hi = _total(out1, out2, ce1=3.0, ce2=5.0)
    lo = _total(out1, out2, ce1=2.0, ce2=5.0)
    # Raising ce1 by 1 adds (1 + anchor weight) but shrinks the refinement gap by 1.
    expected = (1.0 + ANCHOR_CE_WEIGHT) - REFINEMENT_LOSS_WEIGHT
    assert abs((hi - lo) - expected) < 1e-6
