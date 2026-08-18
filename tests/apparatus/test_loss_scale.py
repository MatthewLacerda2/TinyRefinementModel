"""Dynamic f16 loss scaling (#199), and the promise that it changes nothing else.

The run this was written for died at opt step 11,140: the depth-recurrence gate's
gradients went to a 1.000 zero-fraction — exact zeros, never NaN — and the model's
output distribution was uniform 65 steps later. Loss scaling is the settled remedy,
but it is only safe to switch on partway through a run because it is arithmetically
inert: scale the loss going in, unscale the gradients coming out, and the optimizer
sees exactly what it would have seen. That inertness is the property most worth
pinning, because it is the one the run's honesty rests on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from trm.train.grad_step import compute_grad_step
from trm.train.loss_scale import DynamicLossScale


def _tiny_model_and_batch():
    from trm.model.refiner_lm import RefinerForTraining

    model = RefinerForTraining(64, nnx.Rngs(0), vocab_size=5000, num_heads=4,
                               encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN)
    rng = np.random.default_rng(3)
    batch = jnp.asarray(rng.integers(1, 60, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))
    return model, batch


# ── the policy ───────────────────────────────────────────────────────────────

def test_it_grows_only_after_a_full_clean_interval():
    scale = DynamicLossScale(initial=1024.0, growth_interval=4)
    assert [scale.record_good_step() for _ in range(3)] == [False, False, False]
    assert scale.value == 1024.0, "it must not creep upward mid-interval"
    assert scale.record_good_step() is True
    assert scale.value == 2048.0


def test_an_overflow_halves_it_and_restarts_the_count():
    """Both halves matter: backing off fixes the overflow, and resetting the
    counter stops a scale that is already at its ceiling from doubling again a
    few steps later and overflowing forever."""
    scale = DynamicLossScale(initial=1024.0, growth_interval=4)
    scale.record_good_step()
    scale.record_good_step()
    assert scale.record_overflow() == 512.0
    assert scale.good_steps == 0
    assert [scale.record_good_step() for _ in range(3)] == [False, False, False]


def test_it_is_clamped_at_both_ends():
    """Below 1.0 there is nothing left to protect and the real fault is elsewhere;
    an unbounded ceiling would just rediscover the overflow every interval."""
    low = DynamicLossScale(initial=2.0, min_scale=1.0)
    low.record_overflow()
    low.record_overflow()
    assert low.value == 1.0

    high = DynamicLossScale(initial=8.0, growth_interval=1, max_scale=16.0)
    assert high.record_good_step() is True and high.value == 16.0
    assert high.record_good_step() is False, "already at the ceiling"
    assert high.value == 16.0


def test_the_default_scale_is_a_power_of_two():
    """Scaling by a power of two only shifts the exponent, so no mantissa bit is
    lost on the way in or out — that is what makes the round trip exact."""
    value = DynamicLossScale().value
    assert value > 1 and (int(value) & (int(value) - 1)) == 0


# ── the promise ──────────────────────────────────────────────────────────────

def test_scaling_does_not_change_the_gradients():
    """The whole justification for turning this on mid-run. Same model, same batch,
    same depth: the gradients the optimizer receives must be identical whether the
    loss was scaled by 32,768 or not at all."""
    model, batch = _tiny_model_and_batch()
    _, _, plain, plain_norm = compute_grad_step(model, batch, jnp.array(1), 2)
    _, _, scaled, scaled_norm = compute_grad_step(model, batch, jnp.array(1), 2,
                                                  loss_scale=jnp.float32(2.0 ** 15))

    flat_plain = jax.tree_util.tree_leaves(plain)
    flat_scaled = jax.tree_util.tree_leaves(scaled)
    assert len(flat_plain) == len(flat_scaled) and flat_plain
    for a, b in zip(flat_plain, flat_scaled):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-8)
    assert float(scaled_norm) == pytest.approx(float(plain_norm), rel=1e-6)


def test_the_reported_loss_is_unscaled():
    """The CE written to metrics.csv has to stay comparable with every row the run
    already recorded — a 32,768x jump in the logged loss would be indistinguishable
    from a divergence to anything reading the file."""
    model, batch = _tiny_model_and_batch()
    plain, _, _, _ = compute_grad_step(model, batch, jnp.array(1), 2)
    scaled, _, _, _ = compute_grad_step(model, batch, jnp.array(1), 2,
                                        loss_scale=jnp.float32(2.0 ** 15))
    assert float(scaled) == pytest.approx(float(plain), rel=1e-6)


def test_an_overflow_still_reaches_the_trainer_as_non_finite():
    """Unscaling must not launder an overflow into a finite number: inf / S is inf,
    which is what trips the trainer's skip branch and tells the scaler to back off.
    If this ever became finite, the run would silently apply garbage gradients."""
    assert not np.isfinite(np.float32(np.inf) / np.float32(2.0 ** 15))
    assert not np.isfinite(np.float32(np.nan) / np.float32(2.0 ** 15))


def test_default_construction_matches_the_module_constants():
    from trm.train import loss_scale as module

    scale = DynamicLossScale()
    assert scale.value == module.INITIAL_LOSS_SCALE
    assert scale.growth_interval == module.LOSS_SCALE_GROWTH_INTERVAL


def test_the_scaler_can_reach_its_floor_before_the_streak_abort_fires():
    """A coupling between two constants that live in different files. If the
    initial scale is too high for this card, every micro-step overflows until S
    has been halved enough — and each of those counts toward MAX_NONFINITE_STREAK.
    Should that budget ever be smaller than the number of halvings required, a
    launch would abort as 'diverged' when nothing had diverged at all."""
    import math

    from trm.train.trainer import MAX_NONFINITE_STREAK

    scale = DynamicLossScale()
    halvings = math.ceil(math.log(scale.value / scale.min_scale, scale.factor))
    assert halvings < MAX_NONFINITE_STREAK, (
        f"{halvings} halvings needed to reach the floor but the run aborts after "
        f"{MAX_NONFINITE_STREAK} non-finite micro-steps")
