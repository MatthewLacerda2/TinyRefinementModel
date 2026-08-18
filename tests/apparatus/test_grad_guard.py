"""Per-micro-step gradient clipping (#201), against the distribution that motivated it.

The run died at opt step 11,650 with every telemetry reading looking healthy, because
the only clip in the pipeline ran on the mean of 128 micro-steps. Replaying 9,000 of
those micro-steps showed why that cannot work here: p50 11.4, p99 603.5, max 75,331 —
the top 1% of micro-steps carry 64.5% of the total gradient mass, so the accumulated
mean points where the outlier points and the clip only resizes it.

These pin the two things a guard like this gets wrong in opposite directions: letting
an outlier raise the bar meant to catch it, and clipping so much ordinary traffic that
it quietly becomes a different training recipe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trm.config import MAX_SEQ_LEN
from trm.train.grad_guard import GradientNormGuard
from trm.train.grad_step import compute_grad_step


def _settled(norm=10.0, warmup=8, **kwargs):
    """A guard past warmup with its EMA sitting at `norm`."""
    guard = GradientNormGuard(warmup=warmup, **kwargs)
    for _ in range(warmup * 3):
        guard.observe(norm)
    return guard


# ── the rule ─────────────────────────────────────────────────────────────────

def test_nothing_is_clipped_during_warmup():
    """An EMA seeded from one micro-step is not an estimate of anything, and
    clipping against it would corrupt the opening steps of every fresh run."""
    guard = GradientNormGuard(warmup=16)
    for _ in range(15):
        assert guard.observe(50.0) is None
    assert guard.threshold is None and guard.clipped == 0


def test_the_ceiling_is_a_multiple_of_the_typical_norm():
    guard = _settled(norm=10.0, multiplier=8.0)
    assert guard.threshold == pytest.approx(80.0, rel=1e-3)


def test_an_outlier_barely_moves_the_bar_that_should_catch_it():
    """Feed the EMA raw norms and one 75,331 drags the ceiling up for hundreds of
    micro-steps afterwards — exactly when the guard is most needed. The capped
    update keeps a single monster near-powerless."""
    guard = _settled(norm=10.0, multiplier=8.0)
    before = guard.threshold
    guard.observe(75_330.7)
    assert guard.threshold <= before * 1.02, (
        f"one monster moved the ceiling {before:.1f} -> {guard.threshold:.1f}")
    assert guard.clipped == 1


def test_the_ceiling_never_climbs_to_meet_the_outliers():
    """The regression this file exists for. The first implementation fed the CEILING
    back into the EMA; since the ceiling is already 8x the EMA, every clipped step
    raised it 7% and 200 of them ran it up 800,000-fold — the guard disabled itself
    precisely while firing. It must stay orders of magnitude below the monsters even
    under sustained assault."""
    guard = _settled(norm=10.0, multiplier=8.0)
    for _ in range(200):
        guard.observe(75_330.7)
    assert guard.threshold < 75_330.7 / 10, (
        f"ceiling ran up to {guard.threshold:.0f}, within reach of the outliers it "
        f"is supposed to clip")


def test_non_finite_norms_are_ignored_entirely():
    """They belong to the loss scaler (#199). One inf folded into the EMA would
    destroy it permanently and disable the guard for the rest of the run."""
    guard = _settled(norm=10.0)
    healthy = guard.ema
    guard.observe(float("inf"))
    guard.observe(float("nan"))
    assert guard.ema == pytest.approx(healthy)
    assert guard.observed == 24, "non-finite steps must not count as observations"


# ── against the real distribution ────────────────────────────────────────────

def _measured_sample(n=9000, seed=0):
    """A heavy-tailed sample matching the measured quantiles (p50 11.4, p90 24.7,
    p99 603.5, p99.9 7,617, max 75,331). Lognormal reproduces that shape closely."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(11.4), 1.35, size=n))


def test_it_clips_the_tail_and_leaves_the_body_alone():
    """The property that makes this safe to switch on mid-run: the overwhelming
    majority of micro-steps must pass through bit-identical. A guard that clips a
    third of the traffic is not a guard, it is a different recipe."""
    guard = GradientNormGuard()
    for norm in _measured_sample():
        guard.observe(float(norm))
    assert 0.001 < guard.clip_rate < 0.10, (
        f"clip rate {guard.clip_rate:.2%} — outside the band that is 'catch the tail, "
        f"leave the body'")


def test_the_ceiling_tracks_a_shrinking_run():
    """Gradient magnitudes fall as a model converges. A constant calibrated against
    today's p50 of 11.4 would stop clipping anything by step 25,000 — which is the
    whole reason this is an EMA and not a number in config.py."""
    guard = GradientNormGuard(warmup=64)
    for _ in range(2000):
        guard.observe(100.0)
    high = guard.threshold
    for _ in range(2000):
        guard.observe(1.0)
    assert guard.threshold < high / 10, "the ceiling did not follow the run down"


# ── the wiring ───────────────────────────────────────────────────────────────

def test_the_clip_is_applied_to_the_gradients_and_the_reported_norm_is_not():
    """grad_norm is the run's oldest continuous telemetry and its docstring promises
    it is pre-clip. Rewriting that meaning would silently break every historical
    comparison in metrics.csv, so the clip must show in the gradients only."""
    from trm.model.refiner_lm import RefinerForTraining
    from flax import nnx

    model = RefinerForTraining(64, nnx.Rngs(0), vocab_size=5000, num_heads=4,
                               encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN)
    rng = np.random.default_rng(3)
    batch = jnp.asarray(rng.integers(1, 60, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))

    _, _, loose, raw_norm = compute_grad_step(model, batch, jnp.array(1), 2)
    ceiling = float(raw_norm) / 10.0
    _, _, tight, clipped_report = compute_grad_step(model, batch, jnp.array(1), 2,
                                                    clip_norm=jnp.float32(ceiling))

    assert float(clipped_report) == pytest.approx(float(raw_norm), rel=1e-6), \
        "the reported norm must stay pre-clip"
    tight_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g))
                                    for g in jax.tree_util.tree_leaves(tight))))
    assert tight_norm == pytest.approx(ceiling, rel=1e-3), "gradients were not clipped"
    # Direction preserved: clipping rescales, it does not rotate.
    a = np.concatenate([np.asarray(g).ravel() for g in jax.tree_util.tree_leaves(loose)])
    b = np.concatenate([np.asarray(g).ravel() for g in jax.tree_util.tree_leaves(tight)])
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine == pytest.approx(1.0, abs=1e-5)


def test_an_infinite_ceiling_is_exactly_todays_behaviour():
    """The default, and what every caller that has not opted in still gets."""
    from trm.model.refiner_lm import RefinerForTraining
    from flax import nnx

    model = RefinerForTraining(64, nnx.Rngs(0), vocab_size=5000, num_heads=4,
                               encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN)
    rng = np.random.default_rng(3)
    batch = jnp.asarray(rng.integers(1, 60, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))

    _, _, default, _ = compute_grad_step(model, batch, jnp.array(1), 2)
    _, _, explicit, _ = compute_grad_step(model, batch, jnp.array(1), 2,
                                          clip_norm=jnp.float32(jnp.inf))
    for a, b in zip(jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(explicit)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
