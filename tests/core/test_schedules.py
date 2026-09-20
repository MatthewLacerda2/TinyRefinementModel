"""LR-horizon resolution (#83): DECAY_STEPS derives from the run's token budget
so the cosine bottoms out when training ends, not 2.0B tokens in.

Pinned here: the derivation math, the acceptance criteria (end step hits
end_value, half-budget is mid-cosine), the unset-env default that keeps the
golden run untouched, the loud failure on a degenerate budget, the deliberate
decision that the λ anneals keep their own absolute horizon, and the recording
of budget + resolved horizon in run metadata. Env-override cases are fresh imports
in one child interpreter (`import_config_under`, tests/conftest.py), because config.py
reads the environment at import time.
"""

import os

import numpy as np
import pytest

from trm.config import TOKENS_PER_OPT_STEP
from trm.model.reasoner import LAMBDA_DECAY_STEPS, diversity_lambda_schedule, forget_lambda_schedule
from trm.train.schedules import (
    DECAY_STEPS,
    PEAK_LR,
    WARMUP_STEPS,
    build_learning_schedule,
    resolve_decay_steps,
)

LR_PEAK, LR_END = 6e-4, 6e-6


def test_tokens_per_opt_step_matches_recipe():
    """The issue's arithmetic: 128 accumulation × batch 1 × 1024 target tokens
    per micro-step. If the batch recipe changes, this constant must follow."""
    assert TOKENS_PER_OPT_STEP == 128 * 1 * 2 * 512


@pytest.mark.skipif(bool(os.environ.get("TRAIN_TOKEN_BUDGET")),
                    reason="suite invoked with an explicit TRAIN_TOKEN_BUDGET")
def test_unset_budget_keeps_historical_horizon():
    """No budget → today's 15000, so existing configs (and the golden run's
    trajectory) resolve exactly as before this change."""
    assert resolve_decay_steps(None) == 15000
    assert DECAY_STEPS == 15000


def test_derived_horizon_is_budget_over_tokens_per_step():
    assert resolve_decay_steps(40_000 * TOKENS_PER_OPT_STEP) == 40_000


def test_anneal_ends_at_budget_and_half_budget_is_mid_cosine():
    """The acceptance criteria: schedule value at the configured end step equals
    end_value; at half-budget it's mid-cosine."""
    decay = resolve_decay_steps(30_000 * TOKENS_PER_OPT_STEP)
    sched = build_learning_schedule(decay)

    assert np.isclose(float(sched(decay)), LR_END, rtol=1e-6)
    assert np.isclose(float(sched(decay * 2)), LR_END, rtol=1e-6)  # floor, not rebound

    half = decay // 2
    frac = (half - WARMUP_STEPS) / (decay - WARMUP_STEPS)
    expected = LR_END + 0.5 * (LR_PEAK - LR_END) * (1 + np.cos(np.pi * frac))
    assert np.isclose(float(sched(half)), expected, rtol=1e-3)


def test_peak_lr_defaults_to_the_adopted_value(monkeypatch):
    """6e-4 since #388, the peak #287 measured; the whole schedule follows it (init
    peak/10, end peak/100)."""
    assert PEAK_LR == LR_PEAK
    sched = build_learning_schedule(15_000)
    assert np.isclose(float(sched(WARMUP_STEPS)), LR_PEAK, rtol=1e-5)
    assert np.isclose(float(sched(0)), 6e-5, rtol=1e-5)
    assert np.isclose(float(sched(15_000)), LR_END, rtol=1e-5)


def test_the_whole_schedule_scales_with_the_peak():
    """A sweep of the peak moves one variable, the LR scale: init and end follow
    it at peak/10 and peak/100, so the shape at every step is identical and only
    the height differs. Three separate knobs would be three variables."""
    base, tripled = build_learning_schedule(15_000, peak_lr=1e-4), build_learning_schedule(15_000, peak_lr=3e-4)
    for step in (0, WARMUP_STEPS, 5_000, 15_000):
        assert np.isclose(float(tripled(step)), 3.0 * float(base(step)), rtol=1e-5), step


def test_budget_inside_warmup_fails_loud():
    with pytest.raises(ValueError):
        resolve_decay_steps(WARMUP_STEPS * TOKENS_PER_OPT_STEP // 2)


def test_lambda_schedules_keep_their_own_horizon():
    """Deliberate (#83): the λ anneals relax regularization pressure over early
    training — absolute-step dynamics — and must not silently stretch with the
    LR horizon. Their end values land at LAMBDA_DECAY_STEPS regardless."""
    assert LAMBDA_DECAY_STEPS == 15000
    assert np.isclose(float(forget_lambda_schedule(LAMBDA_DECAY_STEPS)), 0.001, rtol=1e-6)
    assert np.isclose(float(diversity_lambda_schedule(LAMBDA_DECAY_STEPS)), 0.1, rtol=1e-6)


def test_env_override_resolves_horizon(import_config_under):
    budget = 20_000 * TOKENS_PER_OPT_STEP
    # Scientific notation is accepted: 2e9 ≈ the historical 2.0B-token horizon.
    exact, scientific = import_config_under(
        [{"TRAIN_TOKEN_BUDGET": str(budget)}, {"TRAIN_TOKEN_BUDGET": "2e9"}],
        attrs=("TRAIN_TOKEN_BUDGET", "trm.train.schedules:DECAY_STEPS"))
    for case in (exact, scientific):
        assert case["ok"], f"the budget override refused to import: {case.get('error')}"
    assert exact["values"] == {"TRAIN_TOKEN_BUDGET": budget, "trm.train.schedules:DECAY_STEPS": 20_000}
    assert scientific["values"] == {"TRAIN_TOKEN_BUDGET": 2_000_000_000,
                                    "trm.train.schedules:DECAY_STEPS": round(2e9 / TOKENS_PER_OPT_STEP)}


def test_horizon_recorded_in_run_metadata():
    from trm.runtime.run_tracker import RunTracker
    params = RunTracker.get_hyperparameters()
    assert params["DECAY_STEPS"] == DECAY_STEPS
    assert "TRAIN_TOKEN_BUDGET" in params


# ── every schedule declares its horizon (#362) ───────────────────────────────

def test_every_schedule_declares_what_its_horizon_scales_with():
    from trm.train import schedules

    # The default is WSD since 2026-09-20, so its decay start declares a horizon too.
    assert set(schedules.SCHEDULE_HORIZONS) == {"warmup", "lr wsd", "mixture ramp", "wsd decay start"}
    kinds = {name: kind for name, (kind, _) in schedules.SCHEDULE_HORIZONS.items()}
    assert kinds == {"warmup": "absolute", "lr wsd": "budget", "mixture ramp": "budget",
                     "wsd decay start": "budget"}
    assert schedules.SCHEDULE_HORIZONS["lr wsd"][1] == schedules.DECAY_STEPS


def test_the_ramp_keeps_the_champions_shape_at_4b_and_scales_down_for_a_pair():
    """A 4B base run resolves to the 10,000-step ramp it always had; a 512-step pair
    turns web to code/math over a third of its own length instead of barely starting."""
    from trm.train.schedules import resolve_curriculum_steps

    assert resolve_curriculum_steps(4_000_000_000, 30518) == 10000
    assert resolve_curriculum_steps(67_108_864, 512) == 168
    assert resolve_curriculum_steps(None, 15000) == 10000, "no budget: the historical ramp"


# ── warmup-stable-decay (#386) ───────────────────────────────────────────────

def test_wsd_holds_the_peak_then_decays_to_the_cosines_own_end():
    import pytest

    from trm.train.schedules import build_learning_schedule, build_wsd_schedule

    wsd = build_wsd_schedule(512, warmup_steps=100, peak_lr=6e-4, decay_fraction=0.2)
    cosine = build_learning_schedule(512, warmup_steps=100, peak_lr=6e-4)
    assert float(wsd(0)) == pytest.approx(float(cosine(0)), rel=1e-6), "same start"
    assert float(wsd(512)) == pytest.approx(float(cosine(512)), rel=1e-6), "same end"
    for step in (100, 300, 409):
        assert float(wsd(step)) == pytest.approx(6e-4, rel=1e-6), f"the stable phase holds the peak ({step})"
    assert 6e-4 > float(wsd(461)) > float(wsd(500)), "the last 20% decays"


def test_a_branched_decay_starts_where_it_is_told():
    """A decay branched from checkpoint N (the base run's stop rule) starts at N."""
    import pytest

    from trm.train.schedules import build_wsd_schedule

    branch = build_wsd_schedule(12_000, warmup_steps=1000, peak_lr=6e-4, decay_start=10_000)
    assert float(branch(9_999)) == pytest.approx(6e-4, rel=1e-6)
    assert float(branch(11_000)) < 6e-4


def test_a_decay_that_cannot_fit_refuses():
    import pytest

    from trm.train.schedules import build_wsd_schedule

    with pytest.raises(ValueError, match="between the warmup"):
        build_wsd_schedule(512, warmup_steps=100, decay_start=50)
