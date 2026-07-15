"""LR-horizon resolution (#83): DECAY_STEPS derives from the run's token budget
so the cosine bottoms out when training ends, not 2.0B tokens in.

Pinned here: the derivation math, the acceptance criteria (end step hits
end_value, half-budget is mid-cosine), the unset-env default that keeps the
golden run untouched, the loud failure on a degenerate budget, the deliberate
decision that the λ anneals keep their own absolute horizon, and the recording
of budget + resolved horizon in run metadata. Env-override cases run in a
subprocess because config.py reads the environment at import time.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from config import TOKENS_PER_OPT_STEP
from schedules import (
    DECAY_STEPS,
    LAMBDA_DECAY_STEPS,
    WARMUP_STEPS,
    build_learning_schedule,
    diversity_lambda_schedule,
    forget_lambda_schedule,
    resolve_decay_steps,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LR_PEAK, LR_END = 1e-4, 1e-6


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


def _resolved_in_subprocess(env_overrides):
    env = {**os.environ, **env_overrides}
    out = subprocess.check_output(
        [sys.executable, "-c",
         "import json, config, schedules; print(json.dumps("
         "{'budget': config.TRAIN_TOKEN_BUDGET, 'decay': schedules.DECAY_STEPS}))"],
        env=env, cwd=REPO,
    )
    return json.loads(out)


def test_env_override_resolves_horizon():
    budget = 20_000 * TOKENS_PER_OPT_STEP
    assert _resolved_in_subprocess({"TRAIN_TOKEN_BUDGET": str(budget)}) == \
        {"budget": budget, "decay": 20_000}
    # Scientific notation is accepted: 2e9 ≈ the historical 2.0B-token horizon.
    resolved = _resolved_in_subprocess({"TRAIN_TOKEN_BUDGET": "2e9"})
    assert resolved == {"budget": 2_000_000_000,
                        "decay": round(2e9 / TOKENS_PER_OPT_STEP)}


def test_horizon_recorded_in_run_metadata():
    from run_tracker import RunTracker
    params = RunTracker.get_hyperparameters()
    assert params["DECAY_STEPS"] == DECAY_STEPS
    assert "TRAIN_TOKEN_BUDGET" in params
