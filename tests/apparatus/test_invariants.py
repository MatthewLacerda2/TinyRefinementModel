"""Metrics with a fixed expectation, used to condemn rows their accumulator broke.

The motivating case (#194): a resume wrote `ce = 1.8746` into metrics.csv, 40%
below its neighbours. Two explanations, very far apart in consequence — a
harmless logging artifact, or the data pipeline re-serving text already trained
on, which is silent in both directions by `split_samples`' own admission. What
settled it was `depth_avg = 2.7641` on the same row: depth is a mean of uniform
draws over 1..MAX_STEPS_LIMIT, so that number is not a measurement of anything.

These tests pin the two properties that make the check worth having: it fires on
the real artifact, and it does not fire on a healthy run.
"""

import math

import pytest

from instruments import invariants
from instruments.runlog import RunLog


def _log(rows):
    return RunLog(run_id="test", metrics=[{"step": s, **r} for s, r in rows], metadata={})


def test_the_depth_corridor_is_derived_from_the_config_not_hardcoded():
    """Change MAX_STEPS_LIMIT or ACCUMULATION_STEPS and the bounds must move with
    them. A hardcoded tolerance silently becomes wrong instead of loudly wrong."""
    from trm.config import ACCUMULATION_STEPS, MAX_STEPS_LIMIT

    mean, sigma = invariants.depth_expectation()
    assert mean == (MAX_STEPS_LIMIT + 1) / 2
    assert sigma == pytest.approx(
        math.sqrt((MAX_STEPS_LIMIT**2 - 1) / 12 / ACCUMULATION_STEPS))


def test_it_catches_the_resume_artifact():
    """The exact row from the #157 run, values unaltered."""
    log = _log([(5950, {"ce": 3.1450, "depth_avg": 4.4062}),
                (5955, {"ce": 1.8746, "depth_avg": 2.7641}),
                (5960, {"ce": 3.1754, "depth_avg": 4.4422})])
    suspect = invariants.suspect_rows(log)
    assert set(suspect) == {5955}
    assert "depth_avg" in suspect[5955][0]


def test_a_condemned_row_takes_its_whole_row_with_it():
    """The accumulator is shared. If depth is impossible, the CE computed from the
    same partial window is wrong too — and it was: both fell by the same ~0.6x."""
    log = _log([(10, {"ce": 3.2, "depth_avg": 4.5}),
                (20, {"ce": 1.87, "depth_avg": 2.76})])
    steps, values = invariants.clean_column(log, "ce")
    assert steps == [10] and values == [3.2], "the CE on a condemned row must go too"


def test_a_healthy_run_has_no_suspect_rows():
    """A check that fires on a healthy run is worse than no check — it teaches the
    reader to ignore it. Depth wobbles by ~1 sigma every row and must stay clean."""
    mean, sigma = invariants.depth_expectation()
    rows = [(i * 5, {"ce": 3.2, "depth_avg": mean + ((-1) ** i) * 2 * sigma})
            for i in range(1, 60)]
    assert invariants.suspect_rows(_log(rows)) == {}


def test_cross_entropy_is_not_bounded_above_by_ln_vocab():
    """A deliberate non-invariant. It is tempting to bound CE by ln(VOCAB_SIZE) on
    the grounds that no model should be worse than uniform — and a freshly
    initialised one can be. This project logged 11.08 at step 5, above
    ln(50304) = 10.83. Bounding it there would condemn the opening rows of every
    single from-scratch run."""
    log = _log([(5, {"ce": 11.0810, "depth_avg": 4.3031})])
    assert invariants.suspect_rows(log) == {}


def test_entropy_is_bounded_above_because_it_is_a_property_of_one_distribution():
    """Unlike CE, which compares two. No distribution over V outcomes has entropy
    exceeding ln(V), so this bound is real."""
    from trm.config import VOCAB_SIZE
    log = _log([(10, {"out_entropy": math.log(VOCAB_SIZE) + 1.0})])
    assert 10 in invariants.suspect_rows(log)


def test_non_finite_values_are_condemned():
    """NaN passes every comparison, so it has to be checked for explicitly — and a
    NaN in the metrics is exactly when you most want to be told."""
    for bad in (float("nan"), float("inf")):
        assert 10 in invariants.suspect_rows(_log([(10, {"ce": bad})]))


def test_blank_cells_are_not_violations():
    """An arch-optional column (#105) is absent, not wrong."""
    assert invariants.suspect_rows(_log([(10, {"ce": 3.2, "temporal_drift": None})])) == {}


def test_fractions_must_lie_in_zero_to_one():
    assert 10 in invariants.suspect_rows(_log([(10, {"zero_frac_dense_max": 1.5})]))
    assert invariants.suspect_rows(_log([(10, {"zero_frac_dense_max": 0.5}) ])) == {}


def test_the_live_run_has_exactly_the_two_known_artifacts():
    """Integration against the real file, when it is present. 1,567 rows, two
    resumes, two artifacts, no false positives — the ratio that makes the check
    trustworthy rather than noisy."""
    import pathlib
    from instruments.runlog import load

    csv = pathlib.Path("runs/run_20260813_214725/metrics.csv")
    if not csv.exists():
        pytest.skip("live run's metrics.csv not present")
    suspect = invariants.suspect_rows(load(str(csv)))
    assert all("depth_avg" in reasons[0] for reasons in suspect.values())
    assert len(suspect) <= 5, f"unexpectedly many suspect rows: {sorted(suspect)}"
