"""Metrics with a fixed range, used to condemn rows their accumulator broke.

Why a fixed-range metric is worth checking is `instruments/invariants.py`'s module
docstring. These tests pin the two properties that make the check worth having: it
fires on an impossible row, and it does not fire on a healthy run.
"""

import math

from instruments import invariants
from instruments.runlog import RunLog


def _log(rows):
    return RunLog(run_id="test", metrics=[{"step": s, **r} for s, r in rows], metadata={})


def test_a_condemned_row_takes_its_whole_row_with_it():
    """The accumulator is shared. If one value on a row is impossible, the CE computed
    from the same window is wrong too."""
    log = _log([(10, {"ce": 3.2, "zero_frac_dense_max": 0.01}),
                (20, {"ce": 1.87, "zero_frac_dense_max": 1.5})])
    steps, values = invariants.clean_column(log, "ce")
    assert steps == [10] and values == [3.2], "the CE on a condemned row must go too"


def test_a_healthy_run_has_no_suspect_rows():
    """A check that fires on a healthy run is worse than no check — it teaches the
    reader to ignore it."""
    rows = [(i * 5, {"ce": 3.2, "out_entropy": 4.0, "zero_frac_dense_max": 1e-4,
                     "grad_norm_avg": 0.5, "act_max": 50.0}) for i in range(1, 60)]
    assert invariants.suspect_rows(_log(rows)) == {}


def test_cross_entropy_is_not_bounded_above_by_ln_vocab():
    """A deliberate non-invariant. It is tempting to bound CE by ln(VOCAB_SIZE) on
    the grounds that no model should be worse than uniform — and a freshly
    initialised one can be. This project logged 11.08 at step 5, above
    ln(50304) = 10.83. Bounding it there would condemn the opening rows of every
    single from-scratch run."""
    log = _log([(5, {"ce": 11.0810})])
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
    """A column the run did not log (#105) is absent, not wrong."""
    assert invariants.suspect_rows(_log([(10, {"ce": 3.2, "out_entropy": None})])) == {}


def test_fractions_must_lie_in_zero_to_one():
    assert 10 in invariants.suspect_rows(_log([(10, {"zero_frac_dense_max": 1.5})]))
    assert invariants.suspect_rows(_log([(10, {"zero_frac_dense_max": 0.5}) ])) == {}
