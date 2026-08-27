"""Guards on the throughput panel's two silent-wrongness modes (#223).

Both were live in run_20260813_214725: the supervisor stopped writing heartbeats
at step 13,890 of 30,520, and the plot drew 46% of a run as though it were the
whole thing — with a "recent mean" and an ETA computed across the dead gap.
"""

import numpy as np

from instruments.plots import GAP_FACTOR, usable_intervals


def _steady(n, cadence_h=1.0, tok_per_s=4500.0):
    """n+1 heartbeats at a constant rate: hours and cumulative tokens."""
    hours = np.arange(n + 1, dtype=float) * cadence_h
    return hours, hours * 3600.0 * tok_per_s


def test_steady_run_recovers_the_true_rate():
    hours, tokens = _steady(20, tok_per_s=4500.0)
    rate, rate_hours, dropped, cadence = usable_intervals(hours, tokens)
    assert dropped == 0
    assert rate.size == 20 and rate_hours.size == 20
    assert np.allclose(rate, 4500.0)
    assert cadence == 1.0


def test_downtime_gap_is_dropped_not_averaged_in():
    """The failure that produced the fake 'throughput decline'.

    A long gap with few tokens in it is downtime, not a slow interval. Averaging
    it in drags the reported rate far below the truth.
    """
    hours, tokens = _steady(10)
    # 30 idle hours pass; only one cadence-worth of tokens is actually trained.
    hours = np.append(hours, hours[-1] + 30.0)
    tokens = np.append(tokens, tokens[-1] + 1.0 * 3600.0 * 4500.0)

    rate, _, dropped, _ = usable_intervals(hours, tokens)
    assert dropped == 1, "the 30h gap must be recognised as a gap"
    assert np.allclose(rate, 4500.0), "surviving intervals keep the true rate"
    # The bug: including it would report ~150 tok/s, a 30x understatement.
    naive = np.diff(tokens) / (np.diff(hours) * 3600.0)
    assert naive.min() < 200.0


def test_interval_just_under_the_threshold_is_kept():
    """Ordinary jitter in beat spacing must not be mistaken for downtime."""
    hours, tokens = _steady(10)
    stretch = GAP_FACTOR * 0.9
    hours = np.append(hours, hours[-1] + stretch)
    tokens = np.append(tokens, tokens[-1] + stretch * 3600.0 * 4500.0)

    rate, _, dropped, _ = usable_intervals(hours, tokens)
    assert dropped == 0
    assert np.allclose(rate, 4500.0)


def test_all_gaps_yields_no_rate_rather_than_a_wrong_one():
    hours = np.array([0.0, 50.0, 300.0])
    tokens = np.array([0.0, 1e6, 2e6])
    rate, rate_hours, dropped, _ = usable_intervals(hours, tokens)
    # Either everything survives (uniform spacing is its own cadence) or nothing
    # does; what must never happen is a silent partial answer with dropped == 0
    # while intervals were in fact discarded.
    assert rate.size == rate_hours.size
    assert dropped == 0 or rate.size == 0


def test_zero_length_intervals_do_not_divide_by_zero():
    """Two beats sharing a timestamp — seen when a relaunch replays a step."""
    hours = np.array([0.0, 1.0, 1.0, 2.0])
    tokens = np.array([0.0, 1.62e7, 1.62e7, 3.24e7])
    rate, _, _, _ = usable_intervals(hours, tokens)
    assert np.all(np.isfinite(rate))


def test_empty_and_single_beat_are_handled():
    for hours, tokens in [(np.array([0.0]), np.array([0.0])),
                          (np.array([]), np.array([]))]:
        rate, rate_hours, dropped, cadence = usable_intervals(hours, tokens)
        assert rate.size == 0 and rate_hours.size == 0
        assert dropped == 0 and cadence == 0.0
