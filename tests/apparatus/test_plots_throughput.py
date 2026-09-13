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


# --- measured from the run's own clock when it has one (#186, #177) ------------

def _clocked_run(tmp_path, rows, header="step,ce,wall_clock,mix"):
    import datetime
    from instruments import runlog

    run_dir = tmp_path / "run_20990101_000000"
    run_dir.mkdir()
    start = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)
    lines = [header] + [f"{step},3.0,{(start + datetime.timedelta(seconds=sec)):%Y-%m-%dT%H:%M:%SZ},"
                        f"fineweb-edu=0.850 codeparrot=0.100 finemath=0.050" for step, sec in rows]
    (run_dir / "metrics.csv").write_text("\n".join(lines) + "\n")
    return runlog.load(str(run_dir / "metrics.csv"))


def test_wall_clock_and_mix_are_read_as_what_they_are(tmp_path):
    import datetime
    log = _clocked_run(tmp_path, [(5, 0), (10, 150)])
    assert log.metrics[1]["wall_clock"] == datetime.datetime(2099, 1, 1, 0, 2, 30, tzinfo=datetime.timezone.utc)
    assert log.metrics[0]["mix"].startswith("fineweb-edu=0.850")


def test_throughput_uses_logged_rows_thinned_to_end_to_end_intervals(tmp_path):
    """A row every ~2.5 minutes, thinned to >=30-minute intervals, so a point is a
    rate across checkpoints and probes rather than five opt steps of noise."""
    from instruments.plots import clock_samples
    rows = [(5 * i, 150 * i) for i in range(1, 100)]
    samples, source = clock_samples(_clocked_run(tmp_path, rows))
    assert source == "metrics"
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(samples, samples[1:])]
    assert all(g >= 1800 for g in gaps[:-1]), "every interval but the tail is end-to-end"
    assert samples[0][1] == 5 and samples[-1][1] == rows[-1][0], "first and last rows always kept"


def test_a_run_without_wall_clock_falls_back_to_heartbeats(tmp_path):
    from instruments import runlog
    from instruments.plots import clock_samples
    run_dir = tmp_path / "run_20260101_000000"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("step,ce\n5,3.0\n10,2.9\n")
    _, source = clock_samples(runlog.load(str(run_dir / "metrics.csv")))
    assert source == "heartbeats"


def test_the_figure_says_measured_when_it_is(tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    from instruments import plots
    rows = [(5 * i, 150 * i) for i in range(1, 200)]
    result = plots.throughput_progress(_clocked_run(tmp_path, rows), tmp_path)
    assert result is not None and result["coverage"] == 1.0
    assert "heartbeat" not in capsys.readouterr().out
