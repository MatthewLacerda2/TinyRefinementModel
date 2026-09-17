"""The supervisor loop against a real child process, and its heartbeat: the issue
feed, its cadence, and the heartbeat file that survives relaunch. Split out of
test_supervisor.py (#325)."""

import sys
import textwrap


from trm.runtime.supervisor import (
    BUDGET_COMPLETE,
    GAVE_UP,
    Limits,
)


# --- the loop, against a real child process -----------------------------------

def _supervisor_over(tmp_path, script: str, limits: Limits, **kw):
    from trm.runtime.supervisor import Supervisor

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(script))
    reported = []
    sup = Supervisor(
        command=(sys.executable, str(child)),
        limits=limits,
        log_path=tmp_path / "train.log",
        metrics_csv=tmp_path / "metrics.csv",
        poll_seconds=0.2,
        heartbeat_every=1,
        report=reported.append,
        **kw,
    )
    return sup, reported


def test_the_supervisor_stops_a_child_that_reaches_the_budget(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    sup, reported = _supervisor_over(tmp_path, f"""
        import time, pathlib
        p = pathlib.Path({str(csv_path)!r})
        p.write_text("step,ce\\n")
        for step in range(0, 200, 10):
            p.write_text("step,ce\\n" + f"{{step}},3.0\\n")
            time.sleep(0.1)
        time.sleep(60)
    """, Limits(stop_step=50, max_retries=0))

    assert sup.run() == BUDGET_COMPLETE
    assert any("BUDGET_COMPLETE" in line for line in reported)


def test_the_supervisor_relaunches_a_crashing_child_then_gives_up(tmp_path):
    sup, reported = _supervisor_over(tmp_path, """
        import sys
        sys.exit(1)
    """, Limits(stop_step=10_000, max_retries=1))

    assert sup.run() == GAVE_UP
    assert sum("relaunched" in line for line in reported) == 1


def test_the_heartbeat_survives_github_being_unreachable(tmp_path, monkeypatch, capsys):
    """A supervisor that died because the network blipped would abandon a
    running job over something that has nothing to do with the run."""
    from trm.runtime import supervisor as sup_mod

    def explode(*a, **k):
        raise OSError("gh: command not found")

    monkeypatch.setattr(sup_mod.subprocess, "run", explode)
    sup_mod.github_reporter(16)("still going")
    out = capsys.readouterr().out
    assert "still going" in out, "the message still reaches the local log"
    assert "heartbeat to #16 failed" in out


# --- the heartbeat is for humans, and humans stop reading -----------------------

def test_routine_reports_are_daily_not_hourly():
    """At hourly cadence a nine-day run posts ~270 identical RUNNING lines to its
    tracking issue — #157 had 57 in two days — burying the launch decision, the
    findings and the diagnoses the issue exists to record. A status feed nobody
    can read is not a status feed."""
    from trm.runtime.supervisor import Supervisor
    import pathlib
    sup = Supervisor(command=(), limits=Limits(stop_step=1),
                     log_path=pathlib.Path("x"), metrics_csv=pathlib.Path("y"))
    assert sup.heartbeat_every * sup.poll_seconds / 3600 == 24.0


def test_the_heartbeat_cadence_is_settable_in_hours(tmp_path):
    """Hours, not poll counts — the cadence a person cares about should not require
    dividing by a polling interval they did not choose."""
    from trm.runtime import supervisor as sup_mod

    captured = {}

    class Stub:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self):
            return sup_mod.BUDGET_COMPLETE

    original = sup_mod.Supervisor
    sup_mod.Supervisor = Stub
    try:
        sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path),
                      "--log", str(tmp_path / "t.log"), "--no-gpu-lock", "--skip-fit-gate",
                      "--heartbeat-hours", "6", "--poll-seconds", "300"])
    finally:
        sup_mod.Supervisor = original

    assert captured["heartbeat_every"] == 72, "6h at a 300s poll is 72 polls"


def test_a_decision_always_reports_regardless_of_cadence():
    """Quieting the routine case must not quiet the events that matter. Every
    non-CONTINUE decision posts immediately — that is what makes a daily heartbeat
    safe rather than negligent."""
    from pathlib import Path
    from trm.runtime import supervisor as sup_mod
    source = Path(sup_mod.__file__).read_text()
    assert "decision.action != CONTINUE" in source, (
        "decisions must bypass the heartbeat interval"
    )


# --- the heartbeat file survives relaunch (#224) --------------------------------

def test_the_heartbeat_file_keeps_appending_across_relaunches(tmp_path):
    """run_20260813_214725 lost six days of heartbeats because they went to whatever
    stdout the relaunching shell provided. The supervisor now owns the file: a
    crash-relaunch inside one supervisor, and a whole new supervisor process on the
    same run, both append to it."""
    beats = tmp_path / "run_x.supervisor.log"
    crash = """
        import sys
        sys.exit(1)
    """
    sup, _ = _supervisor_over(tmp_path, crash, Limits(stop_step=10_000, max_retries=1),
                              heartbeat_log=beats)
    assert sup.run() == GAVE_UP
    first = beats.read_text()
    assert "supervising pid" in first and "relaunched as pid" in first and "GAVE_UP" in first

    again, _ = _supervisor_over(tmp_path, crash, Limits(stop_step=10_000, max_retries=0),
                                heartbeat_log=beats)
    again.run()
    text = beats.read_text()
    assert text.startswith(first), "a new supervisor must append, never truncate"
    assert text.count("supervising pid") == 2


def test_routine_beats_reach_the_file_hourly_without_flooding_the_issue(tmp_path):
    """The issue feed stays daily (it buries decisions otherwise); the file is what
    the throughput plot samples, so it gets the finer cadence — in the exact
    format the plot's reader parses."""
    import re
    from instruments.plots import _HEARTBEAT

    beats = tmp_path / "run_x.supervisor.log"
    csv_path = tmp_path / "metrics.csv"
    sup, reported = _supervisor_over(tmp_path, f"""
        import time, pathlib
        p = pathlib.Path({str(csv_path)!r})
        for step in range(0, 60, 2):
            p.write_text("step,ce\\n" + f"{{step}},3.0\\n")
            time.sleep(0.1)
        time.sleep(60)
    """, Limits(stop_step=50, max_retries=0), heartbeat_log=beats)
    sup.heartbeat_every, sup.log_every = 10_000, 1

    assert sup.run() == BUDGET_COMPLETE
    routine = [ln for ln in beats.read_text().splitlines() if "RUNNING" in ln]
    assert routine, "routine beats must reach the file"
    assert all(_HEARTBEAT.match(ln) for ln in routine), routine[:2]
    assert not any("RUNNING" in ln for ln in reported), "the issue feed stays quiet"
    assert any(re.search("BUDGET_COMPLETE", ln) for ln in reported), "decisions still report"


def test_main_defaults_the_heartbeat_file_to_where_the_plot_reads_it(tmp_path):
    from trm.runtime import supervisor as sup_mod

    captured = {}

    class Stub:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self):
            return sup_mod.BUDGET_COMPLETE

    run_dir = tmp_path / "run_20990101_000000"
    original = sup_mod.Supervisor
    sup_mod.Supervisor = Stub
    try:
        sup_mod.main(["--stop-step", "10", "--run-dir", str(run_dir),
                      "--log", str(tmp_path / "t.log"), "--no-gpu-lock", "--skip-fit-gate"])
    finally:
        sup_mod.Supervisor = original

    assert captured["heartbeat_log"] == tmp_path / "run_20990101_000000.supervisor.log"
