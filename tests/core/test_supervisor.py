"""The unattended-run guards, and the race that made them worth writing down.

The #16 base run was supervised by two bash processes talking through marker
lines in a log file. The sentinel asked "the trainer is gone — did it crash, or
did the watchdog stop it?" by grepping a file the watchdog had not finished
writing, and in that window it read a *completed* run as a crash and relaunched
it. The fix in production was a longer sleep, which narrows the window without
closing it.

`trm/runtime/supervisor.py` closes it by making the decision a pure function
whose guards are evaluated before liveness. The load-bearing test in this file
is `test_a_finished_run_is_not_a_crash_even_though_the_process_is_gone` — it is
the original bug, stated as an assertion.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from trm.runtime.supervisor import (
    BUDGET_COMPLETE,
    CONTINUE,
    CRASHED,
    GAVE_UP,
    GIVE_UP,
    KILL,
    KILLED_DIVERGENCE,
    KILLED_DISK,
    KILLED_OOM,
    KILLED_PLATEAU,
    RELAUNCH,
    RESTART,
    RUNNING,
    STALLED,
    STOP,
    WALLCLOCK_COMPLETE,
    GpuLock,
    Limits,
    Observation,
    Preflight,
    State,
    check_disk_headroom,
    decide,
    DELIBERATE,
    plateau_in,
    read_progress,
)

LIMITS = Limits(stop_step=1000, max_ce=6.5, divergence_checks=2, max_retries=2)


def obs(**kw):
    base = dict(step=10, ce=3.0, plateau_detected=False, alive=True, elapsed_hours=0.0)
    return Observation(**{**base, **kw})


# --- the race the original got wrong ------------------------------------------

def test_a_finished_run_is_not_a_crash_even_though_the_process_is_gone():
    """THE regression test. The trainer reached its budget and exited; the
    supervisor looks a moment later and sees a dead process. The original
    sentinel called this a crash and relaunched a completed run."""
    decision = decide(obs(step=1000, alive=False), LIMITS, State())
    assert decision.outcome == BUDGET_COMPLETE
    assert decision.action == STOP


def test_a_deliberate_kill_is_not_a_crash_either():
    """Same shape, other guards: a plateau kill or a divergence kill must not
    come back as a relaunchable crash on the next look."""
    assert decide(obs(alive=False, plateau_detected=True), LIMITS, State()).outcome == KILLED_PLATEAU

    state = State(entered_band=True, divergence_streak=1)
    assert decide(obs(alive=False, ce=99.0), LIMITS, state).outcome == KILLED_DIVERGENCE


def test_a_genuine_crash_before_budget_still_relaunches():
    """The guard-first ordering must not have made every death look clean."""
    state = State()
    decision = decide(obs(step=400, alive=False), LIMITS, state)
    assert (decision.action, decision.outcome) == (RELAUNCH, CRASHED)
    assert state.retries_used == 1


# --- the decision table -------------------------------------------------------

def test_a_healthy_run_is_left_alone():
    decision = decide(obs(step=500, ce=3.2), LIMITS, State())
    assert (decision.action, decision.outcome) == (CONTINUE, RUNNING)


def test_plateau_outranks_everything():
    """A CE-plateau SFT flip silently contaminates a pretrain run, so it is worth
    killing an otherwise healthy — even a finished-looking — run."""
    decision = decide(obs(step=1000, plateau_detected=True), LIMITS, State())
    assert (decision.action, decision.outcome) == (KILL, KILLED_PLATEAU)


def test_divergence_needs_a_streak_not_a_blip():
    """One bad reading is noise; the guard exists for a warm restart that is
    actually blowing up, and killing a run on a single spike wastes it."""
    state = State(entered_band=True)
    assert decide(obs(ce=99.0), LIMITS, state).action == CONTINUE
    assert state.divergence_streak == 1
    assert decide(obs(ce=99.0), LIMITS, state).outcome == KILLED_DIVERGENCE


def test_a_recovered_ce_clears_the_streak():
    state = State(entered_band=True)
    decide(obs(ce=99.0), LIMITS, state)
    decide(obs(ce=3.0), LIMITS, state)
    assert state.divergence_streak == 0
    assert decide(obs(ce=99.0), LIMITS, state).action == CONTINUE, "the streak restarted"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_ce_is_divergence_from_the_very_first_step(bad):
    """NaN/inf is never a young run — it is a broken one, at any age. This one
    does not wait for the band to arm."""
    assert decide(obs(step=1, ce=bad), LIMITS, State()).action == CONTINUE
    state = State(divergence_streak=1)
    assert decide(obs(step=1, ce=bad), LIMITS, state).outcome == KILLED_DIVERGENCE


def test_an_out_of_band_ce_is_divergence_once_the_run_has_been_in_band():
    state = State(entered_band=True, divergence_streak=1)
    assert decide(obs(ce=7.0), LIMITS, state).outcome == KILLED_DIVERGENCE


# --- the band has to arm itself, or it kills every fresh run -------------------

def test_a_fresh_runs_opening_ce_is_not_divergence():
    """THE launch-blocker this guard shipped with. A from-scratch model starts at
    ln(50304) ~= 10.8 and the champion run did not get under 6.5 until opt step
    340, ~2.7 hours in — while the supervisor polls every 5 minutes. An
    always-armed band killed the run at ~10 minutes, every time, and only looked
    correct because it was written for a warm restart."""
    state = State()
    for step, ce in [(5, 11.08), (10, 10.4), (15, 9.9), (20, 9.7), (40, 8.6)]:
        assert decide(obs(step=step, ce=ce), LIMITS, state).action == CONTINUE
    assert state.divergence_streak == 0
    assert not state.entered_band


def test_the_band_arms_the_moment_the_run_gets_under_it():
    """Descending through the band is what proves the run can be judged by it;
    after that a climb back out is real divergence."""
    state = State()
    decide(obs(step=100, ce=8.0), LIMITS, state)
    decide(obs(step=340, ce=6.4), LIMITS, state)
    assert state.entered_band
    assert decide(obs(step=400, ce=9.0), LIMITS, state).action == CONTINUE
    assert decide(obs(step=405, ce=9.0), LIMITS, state).outcome == KILLED_DIVERGENCE


def test_a_resumed_run_is_armed_by_its_first_reading():
    """A resume reads a metrics.csv that already holds low CE, so the guard is
    live from the first poll — the case it was originally written for keeps its
    protection."""
    state = State()
    assert decide(obs(step=500, ce=3.1), LIMITS, state).action == CONTINUE
    assert state.entered_band


def test_a_missing_ce_is_not_divergence():
    """A run that has not logged a metric row yet is young, not broken —
    treating None as divergence would kill every run at startup."""
    state = State()
    assert decide(obs(ce=None), LIMITS, state).action == CONTINUE
    assert state.divergence_streak == 0


def test_retries_are_finite():
    state = State()
    for expected in (RELAUNCH, RELAUNCH, GIVE_UP):
        assert decide(obs(step=1, alive=False), LIMITS, state).action == expected
    assert decide(obs(step=1, alive=False), LIMITS, state).outcome == GAVE_UP


def test_the_wallclock_cap_stops_a_run_that_is_making_progress():
    limits = Limits(stop_step=10_000, max_hours=8.0)
    decision = decide(obs(step=500, elapsed_hours=8.1), limits, State())
    assert (decision.action, decision.outcome) == (STOP, WALLCLOCK_COMPLETE)


def test_the_budget_stop_outranks_the_wallclock_cap():
    """Both firing at once means the run finished; report the reason that is
    about the science, not the one about the clock."""
    limits = Limits(stop_step=1000, max_hours=1.0)
    assert decide(obs(step=1000, elapsed_hours=99.0), limits, State()).outcome == BUDGET_COMPLETE


def test_a_wedged_run_is_restarted_even_though_it_is_alive():
    """The motivating failure: a detached job stalled three days unnoticed,
    because a process that is alive and doing nothing looks exactly like a
    process that is working."""
    limits = Limits(stop_step=10_000, stall_polls=3, max_retries=1)
    state = State()
    for _ in range(3):
        assert decide(obs(step=42), limits, state).action == CONTINUE
    decision = decide(obs(step=42), limits, state)
    assert (decision.action, decision.outcome) == (RESTART, STALLED)


def test_progress_clears_the_stall_counter():
    limits = Limits(stop_step=10_000, stall_polls=2)
    state = State()
    decide(obs(step=42), limits, state)
    decide(obs(step=42), limits, state)
    decide(obs(step=43), limits, state)  # it moved
    assert state.stalled_polls == 0
    assert decide(obs(step=43), limits, state).action == CONTINUE


def test_a_stall_does_not_outrank_a_finished_run():
    """A run sitting at its stop step is done, not wedged. Restarting it would
    be the sentinel bug wearing a different hat."""
    limits = Limits(stop_step=100, stall_polls=1)
    state = State()
    decide(obs(step=100), limits, state)
    assert decide(obs(step=100), limits, state).outcome == BUDGET_COMPLETE


def test_a_dead_process_is_not_counted_as_stalled():
    """Otherwise a crash would burn a retry as a stall and then another as a
    crash, halving the relaunch budget."""
    limits = Limits(stop_step=10_000, stall_polls=2, max_retries=5)
    state = State()
    decide(obs(step=7, alive=False), limits, state)
    assert state.stalled_polls == 0


def test_stall_restarts_are_finite():
    limits = Limits(stop_step=10_000, stall_polls=1, max_retries=1)
    state = State()
    decide(obs(step=5), limits, state)
    assert decide(obs(step=5), limits, state).action == RESTART
    decide(obs(step=5), limits, state)
    assert decide(obs(step=5), limits, state).outcome == GAVE_UP


# --- reading the run ----------------------------------------------------------

def test_read_progress_takes_the_last_complete_row(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("step,ce,loss\n10,3.5,3.6\n20,3.1,3.2\n")
    assert read_progress(csv_path) == (20, 3.1)


def test_a_torn_final_line_falls_back_to_the_last_good_row(tmp_path):
    """The trainer appends to this file while the supervisor reads it. A partial
    row is normal; reading it as 'no progress' would look like a stalled run."""
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("step,ce,loss\n10,3.5,3.6\n20,3.1,3.2\n30,")
    assert read_progress(csv_path) == (30, None)

    csv_path.write_text("step,ce,loss\n10,3.5,3.6\n,,\n")
    assert read_progress(csv_path) == (10, 3.5)


def test_a_missing_or_empty_metrics_file_reads_as_no_progress(tmp_path):
    assert read_progress(tmp_path / "nope.csv") == (0, None)
    (tmp_path / "empty.csv").write_text("step,ce\n")
    assert read_progress(tmp_path / "empty.csv") == (0, None)


def test_plateau_detection_matches_the_trainers_own_wording(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Step 0100 | CE: 3.5\n")
    assert not plateau_in(log)
    log.write_text("Step 0100 | CE: 3.5\n🔄 CE Plateau Detected — switching phase\n")
    assert plateau_in(log)
    assert not plateau_in(tmp_path / "absent.log")


# --- preflight ----------------------------------------------------------------

def test_disk_headroom_refuses_a_nearly_full_disk(tmp_path):
    """A long run that fills the disk dies with a corrupt final checkpoint — it
    costs the compute AND the artifact, which is the worst trade available."""
    with pytest.raises(Preflight, match="free"):
        check_disk_headroom(tmp_path, min_free_gb=1e9)
    assert check_disk_headroom(tmp_path, min_free_gb=0.0) > 0


def test_the_gpu_lock_is_a_serial_queue(tmp_path):
    first = GpuLock(tmp_path / "gpu.lock", label="run-a")
    first.acquire()
    with pytest.raises(Preflight, match="one run at a time"):
        GpuLock(tmp_path / "gpu.lock", label="run-b").acquire()
    first.release()
    GpuLock(tmp_path / "gpu.lock", label="run-b").acquire()  # now free


def test_a_stale_lock_is_taken_over_not_respected(tmp_path):
    """A lock nobody can clear turns one crash into a card that stays idle until
    a human notices — worse than no lock at all."""
    path = tmp_path / "gpu.lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    path.write_text(f"{dead.pid} long-gone\n")

    lock = GpuLock(path, label="new-run")
    lock.acquire()
    assert lock.holder()[0] == os.getpid()


def test_an_unreadable_lock_is_stale(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("garbage written by a half-finished process\n")
    GpuLock(path).acquire()


def test_releasing_never_removes_someone_elses_lock(tmp_path):
    """A supervisor that took over a stale lock must not delete whatever
    replaced it — that would hand the card to two runs at once."""
    path = tmp_path / "gpu.lock"
    mine = GpuLock(path, label="mine")
    mine.acquire()
    path.write_text("999999 someone-else\n")
    mine.release()
    assert path.exists(), "we only ever remove our own lock"


def test_the_lock_releases_on_the_way_out_of_a_with_block(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError):
        with GpuLock(path, label="x"):
            raise RuntimeError("the run blew up")
    assert not path.exists()


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


def test_the_supervisor_kills_a_child_that_trips_the_plateau_guard(tmp_path):
    log = tmp_path / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    sup, _ = _supervisor_over(tmp_path, """
        import time
        print("CE Plateau Detected", flush=True)
        time.sleep(60)
    """, Limits(stop_step=10_000, max_retries=0))

    assert sup.run() == KILLED_PLATEAU


# --- an OOM is not a crash (2026-08-13, 2026-08-15) ----------------------------

def test_an_oom_is_terminal_not_relaunchable():
    """Three launches were lost to this in one day. An OOM is deterministic: the
    same config allocating the same tensors on the same card fails identically, so
    relaunching spends the retry budget and ~15 minutes to reach the same place —
    and then reports GAVE_UP, which reads like flakiness and sends the next reader
    hunting for a race instead of looking at memory."""
    state = State()
    decision = decide(obs(step=12, alive=False, oom_detected=True), LIMITS, state)
    assert decision.outcome == KILLED_OOM
    assert decision.action != RELAUNCH
    assert state.retries_used == 0, "an OOM must not spend a retry"


def test_an_oom_outcome_is_deliberate_not_a_crash():
    """So the next poll does not pick it back up as something to relaunch."""
    from trm.runtime.supervisor import DELIBERATE
    assert KILLED_OOM in DELIBERATE


def test_a_crash_without_an_oom_still_relaunches():
    """The classification must not have swallowed ordinary crashes."""
    state = State()
    decision = decide(obs(step=400, alive=False, oom_detected=False), LIMITS, state)
    assert (decision.action, decision.outcome) == (RELAUNCH, CRASHED)


def test_an_oom_while_still_alive_is_not_terminal():
    """A logged allocator warning that the run recovered from is not a death. Only
    a dead process plus an OOM in its output ends the run."""
    assert decide(obs(step=400, alive=True, oom_detected=True), LIMITS, State()).action == CONTINUE


# --- the log is append-mode: only THIS launch's output is evidence -------------

def test_a_previous_sessions_markers_are_not_read_as_this_run(tmp_path):
    """The log is opened in append mode, so a resumed run writes after whatever the
    last session left. Reading from byte 0, a plateau or an OOM from hours ago is
    still 'detected' — and the supervisor kills a healthy run on the strength of a
    dead session's output. This bit nothing yet only because no run had resumed
    after a failure that printed one."""
    from trm.runtime.supervisor import oom_in

    log = tmp_path / "train.log"
    log.write_text("...RESOURCE_EXHAUSTED: Out of memory\n🔄 CE Plateau Detected\n")
    stale = log.stat().st_size

    assert oom_in(log) and plateau_in(log), "reading from 0 sees the old session"
    assert not oom_in(log, stale), "reading from this launch's offset does not"
    assert not plateau_in(log, stale)

    with log.open("a") as f:
        f.write("Step 0005 | CE: 3.5\n")
    assert not oom_in(log, stale), "healthy output after the offset stays clean"

    with log.open("a") as f:
        f.write("CUDA_ERROR_OUT_OF_MEMORY: out of memory\n")
    assert oom_in(log, stale), "a fresh OOM after the offset is still caught"


@pytest.mark.parametrize("marker", [
    "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 596.39MiB",
    "cuMemAllocAsync failed: CUDA_ERROR_OUT_OF_MEMORY: out of memory",
    "Allocator (GPU_0_bfc) ran out of memory trying to allocate 720.0KiB",
])
def test_every_allocator_spells_oom_differently(tmp_path, marker):
    """BFC, cuda_async and the command-buffer path each word it their own way, and
    this project has now seen all three."""
    from trm.runtime.supervisor import oom_in
    log = tmp_path / "train.log"
    log.write_text(marker + "\n")
    assert oom_in(log)


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


# --- every guard, in every regime (#176) ----------------------------------------
#
# The divergence guard killed every fresh run because it was written for a warm
# restart and only ever tested mid-flight. Each guard carries an assumption about
# which regime the run is in; these tables state it for all of them.
#
#   cold      a fresh model: CE opens at ~10.8, metrics absent through a long compile
#   mid       in band, steps advancing
#   resumed   a new supervisor on a checkpointed run: first reading already low and late
#   relaunch  the same supervisor after a crash: the step reads *behind* the dead
#             launch while the trainer replays from its checkpoint

def _play(polls, limits=None, state=None):
    """Feed (step, ce, alive) polls to decide(); return the decisions."""
    limits = limits or Limits(stop_step=30_000)
    state = state or State()
    return [decide(Observation(step=s, ce=ce, plateau_detected=False, alive=alive), limits, state)
            for s, ce, alive in polls], state


COMPILE = [(0, None, True)] * 11                      # 11 silent polls: under stall_polls
COLD = COMPILE + [(5, 10.8), (10, 10.5), (340, 6.4)]
MID = [(5000, 3.5), (5005, 3.4), (5010, 3.5)]
RESUMED = [(4992, 3.4), (4992, 3.4), (4997, 3.4)]


def _alive(seq):
    return [p if len(p) == 3 else (*p, True) for p in seq]


@pytest.mark.parametrize("regime", [COLD, MID, RESUMED], ids=["cold", "mid", "resumed"])
def test_a_healthy_run_is_left_alone_in_every_regime(regime):
    decisions, _ = _play(_alive(regime))
    assert all(d.action == CONTINUE for d in decisions), [d.reason for d in decisions]


@pytest.mark.parametrize("regime", [COLD, MID, RESUMED], ids=["cold", "mid", "resumed"])
def test_non_finite_ce_kills_in_every_regime(regime):
    decisions, _ = _play(_alive(regime + [(99_990, float("nan")), (99_991, float("nan"))]),
                         Limits(stop_step=10**6))
    assert decisions[-1].outcome == KILLED_DIVERGENCE


def test_high_ce_is_divergence_only_once_the_run_has_been_in_band():
    cold, _ = _play(_alive(COMPILE + [(5, 10.8), (6, 10.7), (7, 10.6)]))
    assert all(d.action == CONTINUE for d in cold), "cold: opening CE is not divergence"
    for regime in (MID, RESUMED):
        warm, _ = _play(_alive(regime + [(6000, 9.0), (6001, 9.0)]))
        assert warm[-1].outcome == KILLED_DIVERGENCE


@pytest.mark.parametrize("regime", [COLD, MID, RESUMED], ids=["cold", "mid", "resumed"])
def test_a_wedge_is_caught_in_every_regime(regime):
    last = _alive(regime)[-1]
    decisions, _ = _play(_alive(regime) + [last] * Limits.stall_polls)
    assert decisions[-1].outcome == STALLED


def test_a_relaunch_replaying_from_its_checkpoint_is_not_a_stall():
    """The bug this table found. The dead launch reached 5055; the relaunch resumes
    at checkpoint 4992, metrics.csv is trimmed back to it, and the replay takes the
    stall budget and then some. It was judged against 5055 and read as wedged."""
    limits = Limits(stop_step=30_000, max_retries=2)
    before, state = _play(_alive([(5050, 3.4), (5055, 3.4), (5055, 3.4, False)]), limits)
    assert before[-1].action == RELAUNCH
    replay = [(4991, 3.4)] * 3 + [(4991 + i, 3.4) for i in range(5, 64, 5)]
    after, _ = _play(_alive(replay), limits, state)
    assert all(d.action == CONTINUE for d in after), [d.reason for d in after if d.action != CONTINUE]


def test_a_stall_restart_judges_the_new_launch_from_its_own_start():
    limits = Limits(stop_step=30_000, max_retries=2)
    wedged, state = _play(_alive([(5055, 3.4)] * (Limits.stall_polls + 1)), limits)
    assert wedged[-1].outcome == STALLED
    after, _ = _play(_alive([(4991, 3.4)] * 4 + [(4991 + i, 3.4) for i in range(5, 64, 5)]), limits, state)
    assert all(d.action == CONTINUE for d in after)


def test_what_a_relaunch_does_not_reset():
    """The retry budget spans launches, or a crash loop never ends; and a run that
    has been in band stays armed, since its resumed CE is already low."""
    _, state = _play(_alive([(5000, 3.4), (5000, 3.4, False)]), Limits(stop_step=30_000, max_retries=2))
    assert state.retries_used == 1 and state.entered_band


@pytest.mark.parametrize("regime", [COLD, MID, RESUMED], ids=["cold", "mid", "resumed"])
def test_the_budget_stop_holds_in_every_regime(regime):
    decisions, _ = _play(_alive(regime), Limits(stop_step=_alive(regime)[-1][0]))
    assert decisions[-1].outcome == BUDGET_COMPLETE


# --- the disk is checked while the run consumes it, not once at launch (#190) ---

def _disk(free_gb, checkpoint_gb, alive=True, step=5000):
    return Observation(step=step, ce=3.4, plateau_detected=False, alive=alive,
                       free_gb=free_gb, checkpoint_gb=checkpoint_gb)


def test_a_run_that_would_not_fit_its_next_checkpoint_stops_cleanly():
    """The next write lands in the rolling and best dirs at once; stop before it,
    with a TERM the trainer survives, not after a torn write."""
    d = decide(_disk(free_gb=4.0, checkpoint_gb=1.7), Limits(stop_step=30_000), State())
    assert (d.action, d.outcome) == (STOP, KILLED_DISK)
    assert KILLED_DISK in DELIBERATE, "a disk stop is not a crash to relaunch"


def test_the_157_steady_state_is_healthy_not_a_refusal():
    """19GB free with 1.7GB checkpoints: below the 20GB launch floor, which is what
    would have refused the run's own relaunch. In-run the requirement is the next
    write, and 19GB clears it easily."""
    limits = Limits(stop_step=30_000)
    assert decide(_disk(free_gb=19.0, checkpoint_gb=1.7), limits, State()).action == CONTINUE


def test_a_dead_run_on_a_full_disk_is_not_relaunched_into_a_torn_write():
    d = decide(_disk(free_gb=3.0, checkpoint_gb=1.7, alive=False), Limits(stop_step=30_000), State())
    assert d.outcome == KILLED_DISK


def test_before_the_first_checkpoint_the_launch_floor_is_all_there_is():
    """No checkpoint yet means no known requirement; the guard stays out of it
    rather than guess, and the launch precheck already covered this moment."""
    assert decide(_disk(free_gb=0.5, checkpoint_gb=None), Limits(stop_step=30_000), State()).action == CONTINUE


def test_the_budget_stop_outranks_the_disk_guard():
    d = decide(_disk(free_gb=1.0, checkpoint_gb=1.7, step=30_000), Limits(stop_step=30_000), State())
    assert d.outcome == BUDGET_COMPLETE


def test_largest_checkpoint_counts_rolling_and_best_and_ignores_torn_writes(tmp_path):
    from trm.runtime.supervisor import largest_checkpoint_gb

    assert largest_checkpoint_gb(tmp_path / "nope") is None
    for rel, size, finalized in (("100", 1000, True), ("best_val_ce/100", 3000, True), ("200", 9000, False)):
        step = tmp_path / rel
        step.mkdir(parents=True)
        (step / "blob").write_bytes(b"x" * size)
        if finalized:
            (step / "_CHECKPOINT_METADATA").write_text("{}")
    assert largest_checkpoint_gb(tmp_path) == (3000 + 2) / 1e9


# --- the fit gate: the real trainer, briefly, before committing the card (#168) ---

def _gate(tmp_path, body, **kw):
    from trm.runtime.supervisor import preflight_fit
    script = tmp_path / "fake_trainer.py"
    script.write_text(textwrap.dedent(body))
    return preflight_fit(command=[sys.executable, str(script)], workroot=tmp_path / "runs",
                         poll_s=0.1, tokens_per_opt_step=1000, **kw)


def test_the_gate_passes_on_a_first_logged_row_and_leaves_nothing_behind(tmp_path):
    (tmp_path / "runs").mkdir()
    result = _gate(tmp_path, """
        import os, pathlib, time
        assert os.environ["VAL_EVERY_OPT_STEPS"] == "1" and os.environ["CHECKPOINT_EVERY_OPT_STEPS"] == "1"
        print("Step 0005 | CE: 9.1 | Compute: 2.000s", flush=True)
        p = pathlib.Path("runs/run_fitgate/metrics.csv")
        p.write_text("step,ce,arena_peak_mib\\n5,9.1,4112\\n")
        time.sleep(60)
    """)
    assert result.ok and result.arena_peak_mib == 4112.0
    assert result.tokens_per_second == 5 * 1000 / 2.0
    assert list((tmp_path / "runs").iterdir()) == [], "the probe's directory must be removed"


def test_the_gate_refuses_an_out_of_memory(tmp_path):
    (tmp_path / "runs").mkdir()
    result = _gate(tmp_path, """
        import time
        print("2026-08-13 E external/xla: RESOURCE_EXHAUSTED: Out of memory allocating 626MiB", flush=True)
        time.sleep(60)
    """)
    assert not result.ok and "does not fit" in result.reason
    assert list((tmp_path / "runs").iterdir()) == []


def test_the_gate_refuses_a_trainer_that_dies_or_never_logs(tmp_path):
    (tmp_path / "runs").mkdir()
    died = _gate(tmp_path, """
        import sys
        print("Traceback: data root missing", flush=True)
        sys.exit(3)
    """)
    assert not died.ok and "exited (3)" in died.reason and "data root missing" in died.reason
    hung = _gate(tmp_path, "import time\ntime.sleep(60)\n", timeout_s=1.0)
    assert not hung.ok and "no logged row" in hung.reason
    assert list((tmp_path / "runs").iterdir()) == []


def test_the_gate_never_points_the_probe_at_the_real_run():
    from trm.runtime.supervisor import _without_path_flags
    assert _without_path_flags(["--checkpoint-path", "runs/run_real/checkpoints", "--new-run", "--x", "1"]) == ["--x", "1"]
    assert _without_path_flags(["--checkpoint-path=runs/run_real/checkpoints"]) == []


def test_a_refused_gate_stops_the_launch_and_releases_the_card(tmp_path, monkeypatch):
    from trm.runtime import supervisor as sup_mod

    launched = []

    class Stub:
        def __init__(self, **kw):
            pass

        def run(self):
            launched.append(True)
            return sup_mod.BUDGET_COMPLETE

    lock_path = tmp_path / "gpu.lock"
    monkeypatch.setattr(sup_mod, "Supervisor", Stub)
    monkeypatch.setattr(sup_mod, "GpuLock", lambda label="": GpuLock(lock_path, label))
    monkeypatch.setattr(sup_mod, "preflight_fit",
                        lambda args: sup_mod.FitResult(False, "out of memory — this config does not fit"))
    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log")])
    assert code == 1 and launched == []
    assert not lock_path.exists(), "a refused launch must not keep holding the card"

    monkeypatch.setattr(sup_mod, "preflight_fit", lambda args: (_ for _ in ()).throw(AssertionError("gate ran")))
    assert sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
                         "--skip-fit-gate"]) == 0
    assert launched == [True]


def test_a_killed_gates_leftovers_are_swept_by_the_next_gate(tmp_path):
    runs = tmp_path / "runs"
    (runs / ".fitgate_20260101_000000" / "runs").mkdir(parents=True)
    (runs / "run_real").mkdir()
    _gate(tmp_path, "import sys\nsys.exit(1)\n")
    assert sorted(p.name for p in runs.iterdir()) == ["run_real"], "sweep probe dirs, never a real run"
