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
