"""Supervisor preflight: the disk-headroom refusal and the GPU lock that makes the
card a serial queue. Split out of test_supervisor.py (#325)."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from trm.runtime import supervisor as sup_mod
from trm.runtime.supervisor import (
    REFUSED_RESUME,
    GpuLock,
    Limits,
    Preflight,
    Supervisor,
    check_disk_headroom,
    resumed_checkpoint_dir,
)
from trm.settings import Config

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
    with pytest.raises(RuntimeError), GpuLock(path, label="x"):
        raise RuntimeError("the run blew up")
    assert not path.exists()


def test_the_lock_names_one_place_on_the_box_not_one_per_worktree(tmp_path):
    """Almost every change here is built in a worktree off origin/main. A lock that
    resolved to the worktree's own `runs/` would queue against itself and see
    nothing of the run actually holding the card (#445)."""
    from trm.runtime.gpu_lock import shared_runs_dir

    main = tmp_path / "checkout"
    (main / ".git" / "worktrees" / "feature").mkdir(parents=True)
    worktree = tmp_path / "feature"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'feature'}\n")

    assert shared_runs_dir(worktree) == main / "runs"
    # An ordinary checkout, and anything unreadable, keep the old behaviour.
    (main / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert shared_runs_dir(main) == main / "runs"
    odd = tmp_path / "odd"
    (odd).mkdir()
    (odd / ".git").write_text("not a gitdir line\n")
    assert shared_runs_dir(odd) == odd / "runs"


# --- a checkpoint the trainer would refuse to resume (#505) ---------------------

# A resume state ResumeState refuses: its required fields, and one key misspelled.
MISSPELLED = {"run_id": "run_X", "ce_history": [3.4], "best_ce": 3.1, "best_loss": 3.2,
              "best_avg_ce": 3.4, "last_improvement_step": 100, "sampels_seen": 5}


def _write_checkpoint(checkpoints, step, state):
    path = checkpoints / str(step)
    (path / "monitor_state").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("{}")
    (path / "monitor_state" / "metadata").write_text(json.dumps(state))


def test_the_supervisor_refuses_an_unresumable_checkpoint_once_before_launching(tmp_path, monkeypatch):
    """The pass bar of #505: refused once, the field named, nothing launched and no
    session appended — where it used to crash-loop to GAVE_UP."""
    run_dir = tmp_path / "run_20990101_000000"
    metadata = run_dir / "run_metadata.json"
    run_dir.mkdir()
    metadata.write_text('{"sessions": [{"start": "2099-01-01"}]}\n')
    before = metadata.read_bytes()
    _write_checkpoint(run_dir / "checkpoints", 1279, MISSPELLED)

    def no_launch(*a, **k):
        raise AssertionError("nothing may be launched onto a refused checkpoint")
    monkeypatch.setattr(sup_mod.subprocess, "Popen", no_launch)
    monkeypatch.setattr(sup_mod, "preflight_fit", no_launch)

    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(run_dir), "--log", str(tmp_path / "t.log"),
                         "--min-free-gb", "0", "--no-gpu-lock", "--", "--checkpoint-path", str(run_dir / "checkpoints")])

    assert code == 1
    heartbeat = (tmp_path / "run_20990101_000000.supervisor.log").read_text()
    assert heartbeat.count(f"{REFUSED_RESUME}:") == 1 and "sampels_seen" in heartbeat
    assert metadata.read_bytes() == before


def test_a_relaunch_onto_an_unresumable_checkpoint_stops_instead_of_retrying(tmp_path):
    """The checkpoint went bad after the launch passed its preflight: the dead child
    is not relaunched into the same refusal, twice, and then GAVE_UP."""
    checkpoints = tmp_path / "checkpoints"
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import json, pathlib, sys
        path = pathlib.Path({str(checkpoints)!r}) / "1279"
        (path / "monitor_state").mkdir(parents=True)
        (path / "_CHECKPOINT_METADATA").write_text("{{}}")
        (path / "monitor_state" / "metadata").write_text(json.dumps({MISSPELLED!r}))
        sys.exit(1)
    """))
    reported = []
    sup = Supervisor(command=(sys.executable, str(child)), limits=Limits(stop_step=10_000, max_retries=2),
                     log_path=tmp_path / "t.log", metrics_csv=tmp_path / "metrics.csv",
                     config=Config.from_env({}), poll_seconds=0.2, report=reported.append,
                     checkpoint_dir=checkpoints)

    assert sup.run() == REFUSED_RESUME
    assert not any("relaunched" in line for line in reported)


def test_the_checkpoint_dir_a_relaunch_resumes_is_read_from_the_trainer_args(tmp_path):
    run = tmp_path / "run_X"
    absolute = tmp_path / "ck"
    assert resumed_checkpoint_dir(["--checkpoint-path", str(absolute)], run) == absolute
    assert resumed_checkpoint_dir(["--checkpoint-path=runs/r/ck"], run) == sup_mod.REPO_ROOT / "runs/r/ck"
    assert resumed_checkpoint_dir([], run) == run / "checkpoints"
    assert resumed_checkpoint_dir(["--new-run", "--checkpoint-path", str(absolute)], run) is None


# --- the lock is never released under a live trainer (#512) ---------------------

def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_giving_up_on_a_wedged_trainer_stops_it_first(tmp_path):
    """The pass bar of #512: a child alive but stuck, with no relaunch left, is
    GAVE_UP, and it is gone by then — even one that ignores SIGTERM."""
    metrics = tmp_path / "metrics.csv"
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import pathlib, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(metrics)!r}).write_text("step,ce\\n5,3.0\\n")
        time.sleep(600)
    """))
    reported = []
    sup = Supervisor(command=(sys.executable, str(child)), limits=Limits(stop_step=10_000, stall_polls=1, max_retries=0),
                     log_path=tmp_path / "t.log", metrics_csv=metrics, config=Config.from_env({}),
                     poll_seconds=0.2, report=reported.append, stop_grace_seconds=0.5)
    pids = []
    launch = sup.launch

    def launch_and_note_pid():
        proc = launch()
        pids.append(proc.pid)
        return proc
    sup.launch = launch_and_note_pid

    assert sup.run() == sup_mod.GAVE_UP
    assert len(pids) == 1 and _gone(pids[0]), "the wedged trainer must not outlive the supervisor's verdict"
    assert sup.survivor is None


class _Unkillable:
    """A child stuck in the driver: SIGTERM and SIGKILL both land, neither ends it."""
    pid = 424242

    def poll(self):
        return None

    def send_signal(self, sig):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("trainer", timeout)


def _unkillable_supervisor(tmp_path, **limits):
    reported = []
    sup = Supervisor(command=(), limits=Limits(stop_step=10_000, **limits), log_path=tmp_path / "t.log",
                     metrics_csv=tmp_path / "metrics.csv", config=Config.from_env({}), poll_seconds=0.01,
                     report=reported.append, stop_grace_seconds=0, kill_wait_seconds=0)
    return sup, reported


def test_a_trainer_that_outlives_sigkill_is_named_not_forgotten(tmp_path):
    sup, reported = _unkillable_supervisor(tmp_path)
    assert sup.stop(_Unkillable()) is False
    assert sup.survivor == _Unkillable.pid
    assert any(f"pid {_Unkillable.pid} is still alive after SIGKILL" in line for line in reported)


def test_a_restart_is_not_launched_beside_a_trainer_that_would_not_die(tmp_path):
    """RESTART stops the wedged child and launches another; if the first will not go,
    the second would only OOM on its card."""
    sup, reported = _unkillable_supervisor(tmp_path, stall_polls=1, max_retries=2)
    launches = []
    sup.launch = lambda: launches.append(1) or _Unkillable()

    assert sup.run() == sup_mod.GAVE_UP
    assert launches == [1], "never a second trainer beside one still alive"
    assert sup.survivor == _Unkillable.pid


def test_a_lock_left_to_a_survivor_holds_the_card_until_it_dies(tmp_path):
    path = tmp_path / "gpu.lock"
    lock = GpuLock(path, label="run-a")
    lock.acquire()
    survivor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        lock.leave_to(survivor.pid)
        assert lock.holder()[0] == survivor.pid
        with pytest.raises(Preflight, match=f"held by pid {survivor.pid}"):
            GpuLock(path, label="run-b").acquire()
    finally:
        survivor.kill()
        survivor.wait()
    GpuLock(path, label="run-b").acquire()  # stale now, so taken over


def test_main_leaves_the_lock_to_a_survivor_instead_of_releasing_it(tmp_path, monkeypatch):
    lock_path = tmp_path / "gpu.lock"

    class Stub:
        survivor = 424242

        def __init__(self, **kw):
            pass

        def run(self):
            return sup_mod.GAVE_UP

    monkeypatch.setattr(sup_mod, "Supervisor", Stub)
    monkeypatch.setattr(sup_mod, "GpuLock", lambda label="": GpuLock(lock_path, label))
    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
                         "--min-free-gb", "0", "--skip-fit-gate"])
    assert code == 1
    assert GpuLock(lock_path).holder()[0] == Stub.survivor, "the card stays held by the trainer that would not die"


def test_a_budget_run_whose_trainer_would_not_die_is_not_scored_on_its_card(tmp_path, monkeypatch):
    """The final yardstick runs on the card; with the trainer still on it, it would
    only fail. Skipped, said, and the exit is non-zero, as for a failed yardstick."""
    class Stub:
        survivor = 424242

        def __init__(self, **kw):
            pass

        def run(self):
            return sup_mod.BUDGET_COMPLETE

    def no_instrument(*a, **k):
        raise AssertionError("nothing may score onto a card a trainer still holds")
    monkeypatch.setattr(sup_mod, "Supervisor", Stub)
    monkeypatch.setattr(sup_mod.subprocess, "run", no_instrument)
    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
                         "--min-free-gb", "0", "--skip-fit-gate", "--no-gpu-lock", "--spec", str(tmp_path / "spec.toml")])
    assert code == 1


def test_the_final_yardstick_scores_while_the_lock_still_names_this_supervisor(tmp_path, monkeypatch):
    """#523: the full-set score runs on the card. Released before it, the lock read
    free for those minutes and a second job could launch onto the card beside it."""
    lock_path = tmp_path / "gpu.lock"
    calls = []

    class Stub:
        survivor = None

        def __init__(self, **kw):
            pass

        def run(self):
            return sup_mod.BUDGET_COMPLETE

        def announce(self, text):
            pass

    def instrument(cmd, **kw):
        calls.append(cmd[3])
        assert GpuLock(lock_path).holder()[0] == os.getpid(), f"{cmd[3]} ran on a card the lock calls free"
        return subprocess.CompletedProcess(cmd, 0, stdout="KEEP\n", stderr="")

    def score(argv, env, lock):
        calls.append(argv[3])
        assert GpuLock(lock_path).holder()[0] == os.getpid(), "the score ran on a card the lock calls free"
        return 0, "scored"
    monkeypatch.setattr(sup_mod, "Supervisor", Stub)
    monkeypatch.setattr(sup_mod, "GpuLock", lambda label="": GpuLock(lock_path, label))
    monkeypatch.setattr(sup_mod.subprocess, "run", instrument)
    monkeypatch.setattr(sup_mod, "score_on_the_card", score)
    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
                         "--min-free-gb", "0", "--skip-fit-gate", "--spec", str(tmp_path / "spec.toml")])
    assert code == 0
    assert calls == ["score", "verdict", "card"]
    assert not lock_path.exists(), "and it is released once the yardstick is done"


# --- an abnormal supervisor exit never leaves its trainer under a free card (#516) ---

def _main_argv(tmp_path):
    return ["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
            "--min-free-gb", "0", "--skip-fit-gate", "--poll-seconds", "0.2", "--stall-polls", "100000"]


class _Sleeper(Supervisor):
    """The real Supervisor, launching a stand-in trainer that only sleeps."""
    def __init__(self, **kw):
        super().__init__(**{**kw, "command": (sys.executable, "-c", "import time; time.sleep(600)")})


def test_an_exception_out_of_run_stops_the_trainer_before_the_lock_goes(tmp_path, monkeypatch):
    """The first pass bar of #516: a bug escaping the loop used to propagate to
    main(), whose `finally` freed the card under a trainer still running."""
    lock_path = tmp_path / "gpu.lock"
    children, gone_at_release = [], []

    class WatchedLock(GpuLock):
        def name_child(self, pid):
            children.append(pid)
            super().name_child(pid)

        def release(self):
            gone_at_release.append(all(_gone(pid) for pid in children))
            super().release()

    class Faulty(_Sleeper):
        def observe(self, proc, started):
            assert f"child {proc.pid}" in lock_path.read_text(), "the trainer is named in the lock"
            raise RuntimeError("a bug escaping observe()")

    monkeypatch.setattr(sup_mod, "GpuLock", lambda label="": WatchedLock(lock_path, label))
    monkeypatch.setattr(sup_mod, "Supervisor", Faulty)
    with pytest.raises(RuntimeError, match="escaping observe"):
        sup_mod.main(_main_argv(tmp_path))
    assert len(children) == 1 and gone_at_release == [True], "the trainer goes before the card is freed"
    assert not lock_path.exists()
    assert "exiting on RuntimeError" in (tmp_path / "run_x.supervisor.log").read_text()


_SUPERVISOR_SCRIPT = """
import pathlib, sys
from trm.runtime import supervisor as sup_mod
from trm.runtime.gpu_lock import GpuLock
sup_mod.GpuLock = lambda label="": GpuLock(pathlib.Path({lock!r}), label)
class Sleeper(sup_mod.Supervisor):
    def __init__(self, **kw):
        super().__init__(**{{**kw, "command": (sys.executable, "-c", "import time; time.sleep(600)")}})
sup_mod.Supervisor = Sleeper
sys.exit(sup_mod.main({argv!r}))
"""


def _supervisor_with_a_trainer(tmp_path):
    """(a real supervisor process, the pid of the trainer it launched and named in the lock)."""
    lock_path = tmp_path / "gpu.lock"
    script = tmp_path / "supervise.py"
    script.write_text(_SUPERVISOR_SCRIPT.format(lock=str(lock_path), argv=_main_argv(tmp_path)))
    root = str(sup_mod.REPO_ROOT)
    proc = subprocess.Popen([sys.executable, str(script)], cwd=root, env={**os.environ, "PYTHONPATH": root},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while time.time() < deadline:
        text = lock_path.read_text() if lock_path.exists() else ""
        if "\nchild " in text:
            return proc, int(text.split("\nchild ")[1].split()[0])
        time.sleep(0.1)
    proc.kill()
    raise AssertionError("the supervisor never named its trainer in the lock")


def _clean_up(supervisor, trainer):
    supervisor.kill()
    supervisor.wait()
    if not _gone(trainer):
        os.kill(trainer, signal.SIGKILL)


def test_a_sigterm_to_the_supervisor_stops_its_trainer_and_frees_the_card(tmp_path):
    """Python's default TERM ran no `finally`: the lock stayed, naming a dead pid, and
    the next launch took it over beside a trainer still on the card."""
    supervisor, trainer = _supervisor_with_a_trainer(tmp_path)
    try:
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=60) == 128 + signal.SIGTERM
        assert _gone(trainer), "the trainer goes with its supervisor"
        assert not (tmp_path / "gpu.lock").exists()
    finally:
        _clean_up(supervisor, trainer)


def test_a_supervisor_killed_outright_leaves_the_card_to_its_trainer(tmp_path):
    """The second pass bar of #516. SIGKILL runs no handler, so nothing is stopped;
    the lock names the trainer, and nothing launches beside it while it lives."""
    supervisor, trainer = _supervisor_with_a_trainer(tmp_path)
    try:
        supervisor.kill()
        supervisor.wait(timeout=30)
        assert not _gone(trainer), "the orphaned trainer is still on the card"
        assert GpuLock(tmp_path / "gpu.lock").live_holder()[0] == trainer
        with pytest.raises(Preflight, match=rf"held by pid {trainer} \(child of dead pid {supervisor.pid}"):
            GpuLock(tmp_path / "gpu.lock", label="run-b").acquire()
    finally:
        _clean_up(supervisor, trainer)


def test_a_lock_whose_holder_and_named_child_both_died_is_stale(tmp_path):
    dead = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range(2)]
    for proc in dead:
        proc.wait()
    path = tmp_path / "gpu.lock"
    path.write_text(f"{dead[0].pid} run-a\nchild {dead[1].pid}\n")
    GpuLock(path, label="run-b").acquire()
    assert GpuLock(path).holder() == (os.getpid(), "run-b")


# --- the final score's eval is held by the lock, however the supervisor ends (#527) ---

# Stands in for `base_run score`: it spawns the "eval" that would hold the card, names
# it in a file, and waits on it, as base_run waits on eval_yardstick.
_SCORE_STUB = ("import subprocess, sys, time; "
               "e = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
               "open(sys.argv[1], 'w').write(str(e.pid)); e.wait()")


def _supervisor_scoring(tmp_path):
    """A process that holds the lock and runs score_on_the_card over the stub, as the
    supervisor does at the end of a budget run. Returns it and the eval's pid."""
    eval_pid = tmp_path / "eval.pid"
    script = tmp_path / "scoring.py"
    script.write_text(textwrap.dedent(f"""
        import pathlib, sys
        from trm.runtime.gpu_lock import GpuLock, exit_on_sigterm
        from trm.runtime import supervisor
        exit_on_sigterm()
        lock = GpuLock(pathlib.Path({str(tmp_path / "gpu.lock")!r}), label="score")
        lock.acquire()
        try:
            supervisor.score_on_the_card([sys.executable, "-c", {_SCORE_STUB!r}, {str(eval_pid)!r}], None, lock)
        finally:
            lock.release()
    """))
    root = str(sup_mod.REPO_ROOT)
    proc = subprocess.Popen([sys.executable, str(script)], cwd=root, env={**os.environ, "PYTHONPATH": root})
    deadline = time.time() + 60
    while not (eval_pid.exists() and eval_pid.read_text()):
        assert time.time() < deadline and proc.poll() is None, "the stub score never started its eval"
        time.sleep(0.1)
    return proc, int(eval_pid.read_text())


def test_a_sigterm_during_the_score_takes_its_eval_down_before_the_lock_goes(tmp_path):
    supervisor, eval_ = _supervisor_scoring(tmp_path)
    try:
        supervisor.send_signal(signal.SIGTERM)
        supervisor.wait(timeout=90)
        assert _gone(eval_), "an orphaned eval would keep the card under a free lock"
        assert not (tmp_path / "gpu.lock").exists()
    finally:
        _clean_up(supervisor, eval_)


def test_a_supervisor_killed_outright_during_the_score_leaves_the_card_held(tmp_path):
    supervisor, eval_ = _supervisor_scoring(tmp_path)
    try:
        supervisor.kill()
        supervisor.wait(timeout=30)
        assert not _gone(eval_)
        with pytest.raises(Preflight, match=rf"child of dead pid {supervisor.pid}"):
            GpuLock(tmp_path / "gpu.lock", label="run-b").acquire()
    finally:
        _clean_up(supervisor, eval_)
