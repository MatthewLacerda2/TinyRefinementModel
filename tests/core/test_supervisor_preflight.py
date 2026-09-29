"""Supervisor preflight: the disk-headroom refusal and the GPU lock that makes the
card a serial queue. Split out of test_supervisor.py (#325)."""

import json
import os
import subprocess
import sys
import textwrap

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
    with pytest.raises(RuntimeError):
        with GpuLock(path, label="x"):
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
                         "--no-gpu-lock", "--", "--checkpoint-path", str(run_dir / "checkpoints")])

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
