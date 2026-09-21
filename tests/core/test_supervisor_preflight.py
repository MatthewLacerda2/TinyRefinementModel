"""Supervisor preflight: the disk-headroom refusal and the GPU lock that makes the
card a serial queue. Split out of test_supervisor.py (#325)."""

import os
import subprocess
import sys

import pytest

from trm.runtime.supervisor import (
    GpuLock,
    Preflight,
    check_disk_headroom,
)


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
