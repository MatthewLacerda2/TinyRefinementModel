"""The serial queue for the one card, and the refusal that enforces it.

There is a single RTX 2060 with 6 GB. Two trainers on it is not a slow run, it is
an OOM — and by this box's `oom_score_adj` the kernel usually takes the Claude
session rather than the second trainer, so the failure lands on whoever is watching
rather than on whatever caused it.

This lives on its own, away from the supervisor, because **everything that puts
work on the card has to be able to take it**. It started inside
`trm/runtime/supervisor.py`, which meant the supervisor queued against itself and
`instruments/experiment.py` — the runner that executes every pre-registered pair,
which is most of what the card actually does — queued against nothing (#445).
Stdlib only, like `layout.py`, so an instrument can import it without dragging jax
into a process that only wanted to ask whether the card is free.
"""

from __future__ import annotations

import os
import pathlib

def shared_runs_dir(checkout: pathlib.Path) -> pathlib.Path:
    """`runs/` in the main checkout, even when the caller is in a linked worktree.

    The lock has to name one place on this box. Almost all work here happens in
    worktrees off `origin/main`, and a worktree that kept its own
    `runs/.gpu.lock` would queue against itself and nothing else — the same bug
    this file was split out of the supervisor to fix, one level down.

    In a linked worktree `.git` is a *file* reading `gitdir: <main>/.git/worktrees/<name>`,
    so the main checkout is the parent of that `.git`. Anything unexpected falls
    back to the checkout we are in, which is the old behaviour.
    """
    marker = checkout / ".git"
    if marker.is_file():
        text = marker.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = pathlib.Path(text.split(":", 1)[1].strip())
            for parent in gitdir.parents:
                if parent.name == ".git":
                    return parent.parent / "runs"
    return checkout / "runs"


# Spelled out rather than imported from the supervisor: that import is the thing
# this file exists to undo.
GPU_LOCK = shared_runs_dir(pathlib.Path(__file__).resolve().parents[2]) / ".gpu.lock"


class Preflight(Exception):
    """A reason not to start. Raised before anything expensive happens."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class GpuLock:
    """The single RTX 2060 is a serial queue; this is the queue.

    A stale lock — the holder died without releasing — is taken over rather than
    respected. A lock nobody can clear is worse than no lock: it turns one crash
    into a card that stays idle until a human notices.
    """

    def __init__(self, path: pathlib.Path = GPU_LOCK, label: str = ""):
        self.path = path
        self.label = label
        self.held = False

    def holder(self) -> tuple[int, str] | None:
        if not self.path.exists():
            return None
        try:
            pid_text, _, label = self.path.read_text().strip().partition(" ")
            return int(pid_text), label
        except ValueError:
            return None  # unreadable lock is a stale lock

    def acquire(self) -> None:
        current = self.holder()
        if current and _pid_alive(current[0]):
            raise Preflight(
                f"the GPU is held by pid {current[0]} ({current[1] or 'unlabelled'}) — "
                f"one run at a time on this card")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()} {self.label}\n")
        self.held = True

    def release(self) -> None:
        """Only ever removes our own lock — a supervisor that took over a stale
        lock must not delete whatever replaced it."""
        current = self.holder()
        if self.held and current and current[0] == os.getpid():
            self.path.unlink(missing_ok=True)
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False
