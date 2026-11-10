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
import signal


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


def exit_on_sigterm():
    """Turn SIGTERM into SystemExit, so a holder stopped from outside unwinds: it
    stops the child it put on the card and its `finally` leaves the lock right
    (#516, #519). Returns the handler it replaced. The trainer does the same
    (`exit_cleanly_on_sigterm`)."""
    def _raise(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # once: a second must not abort the stop
        raise SystemExit(128 + signum)
    return signal.signal(signal.SIGTERM, _raise)


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
    into a card that stays idle until a human notices. "Died" means the holder *and*
    every child it named (`name_child`): a supervisor killed outright leaves its
    trainer on the card, and that card is not free (#516).

    The file is `<holder pid> <label>`, then one `child <pid>` line per named child.
    """

    def __init__(self, path: pathlib.Path = GPU_LOCK, label: str = ""):
        self.path = path
        self.label = label
        self.held = False

    def _read(self) -> tuple[int, str, list[int]] | None:
        """(holder pid, label, named child pids); None when there is no lock, or an
        unreadable one — which is a stale one."""
        try:
            first, *rest = self.path.read_text().strip().splitlines()
            pid_text, _, label = first.partition(" ")
            children = [int(line.split()[1]) for line in rest if line.startswith("child ")]
            return int(pid_text), label, children
        except (FileNotFoundError, ValueError, IndexError):
            return None

    def _write(self, text: str) -> None:
        """Replace the file in one step: a reader never sees it half-written, which
        would read as unreadable, so stale, and be taken over under a live run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        partial.write_text(text)
        os.replace(partial, self.path)

    def holder(self) -> tuple[int, str] | None:
        """(pid, label) of whoever took the lock, alive or not."""
        read = self._read()
        return None if read is None else read[:2]

    def live_holder(self) -> tuple[int, str] | None:
        """(pid, label) of the live process that keeps the card taken — the holder,
        or a child it named that outlived it — or None when the card is free."""
        read = self._read()
        if read is None:
            return None
        pid, label, children = read
        if _pid_alive(pid):
            return pid, label
        for child in children:
            if _pid_alive(child):
                return child, f"child of dead pid {pid}" + (f", {label}" if label else "")
        return None

    def _ours(self) -> bool:
        if not self.held:
            return False
        current = self.holder()
        return current is not None and current[0] == os.getpid()

    def acquire(self) -> None:
        current = self.live_holder()
        if current:
            raise Preflight(
                f"the GPU is held by pid {current[0]} ({current[1] or 'unlabelled'}) — "
                f"one run at a time on this card")
        self._write(f"{os.getpid()} {self.label}\n")
        self.held = True

    def name_child(self, pid: int) -> None:
        """Name `pid`, a process we put on the card, in our lock (#516). A holder
        killed outright (SIGKILL, the OOM killer) runs no cleanup; its lock then keeps
        the card taken for as long as the child lives. One child at a time: each call
        replaces the last, since the one before it is gone or stopped."""
        if self._ours():
            self._write(f"{os.getpid()} {self.label}\nchild {pid}\n")

    def release(self) -> None:
        """Only ever removes our own lock — a supervisor that took over a stale
        lock must not delete whatever replaced it."""
        if self._ours():
            self.path.unlink(missing_ok=True)
        self.held = False

    def leave_to(self, pid: int) -> None:
        """Hand our lock to `pid`, a process on the card this one could not stop
        (#512). The card stays taken while it lives; once it dies, the lock is stale
        and the next launch takes it over."""
        if self._ours():
            self._write(f"{pid} {self.label} (left behind by pid {os.getpid()})\n")
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False
