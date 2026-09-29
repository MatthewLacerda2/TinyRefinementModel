"""The cold tier: what a run copies to the HDD, and what that copy lets it free on the SSD (#458).

CLAUDE.md's storage convention: runs/ on the SSD is the live tier, the HDD the cold one.
For a run launched with a cold root (`COLD_ROOT`), the supervisor calls `ColdTier.tick`
every poll, and each tick:

* mirrors every finalized milestone to <cold>/<run>/milestones/<step>;
* copies the newest full-state rolling checkpoint to <cold>/<run>/fullstate/<step>
  once per stall-rule window (the spec's [stall] window_hours), and once more when
  the run ends. The trainer keeps only ROLLING_KEEP full-state checkpoints, and
  milestones hold weights only (#394), so without this copy a decay can branch from
  about "now" and nowhere else; with it, from the start of a flat stretch (#451);
* while the SSD has less than SSD_KEEP_FREE_GB free, removes from the SSD the oldest
  milestone whose cold copy is byte-identical.

Only that last step deletes, and it is narrow on purpose. `_remove_mirrored_milestone`
is the only function here that removes anything from the live tier. It refuses anything
but a real step dir directly under <run>/checkpoints/milestones/, anything inside a
protected root (runs/data/, the tokenized corpus) or holding one, and anything without a
byte-identical copy on the cold tier. Rolling and best checkpoints are never candidates,
and neither is the newest milestone, which a CPU yardstick pass may still be reading.
tests/core/test_cold_tier.py holds each rule, and that nothing else here deletes.

A copy lands whole or not at all: it is written to a dot-named partial dir, checked
against its source, then renamed into place, so a dir named by a step on the cold tier
is always a complete copy. A tick does nothing at all unless the cold root is there
and on another disk than the run: an unmounted HDD leaves its mount point an empty dir
on the SSD. Stdlib only besides the layout: the supervisor runs it beside a training job.
"""

from __future__ import annotations

import filecmp
import json
import os
import pathlib
import shutil
import tomllib
from dataclasses import dataclass
from typing import Callable

from trm.runtime.layout import MILESTONE_SUBDIR

# Orbax writes this into a step dir when the save commits: a step dir that has it is
# finalized, one that lacks it may still be being written.
MARKER = "_CHECKPOINT_METADATA"
FULLSTATE_SUBDIR = "fullstate"


def finalized_steps(directory: pathlib.Path) -> list[pathlib.Path]:
    """The finalized step dirs directly under `directory`, oldest step first. Only
    digit-named dirs count, so from a run's checkpoints/ this is the rolling ones:
    best_val_ce/, milestones/ and orbax's tmp dirs are not step names."""
    if not directory.is_dir():
        return []
    steps = [p for p in directory.iterdir() if p.name.isdigit() and (p / MARKER).is_file()]
    return sorted(steps, key=lambda p: int(p.name))


def committed_at(step_dir: pathlib.Path) -> float:
    """When orbax committed this checkpoint, in seconds. Read from its metadata, which a
    copy carries verbatim; a file's mtime would be the copy's time on this HDD."""
    marker = step_dir / MARKER
    try:
        return json.loads(marker.read_text())["commit_timestamp_nsecs"] / 1e9
    except (OSError, ValueError, KeyError, TypeError):
        return marker.stat().st_mtime


def _sizes(root: pathlib.Path) -> dict[str, int]:
    def fail(error):
        raise error
    return {os.path.relpath(os.path.join(d, f), root): os.path.getsize(os.path.join(d, f))
            for d, _, files in os.walk(root, onerror=fail) for f in files}


def same_bytes(a: pathlib.Path, b: pathlib.Path) -> bool:
    """Whether two step dirs hold the same files with the same contents."""
    try:
        files = _sizes(a)
        if not files or files != _sizes(b):
            return False
        return all(filecmp.cmp(a / f, b / f, shallow=False) for f in files)
    except OSError:
        return False


def mirror(src: pathlib.Path, dst: pathlib.Path) -> bool:
    """Copy the finalized step dir `src` to `dst`, whole or not at all. False when
    `dst` is already there. Raises OSError when `src` is not a finalized step dir or
    changes under the copy (rolling retention can evict it mid-copy); nothing is left
    at `dst` then, and the next tick tries again."""
    if dst.exists():
        return False
    if not (src / MARKER).is_file():
        raise FileNotFoundError(f"{src} is not a finalized checkpoint")
    partial = dst.parent / f".{dst.name}.partial"
    dst.parent.mkdir(parents=True, exist_ok=True)
    _discard_partial(partial)  # a copy a crash interrupted
    try:
        expected = _sizes(src)
        for d, _, files in os.walk(src):  # empty dirs too: orbax may read one
            out = partial / os.path.relpath(d, src)
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                # Contents only: this HDD is NTFS, where permission bits and times do not copy.
                shutil.copyfile(os.path.join(d, f), out / f)
        if MARKER not in expected or _sizes(partial) != expected:
            raise OSError(f"{src} changed while it was copied")
        os.rename(partial, dst)
    except BaseException:
        _discard_partial(partial)
        raise
    return True


def _discard_partial(partial: pathlib.Path) -> None:
    """Remove an unfinished copy — and nothing that is not one."""
    if not partial.exists():
        return
    if not (partial.name.startswith(".") and partial.name.endswith(".partial")):
        raise ValueError(f"refusing to remove {partial}: not a partial copy")
    shutil.rmtree(partial)


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or path.is_relative_to(root)


def stall_window_hours(spec: pathlib.Path | None) -> float | None:
    """The stall rule's window, from a base-run spec's [stall] table; None without one.
    (instruments/stall.py reads the whole rule; trm/ never imports instruments/.)"""
    if spec is None:
        return None
    with open(spec, "rb") as f:
        table = tomllib.load(f).get("stall")
    return float(table["window_hours"]) if table else None


def cold_root_problem(cold_root: pathlib.Path, live: pathlib.Path) -> str | None:
    """Why `cold_root` cannot be the cold tier of runs on `live`, or None. It is never
    created here: a mount point with the HDD unmounted is an empty dir on the SSD, and
    its TRM_cold/ is then simply missing."""
    if not cold_root.is_dir():
        return f"{cold_root} is not a directory (is the HDD mounted?)"
    if not os.access(cold_root, os.W_OK):
        return f"{cold_root} is not writable"
    if os.stat(cold_root).st_dev == os.stat(live).st_dev:
        return (f"{cold_root} is on the same disk as {live}: a copy there frees nothing and "
                f"survives nothing")
    return None


@dataclass
class ColdTier:
    run_dir: pathlib.Path  # runs/<run_id>, on the SSD
    cold_root: pathlib.Path  # COLD_ROOT; this run's copies go under <cold_root>/<run_id>
    keep_free_gb: float  # SSD_KEEP_FREE_GB
    fullstate_every_hours: float | None  # the stall window; None = only the final copy
    protected: tuple[pathlib.Path, ...] = ()  # roots nothing under may be removed (runs/data/)
    free_gb: Callable[[], float] | None = None  # SSD free space; a test fakes it

    @property
    def cold_run(self) -> pathlib.Path:
        return self.cold_root / self.run_dir.name

    def _ssd_free_gb(self) -> float:
        if self.free_gb is not None:
            return self.free_gb()
        return shutil.disk_usage(self.run_dir).free / 1e9

    def tick(self, final: bool = False) -> list[str]:
        """Mirror, snapshot, prune; one line per thing done. Raises OSError, having
        removed nothing, when the cold root cannot take copies right now: gone (the HDD
        unmounted) or on the SSD itself. Asked every tick, not once: a mount can go."""
        live = self.run_dir if self.run_dir.exists() else self.run_dir.parent
        problem = cold_root_problem(self.cold_root, live)
        if problem:
            raise OSError(problem)
        done = []
        for step in finalized_steps(self.run_dir / "checkpoints" / MILESTONE_SUBDIR):
            if mirror(step, self.cold_run / MILESTONE_SUBDIR / step.name):
                done.append(f"cold: milestone {step.name} mirrored")
        due = self.fullstate_due(final)
        if due is not None and mirror(due, self.cold_run / FULLSTATE_SUBDIR / due.name):
            done.append(f"cold: full-state checkpoint {due.name} copied"
                        + (" (the run's last)" if final else ""))
        return done + self.prune()

    def fullstate_due(self, final: bool = False) -> pathlib.Path | None:
        """The rolling checkpoint to copy now, or None: the newest one, once a stall
        window has passed since the newest copy was committed; at the run's end,
        regardless of the window."""
        rolling = finalized_steps(self.run_dir / "checkpoints")
        if not rolling:
            return None
        newest = rolling[-1]
        copies = finalized_steps(self.cold_run / FULLSTATE_SUBDIR)
        if any(c.name == newest.name for c in copies):
            return None
        if final:
            return newest
        if self.fullstate_every_hours is None:
            return None
        if not copies:
            return newest
        last = max(committed_at(c) for c in copies)
        return newest if committed_at(newest) - last >= self.fullstate_every_hours * 3600 else None

    def prune(self) -> list[str]:
        """While the SSD is under its margin, remove the oldest mirrored milestone."""
        done = []
        for step in finalized_steps(self.run_dir / "checkpoints" / MILESTONE_SUBDIR)[:-1]:
            free = self._ssd_free_gb()
            if free >= self.keep_free_gb:
                break
            if same_bytes(step, self.cold_run / MILESTONE_SUBDIR / step.name):
                self._remove_mirrored_milestone(step)
                done.append(f"cold: milestone {step.name} removed from the SSD ({free:.1f}GB free, "
                            f"keeping {self.keep_free_gb:g}GB); its copy is on the cold tier")
        return done

    def _remove_mirrored_milestone(self, step: pathlib.Path) -> None:
        """The one deletion on the live tier, and every rule it answers to."""
        home = (self.run_dir / "checkpoints" / MILESTONE_SUBDIR).resolve()
        target = step.resolve()
        copy = (self.cold_run / MILESTONE_SUBDIR / step.name).resolve()
        why = None
        if step.is_symlink() or not step.name.isdigit() or target.parent != home:
            why = "not a milestone step dir"
        elif any(_inside(target, r.resolve()) or _inside(r.resolve(), target) for r in self.protected):
            why = "inside a protected root, or holding one"
        elif _inside(copy, target) or _inside(target, copy):
            why = "its cold copy is the same place"
        elif not same_bytes(target, copy):
            why = "no byte-identical copy on the cold tier"
        if why:
            raise ValueError(f"refusing to remove {step}: {why}")
        shutil.rmtree(target)
