"""List a run's checkpoints, and rewind it to an earlier one (#188).

    python -m trm.runtime.rewind runs/run_X/checkpoints                  # list
    python -m trm.runtime.rewind runs/run_X/checkpoints --to-opt-step N  # rewind

Recovery used to be hand-reasoning: after #157 flipped into SFT and OOM'd, the
newest checkpoint was contaminated, and the way back was to read each step's
monitor_state, divide micro-steps by ACCUMULATION_STEPS, and move directories out
of orbax's view by hand.

Why a separate command and not a `--from-step` flag on the trainer: orbax refuses
— silently, `save()` returns False — to write any step below the newest one it
can see. Resuming from an earlier checkpoint while later ones remain would train
on with no checkpoints at all, and the supervisor's crash-relaunch would jump
straight back to the contaminated one. And a flag on the trainer's command line is
replayed on every relaunch, rewinding the run again each time. So a rewind is one
act, done once: newer checkpoints are *set aside* (moved, never deleted) into a
directory orbax ignores, and the next ordinary resume picks up the chosen step.

Reads only the filesystem — no model, no device — so it is safe to run beside a
training job and cheap enough for an unattended runner.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
from dataclasses import dataclass

# Kept in step with trm/runtime/checkpoints.py; a test holds the two together.
BEST_SUBDIR = "best_val_ce"
SET_ASIDE_PREFIX = "set_aside_"


@dataclass
class Checkpoint:
    path: pathlib.Path
    step: int                 # orbax's number: the micro-step
    opt_step: int
    sft_active: bool | None   # None when the monitor state is unreadable

    def describe(self) -> str:
        phase = {True: "SFT", False: "pretrain", None: "phase unknown"}[self.sft_active]
        return f"{self.path.parent.name + '/' if self.path.parent.name == BEST_SUBDIR else ''}" \
               f"{self.step:>9}  opt {self.opt_step:>6}  {phase}"


def checkpoints_in(directory: pathlib.Path, accumulation_steps: int) -> list[Checkpoint]:
    """Finalized orbax step directories, oldest first. An unfinalized one (no
    _CHECKPOINT_METADATA) is a torn write orbax would not restore either."""
    found = []
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not (path.is_dir() and path.name.isdigit() and (path / "_CHECKPOINT_METADATA").exists()):
            continue
        step = int(path.name)
        try:
            state = json.loads((path / "monitor_state" / "metadata").read_text())
            sft = bool(state.get("sft_active") or state.get("sft_start_step") is not None)
        except (OSError, ValueError):
            sft = None
        found.append(Checkpoint(path, step, (step + 1) // accumulation_steps, sft))
    return sorted(found, key=lambda c: c.step)


def resolve(checkpoints: list[Checkpoint], to_opt_step: int) -> Checkpoint:
    """The newest checkpoint at or below the requested opt step. Never a silent
    fallback: no such checkpoint is an error that names what does exist."""
    eligible = [c for c in checkpoints if c.opt_step <= to_opt_step]
    if not eligible:
        have = ", ".join(str(c.opt_step) for c in checkpoints) or "none"
        raise SystemExit(f"no checkpoint at or below opt step {to_opt_step} (have opt steps: {have})")
    return eligible[-1]


def rewind(checkpoint_dir: pathlib.Path, to_opt_step: int, accumulation_steps: int,
           now: datetime.datetime | None = None) -> tuple[Checkpoint, list[pathlib.Path]]:
    """Set aside every checkpoint newer than the chosen one — in the rolling dir and
    the best dir alike, since either would make orbax refuse the resumed run's saves."""
    chosen = resolve(checkpoints_in(checkpoint_dir, accumulation_steps), to_opt_step)
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    shelf = checkpoint_dir / f"{SET_ASIDE_PREFIX}{stamp}_to_opt_{chosen.opt_step}"
    moved = []
    for directory, sub in ((checkpoint_dir, ""), (checkpoint_dir / BEST_SUBDIR, BEST_SUBDIR)):
        for ckpt in checkpoints_in(directory, accumulation_steps):
            if ckpt.step > chosen.step:
                target = shelf / sub / ckpt.path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ckpt.path), str(target))
                moved.append(target)
    return chosen, moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint_dir", type=pathlib.Path)
    ap.add_argument("--to-opt-step", type=int, default=None,
                    help="set aside every checkpoint newer than the newest one at or below this opt step")
    ap.add_argument("--accumulation-steps", type=int, default=None,
                    help="micro-steps per opt step (default: config ACCUMULATION_STEPS). "
                         "Pass the run's own value if it trained under a different one.")
    args = ap.parse_args(argv)
    if args.accumulation_steps is None:
        from trm.config import ACCUMULATION_STEPS
        args.accumulation_steps = ACCUMULATION_STEPS

    if args.to_opt_step is None:
        for directory in (args.checkpoint_dir, args.checkpoint_dir / BEST_SUBDIR):
            for ckpt in checkpoints_in(directory, args.accumulation_steps):
                print(ckpt.describe())
        return 0

    chosen, moved = rewind(args.checkpoint_dir, args.to_opt_step, args.accumulation_steps)
    print(f"resume point: {chosen.describe()}")
    for path in moved:
        print(f"set aside:    {path}")
    if not moved:
        print("nothing newer to set aside — the next resume already loads this step")
    elif chosen.sft_active:
        print("warning: the resume point itself is in the SFT phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
