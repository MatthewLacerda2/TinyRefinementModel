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
import pathlib
import shutil
import json
from dataclasses import dataclass

from trm.runtime.layout import BEST_SUBDIR, MILESTONE_SUBDIR

SET_ASIDE_PREFIX = "set_aside_"


@dataclass
class Checkpoint:
    path: pathlib.Path
    step: int                 # orbax's number: the micro-step
    opt_step: int

    def describe(self) -> str:
        # A milestone holds weights only (#394), so it is a branch point, not a
        # resume point: say so where someone reads the list to pick one.
        weights_only = "" if (self.path / "optimizer").exists() else "  (weights only — not a resume point)"
        return f"{self.path.parent.name + '/' if self.path.parent.name in (BEST_SUBDIR, MILESTONE_SUBDIR) else ''}" \
               f"{self.step:>9}  opt {self.opt_step:>6}{weights_only}"


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
        found.append(Checkpoint(path, step, (step + 1) // accumulation_steps))
    return sorted(found, key=lambda c: c.step)


def refuse_sft_phase_resume(monitor_state: dict, step: int, checkpoint_dir,
                            accumulation_steps: int | None = None) -> None:
    """Refuse to resume a checkpoint written inside the retired SFT phase (#323).

    The in-run SFT flip is gone, so the trainer would resume such a checkpoint as
    pretraining: a different data mixture and 10x the LR, with nothing in the log
    to say so. Checkpoints from before the removal still carry the phase fields;
    ones without them (every checkpoint written since) read as pretraining.
    """
    sft_start_step = monitor_state.get("sft_start_step")
    if sft_start_step is None:
        return
    if accumulation_steps is None:
        from trm.config import ACCUMULATION_STEPS
        accumulation_steps = ACCUMULATION_STEPS
    # The flip happened on an opt-step boundary, after that boundary's saves, so
    # every checkpoint at or below this opt step is still pretraining.
    last_clean_opt_step = (sft_start_step + 1) // accumulation_steps
    raise SystemExit(
        f"❌ checkpoint step {step} is in the SFT phase that began at micro-step {sft_start_step} "
        f"(opt step {last_clean_opt_step}). The in-run SFT flip was removed (#323), so resuming it "
        f"would silently continue as pretraining on a different mixture and LR. Rewind to the "
        f"last pretraining checkpoint first:\n"
        f"    python -m trm.runtime.rewind {checkpoint_dir} --to-opt-step {last_clean_opt_step}\n"
        f"(Under the supervisor this exit reads as a crash: it is relaunched, then reported "
        f"GAVE_UP. These lines are the reason.)")


def refuse_sft_phase_checkpoint_dir(checkpoint_dir, accumulation_steps: int) -> None:
    """The same refusal, read from disk before a launch touches anything: no run
    session appended, no model built, no orbax restore. Reads the newest finalized
    checkpoint's monitor state — the one a resume would load. An unreadable state
    is left for the restore to report."""
    found = checkpoints_in(pathlib.Path(checkpoint_dir), accumulation_steps)
    if not found:
        return
    newest = found[-1]
    try:
        state = json.loads((newest.path / "monitor_state" / "metadata").read_text())
    except (OSError, ValueError):
        return
    refuse_sft_phase_resume(state, newest.step, checkpoint_dir, accumulation_steps)


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
    """Set aside every checkpoint newer than the chosen one — in the rolling, best and
    milestone dirs alike, since any of them would make orbax refuse the resumed run's
    saves. Milestones are set aside too, not kept in place: a milestone from the
    abandoned stretch is still on disk, just out of the resumed run's way."""
    chosen = resolve(checkpoints_in(checkpoint_dir, accumulation_steps), to_opt_step)
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    shelf = checkpoint_dir / f"{SET_ASIDE_PREFIX}{stamp}_to_opt_{chosen.opt_step}"
    moved = []
    for sub in ("", BEST_SUBDIR, MILESTONE_SUBDIR):
        directory = checkpoint_dir / sub
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
        for directory in (args.checkpoint_dir, args.checkpoint_dir / BEST_SUBDIR,
                          args.checkpoint_dir / MILESTONE_SUBDIR):
            for ckpt in checkpoints_in(directory, args.accumulation_steps):
                print(ckpt.describe())
        return 0

    chosen, moved = rewind(args.checkpoint_dir, args.to_opt_step, args.accumulation_steps)
    print(f"resume point: {chosen.describe()}")
    for path in moved:
        print(f"set aside:    {path}")
    if not moved:
        print("nothing newer to set aside — the next resume already loads this step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
