import gc
import glob
import os
import signal

import orbax.checkpoint as ocp
from flax import nnx

from trm.runtime.layout import BEST_SUBDIR, CHECKPOINT_ITEMS, MILESTONE_ITEMS, MILESTONE_SUBDIR, ROLLING_KEEP
from trm.runtime.monitor import LossMonitor
from trm.runtime.resume_state import ResumeState


def discover_latest_run(runs_root="runs"):
    if not os.path.exists(runs_root):
        return None
    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    if run_dirs:
        return os.path.basename(run_dirs[-1])
    return None

def discover_latest_checkpoint_run(runs_root="runs"):
    if not os.path.exists(runs_root):
        return None, None

    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))

    for r_dir in reversed(run_dirs):
        chk_dir = os.path.join(r_dir, "checkpoints")
        if os.path.exists(chk_dir):
            try:
                mngr = ocp.CheckpointManager(
                    chk_dir,
                    item_names=CHECKPOINT_ITEMS,
                )
                if mngr.latest_step() is not None:
                    run_id = os.path.basename(r_dir)
                    return chk_dir, run_id
            except Exception as e:
                # Orbax raises a variety of errors on malformed checkpoint dirs;
                # skip them, but say which directory was skipped and why.
                print(f"⚠️ Skipping unreadable checkpoint dir {chk_dir}: {e}")
    return None, None

# Milestone checkpoints (MILESTONE_SUBDIR, trm/runtime/layout.py) are never evicted (#187).
# Retention keeps the newest, and when a run goes bad the newest are the broken
# ones: #157's SFT flip came within two saves of evicting every clean checkpoint.
# A milestone is kept regardless of recency — the branch point the registry wants
# ("fine-tune from the 1B-token checkpoint"), which rolling retention has always
# deleted by the time a run ends.
#
# Two things make a milestone cheap enough to keep forever (#394): they are spaced
# by doubling (MILESTONE_* in trm/settings.py), and they hold the weights, not the whole
# training state. Weights are what the yardstick, the registry and the trajectory
# figures read; the optimizer state is ~3/4 of a full save and only a resume wants
# it, which is what the rolling checkpoints are for. So a milestone is not a resume
# point, and `python -m trm.runtime.rewind` says so when it lists one.


def make_milestone_manager(checkpoint_path):
    """The milestone dir's manager: keeps every step, and holds no optimizer (#394)."""
    return ocp.CheckpointManager(
        os.path.join(str(checkpoint_path), MILESTONE_SUBDIR),
        item_names=MILESTONE_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )


def milestone_thresholds(first, ratio, count):
    """The token counts a milestone is kept at: first, first*ratio, … capped at
    `count` of them (MILESTONE_FIRST_TOKENS / _RATIO / _MAX_COUNT). `first <= 0`
    turns milestones off."""
    if first <= 0 or count <= 0:
        return ()
    marks, mark = [], float(first)
    for _ in range(count):
        marks.append(int(mark))
        mark *= ratio
    return tuple(marks)


def milestone_due(opt_step, since_opt_step, tokens_per_opt_step, thresholds):
    """Whether a milestone token count was crossed between two optimizer steps.

    Checked every optimizer step, not only where a rolling checkpoint lands: the
    early milestones (8M tokens is 61 opt steps at the shipping config) are denser
    than the rolling cadence, and a milestone that waits for the next boundary is
    not the point in training it claims to be.
    """
    now = opt_step * tokens_per_opt_step
    before = max(since_opt_step, 0) * tokens_per_opt_step
    return any(before < mark <= now for mark in thresholds)


def _monitor_state(monitor, run_id):
    """The JSON side of a save: everything a resume rebuilds the run from."""
    return ResumeState.of(monitor, run_id).saved()


def save_milestone(mngr, step, model, monitor, run_id, wait=False):
    """Persist the weights (plus the small JSON state) at `step` — no optimizer.

    ~550 MB against ~2.4 GB for a full save, which is what lets every milestone of
    a run survive to the end of it (#394). Everything else matches `save_checkpoint`,
    including the one-write-in-flight rule.
    """
    wait_for_pending_saves()
    saved = mngr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            monitor_state=ocp.args.JsonSave(_monitor_state(monitor, run_id)),
            step=ocp.args.JsonSave(step),
        ),
    )
    if not saved:
        raise RuntimeError(
            f"orbax declined to save milestone {step} in {mngr.directory}: its newest is "
            f"step {mngr.latest_step()}.")
    if wait:
        mngr.wait_until_finished()
    else:
        _PENDING.append(mngr)


def _make_best_manager(checkpoint_path):
    """Best-only manager: a sibling 'best_val_ce/' dir holding the best-val-CE checkpoints,
    distinct from the rolling-latest manager so its retention can't drop the
    state a resume must load."""
    return ocp.CheckpointManager(
        os.path.join(checkpoint_path, BEST_SUBDIR),
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=ROLLING_KEEP, create=True),
    )


# Managers whose last save may still be writing to disk (#218).
_PENDING = []


def wait_for_pending_saves():
    """Block until every asynchronous checkpoint write has landed. Called before
    the next save, and on the way out of training — including a SIGTERM."""
    while _PENDING:
        _PENDING.pop().wait_until_finished()


def save_checkpoint(mngr, step, model, optimizer, monitor, run_id, wait=True):
    """Persist the full training state (model + optimizer + monitor + step) under
    `mngr` at `step`. Shared by every manager — they use one save schema.

    With `wait=False` it returns once the state is copied to host, and the disk
    write finishes in orbax's background thread while training goes on (#218):
    the blocking write held the GPU at 0% for ~12s per checkpoint. Measured safe
    on the RTX 2060 at the shipping config: save() returned after the 2.4s copy,
    129 micro-steps and an optimizer apply then ran during the write — reusing the
    donated buffers — and the restored model and optimizer were bit-identical to
    the state at save time. Two rules keep it that way:
      * at most one write in flight: a new save first waits for the previous one,
        so host RAM holds one ~1.7GB copy, not one per coinciding manager;
      * nothing mutable is handed over by reference (ce_history is copied), since
        serialization may happen after this returns.

    Raises if orbax declines the save. It returns False — no exception, no log —
    for any step at or below the newest checkpoint it can see, so a run resumed
    from an earlier checkpoint with later ones still on disk would otherwise train
    on for days writing nothing (#188). `python -m trm.runtime.rewind` is the way
    to resume from an earlier step.
    """
    wait_for_pending_saves()
    saved = mngr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            monitor_state=ocp.args.JsonSave(_monitor_state(monitor, run_id)),
            step=ocp.args.JsonSave(step),
        ),
    )
    if not saved:
        raise RuntimeError(
            f"orbax declined to save step {step} in {mngr.directory}: its newest checkpoint is "
            f"step {mngr.latest_step()}. Checkpoints newer than the resume point are still in "
            f"view — set them aside with `python -m trm.runtime.rewind`.")
    if wait:
        mngr.wait_until_finished()
    else:
        _PENDING.append(mngr)


def exit_cleanly_on_sigterm():
    """Turn SIGTERM into SystemExit, so `finally` blocks run.

    The supervisor stops a run with TERM, then KILL after a 60s grace. Python's
    default TERM action ends the process without unwinding — harmless while every
    save blocked, fatal to an asynchronous one: the budget-stop TERM can land while
    the run's final checkpoint is still being written. Unwinding lets the trainer
    wait for that write (~16s at the shipping config) inside the grace.
    """
    def _raise(signum, _frame):
        # Say so in the log. SystemExit prints no traceback, so a TERM'd trainer
        # used to end mid-stream with nothing to distinguish it from a hard kill —
        # three Muon arms of #26 died that way on 2026-09-14 and the cause was
        # unrecorded.
        print(f"🛑 received SIGTERM (pid {os.getpid()}) — exiting cleanly, waiting for pending "
              f"checkpoint writes", flush=True)
        # Once: a second TERM would abort the wait this one started. Both can come at
        # once — a session-wide TERM reaches the supervisor too, which stops us (#516).
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _raise)


class StopRequest:
    """While the training loop runs, a SIGTERM asks it to stop rather than stopping it (#566).

    Under `exit_cleanly_on_sigterm` alone, a TERM unwinds at once, and the run loses
    everything since its last rolling checkpoint: up to 256 opt steps, about an hour
    of a pair arm, at every pause. With this installed, the TERM only sets
    `requested`. The loop finishes the optimizer window it is in, saves the rolling
    checkpoint there, and exits with the code the TERM would have given. A window
    plus the save fits inside the supervisor's 60 s grace and the pair harness's 120 s.
    """

    def __init__(self):
        self.requested = False
        self._previous = None

    def install(self) -> "StopRequest":
        def _request(_signum, _frame):
            print(f"🛑 received SIGTERM (pid {os.getpid()}) — finishing this optimizer window, saving it, "
                  f"then exiting (#566)", flush=True)
            # Once, as exit_cleanly_on_sigterm does: a second TERM must not cut the save short.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self.requested = True

        self._previous = signal.signal(signal.SIGTERM, _request)
        return self

    def uninstall(self) -> None:
        """Hand SIGTERM back to whoever had it, unless a stop already set it to ignore."""
        if not self.requested and self._previous is not None:
            signal.signal(signal.SIGTERM, self._previous)


def load_or_create_checkpoint(config, model, optimizer, checkpoint_path, force_new_run=False):
    monitor = LossMonitor.of(config)
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=ROLLING_KEEP, create=True),
    )
    best_mngr = _make_best_manager(checkpoint_path)

    if not force_new_run and mngr.latest_step() is not None:
        latest_step = mngr.latest_step()
        print(f"📖 Loading Orbax checkpoint from step {latest_step}...")
        restored = mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(nnx.state(model)),
                optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
                monitor_state=ocp.args.JsonRestore(),
                step=ocp.args.JsonRestore(),
            ),
        )

        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])

        start_step = restored["step"] + 1
        m_state = restored["monitor_state"]
        ResumeState.load(m_state, f"checkpoint step {latest_step} in {checkpoint_path}").restore(
            monitor, micro_step=restored["step"])

        print(f"✅ Resuming from step {start_step} "
              f"({monitor.samples_seen:,} samples consumed)")
        del restored
        gc.collect()
    else:
        if force_new_run:
            print("🆕 Force New Run specified, starting from scratch...")
        else:
            print("🆕 No checkpoint found, starting from scratch...")
        # Micro-steps count from 0, so the trainer's boundary ((step + 1) %
        # ACCUMULATION_STEPS == 0) falls on the micro-step that completes the
        # optimizer's window. From 1, every validation, log row and checkpoint of
        # "opt step N" ran one micro-step early, with N-1 updates applied and 127
        # gradients waiting in the accumulator (#355; found by
        # tests/apparatus/test_trainer_end_to_end.py).
        start_step = 0

    return mngr, best_mngr, monitor, start_step
