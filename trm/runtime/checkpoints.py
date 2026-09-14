import os
import gc
from flax import nnx
import orbax.checkpoint as ocp
from trm.runtime.monitor import LossMonitor

def discover_latest_run(runs_root="runs"):
    if not os.path.exists(runs_root):
        return None
    import glob
    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    if run_dirs:
        return os.path.basename(run_dirs[-1])
    return None

def discover_latest_checkpoint_run(runs_root="runs"):
    if not os.path.exists(runs_root):
        return None, None
    
    import glob
    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    
    for r_dir in reversed(run_dirs):
        chk_dir = os.path.join(r_dir, "checkpoints")
        if os.path.exists(chk_dir):
            try:
                mngr = ocp.CheckpointManager(
                    chk_dir,
                    item_names=("model", "optimizer", "monitor_state", "step"),
                )
                if mngr.latest_step() is not None:
                    run_id = os.path.basename(r_dir)
                    return chk_dir, run_id
            except Exception as e:
                # Orbax raises a variety of errors on malformed checkpoint dirs;
                # skip them, but say which directory was skipped and why.
                print(f"⚠️ Skipping unreadable checkpoint dir {chk_dir}: {e}")
    return None, None

# Sibling subdir of the rolling-latest checkpoints holding the best held-out-CE
# checkpoints. Kept separate so best-retention never evicts the latest. Named for
# its criterion (#222): the old `best/` was selected on noisy train CE and went
# stale on two runs, and a new name keeps those archives from passing for these.
BEST_SUBDIR = "best_val_ce"
CHECKPOINT_ITEMS = ("model", "optimizer", "monitor_state", "step")

# Sibling subdir holding milestone checkpoints, which nothing evicts (#187).
# Retention keeps the newest, and when a run goes bad the newest are the broken
# ones: #157's SFT flip came within two saves of evicting every clean checkpoint.
# A milestone every MILESTONE_EVERY_TOKENS is kept regardless of recency — the
# recovery point when a run goes wrong, and the branch point the registry wants
# ("fine-tune from the 1B-token checkpoint"), which rolling retention has always
# deleted by the time a run ends.
MILESTONE_SUBDIR = "milestones"
MILESTONE_EVERY_TOKENS = int(os.environ.get("MILESTONE_EVERY_TOKENS", 500_000_000))


def make_milestone_manager(checkpoint_path):
    return ocp.CheckpointManager(
        os.path.join(str(checkpoint_path), MILESTONE_SUBDIR),
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )


def milestone_due(opt_step, every_opt_steps, tokens_per_opt_step, every_tokens=MILESTONE_EVERY_TOKENS):
    """Whether a token milestone was crossed since the previous checkpoint boundary.

    Checked only where a rolling checkpoint is written, so a milestone lands on the
    first boundary at or past each multiple of `every_tokens` — exact multiples
    never line up with the save cadence, and crossing is what matters.
    """
    if every_tokens <= 0:
        return False
    tokens_now = opt_step * tokens_per_opt_step
    tokens_before = max(opt_step - every_opt_steps, 0) * tokens_per_opt_step
    return tokens_now // every_tokens > tokens_before // every_tokens


def _make_best_manager(checkpoint_path):
    """Best-only manager: a sibling 'best_val_ce/' dir holding the best-val-CE checkpoints,
    distinct from the rolling-latest manager so its retention can't drop the
    state a resume must load."""
    return ocp.CheckpointManager(
        os.path.join(checkpoint_path, BEST_SUBDIR),
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )


def restore_tolerating_legacy(read, model, model_key="model"):
    """Restore, tolerating variables a *past* version of this model saved and the
    current one no longer defines.

    Orbax matches structure strictly, which is what we want: a checkpoint missing
    weights the model needs must fail loudly rather than load half a network in
    silence. But strictness also orphans a checkpoint the moment a buffer is
    deleted — and #105 deleted the refiner's vestigial hunch buffer, which every
    base-run checkpoint on disk still carries. So: read with the honest structure
    first, and only if that mismatches, retry with the buffers the model itself
    declares its old checkpoints held (`legacy_checkpoint_variables`), dropping
    them from the result. Both attempts are strict, so nothing else slips through.

    `read(model_target)` performs the actual restore and returns the item dict.
    """
    try:
        return read(nnx.state(model))
    except ValueError as mismatch:
        legacy = model.legacy_checkpoint_variables()
        if not legacy:
            raise
        target = nnx.state(model)
        for name, leaf in legacy.items():
            if name in target:
                raise
            target[name] = nnx.Variable(leaf)
        try:
            restored = read(target)
        except ValueError:
            # The legacy buffers weren't the explanation — report the real mismatch.
            raise mismatch
        print(f"📼 Pre-#105 checkpoint: ignoring {', '.join(legacy)} "
              f"(vestigial, never read by this model).")
        for name in legacy:
            del restored[model_key][name]
        return restored


# Managers whose last save may still be writing to disk (#218).
_PENDING = []


def wait_for_pending_saves():
    """Block until every asynchronous checkpoint write has landed. Called before
    the next save, and on the way out of training — including a SIGTERM."""
    while _PENDING:
        _PENDING.pop().wait_until_finished()


def save_checkpoint(mngr, step, model, optimizer, monitor, sft_active, run_id, wait=True):
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
            monitor_state=ocp.args.JsonSave({
                "ce_history": list(monitor.ce_history),
                "best_ce": monitor.best_ce,
                "best_loss": monitor.best_loss,
                "best_avg_ce": monitor.best_avg_ce,
                "best_val_ce": monitor.best_val_ce,
                "last_improvement_step": monitor.last_improvement_step,
                "sft_active": sft_active,
                "sft_start_step": monitor.sft_start_step,
                "run_id": run_id,  # Save run_id inside checkpoint metadata
                # Samples actually consumed, counted as they were served rather
                # than re-derived (#24). Resume rebuilds the data position from
                # this; computing it as step x BATCH_SIZE would mis-seek exactly
                # the run that needs it — one resumed at a different batch size
                # than it was trained at, whose history spans both.
                "samples_seen": monitor.samples_seen,
            }),
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
    import os
    import signal

    def _raise(signum, frame):
        # Say so in the log. SystemExit prints no traceback, so a TERM'd trainer
        # used to end mid-stream with nothing to distinguish it from a hard kill —
        # three Muon arms of #26 died that way on 2026-09-14 and the cause was
        # unrecorded.
        print(f"🛑 received SIGTERM (pid {os.getpid()}) — exiting cleanly, waiting for pending "
              f"checkpoint writes", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _raise)


def load_or_create_checkpoint(model, optimizer, checkpoint_path, force_new_run=False):
    monitor = LossMonitor()
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    best_mngr = _make_best_manager(checkpoint_path)

    if not force_new_run and mngr.latest_step() is not None:
        latest_step = mngr.latest_step()
        print(f"📖 Loading Orbax checkpoint from step {latest_step}...")
        restored = restore_tolerating_legacy(
            lambda model_target: mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardRestore(model_target),
                    optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
                    monitor_state=ocp.args.JsonRestore(),
                    step=ocp.args.JsonRestore(),
                ),
            ),
            model,
        )

        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        
        start_step = restored["step"] + 1
        m_state = restored["monitor_state"]
        monitor.ce_history = m_state.get("ce_history", [])
        monitor.best_ce = m_state.get("best_ce", float("inf"))
        monitor.best_loss = m_state.get("best_loss", float("inf"))
        monitor.best_avg_ce = m_state.get("best_avg_ce", monitor.best_ce)
        # Absent before #222: the first val probe after resume sets a new best.
        monitor.best_val_ce = m_state.get("best_val_ce", float("inf"))
        monitor.last_improvement_step = m_state.get("last_improvement_step", 0)
        monitor.sft_start_step = m_state.get("sft_start_step", None)
        # Checkpoints written before #24 have no samples_seen; every one of them
        # was trained at BATCH_SIZE=1, so one sample per micro-step is the exact
        # value, not a guess.
        monitor.samples_seen = m_state.get("samples_seen", restored["step"])

        print(f"✅ Resuming from step {start_step} "
              f"({monitor.samples_seen:,} samples consumed)")
        del restored 
        gc.collect()
    else:
        if force_new_run:
            print("🆕 Force New Run specified, starting from scratch...")
        else:
            print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1

    return mngr, best_mngr, monitor, start_step
