import os
import gc
from flax import nnx
import orbax.checkpoint as ocp
from monitor import LossMonitor

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

# Sibling subdir of the rolling-latest checkpoints holding the best-CE
# checkpoints. Kept separate so best-retention never evicts the latest.
BEST_SUBDIR = "best"
CHECKPOINT_ITEMS = ("model", "optimizer", "monitor_state", "step")


def _make_best_manager(checkpoint_path):
    """Best-only manager: a sibling 'best/' dir holding the best-CE checkpoints,
    distinct from the rolling-latest manager so its retention can't drop the
    state a resume must load."""
    return ocp.CheckpointManager(
        os.path.join(checkpoint_path, BEST_SUBDIR),
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )


def save_checkpoint(mngr, step, model, optimizer, monitor, sft_active, run_id):
    """Persist the full training state (model + optimizer + monitor + step) under
    `mngr` at `step`, then block until the write lands. Shared by the
    rolling-latest and best-only managers — they use one save schema."""
    mngr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            monitor_state=ocp.args.JsonSave({
                "ce_history": monitor.ce_history,
                "best_ce": monitor.best_ce,
                "best_loss": monitor.best_loss,
                "best_avg_ce": monitor.best_avg_ce,
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
    mngr.wait_until_finished()


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
        monitor.ce_history = m_state.get("ce_history", [])
        monitor.best_ce = m_state.get("best_ce", float("inf"))
        monitor.best_loss = m_state.get("best_loss", float("inf"))
        monitor.best_avg_ce = m_state.get("best_avg_ce", monitor.best_ce)
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
