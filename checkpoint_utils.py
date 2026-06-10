import os
import gc
import jax.numpy as jnp
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
            except Exception:
                pass
    return None, None

def load_or_create_checkpoint(model, optimizer, checkpoint_path, force_new_run=False):
    monitor = LossMonitor()
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "optimizer", "monitor_state", "step"),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )

    run_id = None
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
        run_id = m_state.get("run_id", None)
        
        print(f"✅ Resuming from step {start_step}")
        del restored 
        gc.collect()
    else:
        if force_new_run:
            print("🆕 Force New Run specified, starting from scratch...")
        else:
            print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1

    return mngr, monitor, start_step, optimizer, run_id
