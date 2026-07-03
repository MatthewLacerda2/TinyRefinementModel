import os

# Must be set before JAX initializes (imported transitively via trainer).
# Preallocated BFC arena: benchmarked 17-22% faster than the old `platform`
# allocator (synchronous cudaMalloc per buffer) — see docs/PERFORMANCE_PLAN.md
# results log, 2026-06-10. The display does not run on this GPU, so claiming
# 85% of VRAM up front is safe. setdefault keeps it overridable from the shell.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import gc
import argparse
import threading
import multiprocessing as mp

from flax import nnx

from optimizers import create_sft_optimizer
from trainer import (
    init_model_and_optimizer,
    setup_data_pipeline,
    train_loop,
)
from run_tracker import RunTracker
from checkpoint_utils import discover_latest_run, discover_latest_checkpoint_run, load_or_create_checkpoint

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train the Dynamic Latent Reasoner")
    parser.add_argument("--new-run", action="store_true", help="Force starting a brand new training run from scratch (ignores existing checkpoints)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Custom folder for Orbax checkpoints")
    args = parser.parse_args()

    # 1. Resolve checkpoint path and run_id to resume (if any)
    checkpoint_run_id = None
    active_checkpoint_path = None

    if args.checkpoint_path is not None:
        active_checkpoint_path = os.path.abspath(args.checkpoint_path)
        # Try to extract run_id from path if it follows runs/run_xxx/checkpoints
        parts = active_checkpoint_path.split(os.sep)
        for part in parts:
            if part.startswith("run_"):
                checkpoint_run_id = part
                break
        print(f"📁 Using custom checkpoint path: {active_checkpoint_path}")
    elif not args.new_run:
        # Auto-discover the latest checkpointed run
        discovered_path, discovered_run_id = discover_latest_checkpoint_run()
        if discovered_path is not None:
            active_checkpoint_path = discovered_path
            checkpoint_run_id = discovered_run_id
            print(f"🔎 Auto-discovered latest checkpointed run: {checkpoint_run_id}")
        else:
            # Fallback to the latest run folder even if it hasn't saved checkpoints yet
            discovered_run_id = discover_latest_run()
            if discovered_run_id is not None:
                checkpoint_run_id = discovered_run_id
                active_checkpoint_path = os.path.join("runs", checkpoint_run_id, "checkpoints")
                print(f"🔎 Auto-discovered latest run (no checkpoints yet): {checkpoint_run_id}")

    # 2. Start/Resume Run Tracker session
    run_tracker = RunTracker()
    run_tracker.start_session(run_id=checkpoint_run_id)
    if active_checkpoint_path is None:
        active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")

    active_checkpoint_path = os.path.abspath(active_checkpoint_path)

    sft_phase_event = threading.Event()

    model, optimizer = init_model_and_optimizer()

    mngr, best_mngr, monitor, start_step = load_or_create_checkpoint(
        model, optimizer, active_checkpoint_path, force_new_run=args.new_run
    )

    # Set event if resuming in SFT phase
    if monitor.sft_start_step is not None:
        print(f"🔄 Resuming in SFT phase (started at step {monitor.sft_start_step})")
        sft_phase_event.set()

        old_state = nnx.state(optimizer)
        del optimizer
        gc.collect()

        optimizer = create_sft_optimizer(model, old_state)
        del old_state
        gc.collect()

    data_queue = setup_data_pipeline(start_step, sft_phase_event, monitor.sft_start_step)

    train_loop(model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, sft_phase_event, run_tracker)
