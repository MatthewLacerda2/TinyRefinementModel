import os

# Must be set before JAX initializes (imported transitively via trainer).
#
# CUDA's async mempool, not the default preallocated BFC arena. BFC is the faster
# allocator in the abstract (17-22% over `platform`, whose synchronous cudaMalloc
# per buffer it replaced — docs/PERFORMANCE_PLAN.md results log, 2026-06-10), and
# that is still true. It just cannot serve this model on this card: at dim960 the
# working set leaves ~190MB of slack in the arena, while every optimizer step asks
# for a *contiguous* ~596MB param-tree buffer (138.7M x f32). BFC fragments until
# one of those requests cannot be met and the run dies with free memory still on
# the card — a fragmented free list, not a full one. Measured 2026-08-13 launching
# the #157 base run:
#   BFC, batch 2         RESOURCE_EXHAUSTED at opt step 1     (626MiB request)
#   BFC, batch 1         RESOURCE_EXHAUSTED at opt step ~12   (596MiB request)
#   cuda_async, batch 1  past opt step 95, through validation and a checkpoint
# It costs nothing: 4,681 tok/s under cuda_async against 4,583 under BFC — the same
# step time, so this is not a speed/safety trade.
#
# MEM_FRACTION below sizes BFC's preallocation and is inert while cuda_async is
# selected; it is kept so that overriding the allocator back to a preallocating one
# still gets a sane arena rather than JAX's 75% default.
# setdefault keeps both overridable from the shell.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import gc
import argparse
import threading
import multiprocessing as mp

from flax import nnx

from trm.train.optimizers import create_sft_optimizer
from trm.train.trainer import (
    init_model_and_optimizer,
    setup_data_pipeline,
    train_loop,
)
from trm.runtime.run_tracker import RunTracker
from trm.runtime.checkpoints import discover_latest_run, discover_latest_checkpoint_run, load_or_create_checkpoint

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

    data_queue = setup_data_pipeline(start_step, sft_phase_event, monitor.sft_start_step,
                                     samples_seen=monitor.samples_seen or None)

    train_loop(model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, sft_phase_event, run_tracker)
