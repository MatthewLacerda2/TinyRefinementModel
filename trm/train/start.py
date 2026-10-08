import os

# Must be set before JAX initializes (imported transitively via trainer).
#
# CUDA's async mempool, not the default preallocated BFC arena. BFC is the faster
# allocator in the abstract (17-22% over `platform`, whose synchronous cudaMalloc
# per buffer it replaced — measured 2026-06-10 in ee8b170), and
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
# setdefault keeps both overridable from the shell. These lines are production's
# environment as instruments are checked against it: one that sets either key to
# anything else must say why (tests/apparatus/test_instrument_environment.py, #166).
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import argparse
import multiprocessing as mp

from trm.runtime.checkpoints import (
    discover_latest_checkpoint_run,
    discover_latest_run,
    exit_cleanly_on_sigterm,
    load_or_create_checkpoint,
)
from trm.runtime.rewind import unresumable
from trm.runtime.run_budget import horizon_mismatch, with_recorded_budget
from trm.runtime.run_tracker import RunTracker
from trm.settings import CONFIG
from trm.train.loop import train_loop
from trm.train.schedules import Schedules
from trm.train.trainer import DATA_ROOT, init_model_and_optimizer, setup_data_pipeline

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train the model")
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

    # 2. The run's Config: this process's knobs, read once (trm.settings), with a
    # resumed run's own token budget when the launch set none, so the LR horizon
    # comes from the run and not from whichever shell relaunched it (#197). Everything
    # below is handed this one object (#475).
    config = with_recorded_budget(CONFIG, active_checkpoint_path)
    if config.TRAIN_TOKEN_BUDGET != CONFIG.TRAIN_TOKEN_BUDGET:
        # The number itself lands in the LR horizon banner a few lines into startup;
        # what this says is where it came from.
        print("🗓️ Recovered TRAIN_TOKEN_BUDGET from the resumed run's own metadata (#197)")
    # Resolved before anything is written: a mixture that does not parse or a budget
    # inside the warmup refuses here, not after a run folder exists.
    decay_steps = Schedules.of(config).decay_steps

    # A checkpoint that cannot be resumed (the retired SFT phase, #323; a resume
    # state ResumeState refuses, #477) is refused here, before the session below
    # appends to run_metadata.json (#505). load_or_create_checkpoint repeats the
    # checks as a backstop.
    if active_checkpoint_path is not None and not args.new_run:
        why = unresumable(active_checkpoint_path, config.ACCUMULATION_STEPS)
        if why:
            raise SystemExit(why)

    # 3. Start/Resume Run Tracker session
    run_tracker = RunTracker(config)
    run_tracker.start_session(run_id=checkpoint_run_id)

    # A budget set explicitly to something other than the run's is not replaced
    # above; it is refused here, where the run directory is known (#197).
    complaint = horizon_mismatch(run_tracker.run_dir, decay_steps)
    if complaint is not None:
        raise SystemExit(f"❌ {complaint}")

    if active_checkpoint_path is None:
        assert run_tracker.run_dir is not None  # start_session set it
        active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")

    active_checkpoint_path = os.path.abspath(active_checkpoint_path)

    model, optimizer = init_model_and_optimizer(config)

    mngr, best_mngr, monitor, start_step = load_or_create_checkpoint(
        config, model, optimizer, active_checkpoint_path, force_new_run=args.new_run
    )

    data_queue = setup_data_pipeline(config, start_step, samples_seen=monitor.samples_seen or None,
                                     data_state=monitor.data_state)

    exit_cleanly_on_sigterm()  # so a TERM waits for an in-flight checkpoint write (#218)
    train_loop(config, model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, run_tracker, DATA_ROOT)
