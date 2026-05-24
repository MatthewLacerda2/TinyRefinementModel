import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import time
import threading
import queue
import multiprocessing as mp
from dotenv import load_dotenv
from layers import (
    LATENT_DIM,
    NUM_BLOCKS,
    SHARED_SLOTS,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    MAX_STEPS_LIMIT,
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
    NUM_HEADS,
    NUM_GROUPS,
)
from model import UniversalReasoner
from train_local import (
    compute_grad_step,
    apply_grads,
)
from schedules import (
    learning_schedule,
    weight_decay_schedule,
)

from metrics_logger import LossMonitor, MetricsLogger
from run_tracker import RunTracker
from checkpoint_utils import discover_latest_checkpoint_run, load_or_create_checkpoint
import json
import datetime
import subprocess
import sys

load_dotenv()

LOG_REAL_STEPS = 5
PREFETCH_SIZE = 128

DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", ""))
CHECKPOINT_ROOT = os.path.abspath(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))

if not DATA_ROOT:
    print(f"⚠️ Warning: DATA_ROOT is not set. Data loading will likely fail unless provided via environment.")

from data_loaders import TextDataGenerator, DataMixer

def init_model_and_optimizer():
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)
    
    return model, optimizer

def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)

optimizer_chain = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_schedule,
            weight_decay=weight_decay_schedule,
            mask=weight_decay_mask,
        ),
    ),
    every_k_schedule=ACCUMULATION_STEPS,
    use_grad_mean=True
)

def create_sft_optimizer(model, old_state=None):
    import gc
    print("📉 Recreating optimizer with 10x LR penalty for SFT phase...")
    
    sft_lr_schedule = lambda step: learning_schedule(step) * 0.1
    
    sft_chain = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=sft_lr_schedule,
                weight_decay=weight_decay_schedule,
                mask=weight_decay_mask,
            ),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True
    )

    new_opt = nnx.Optimizer(model, sft_chain, wrt=nnx.Param)
    if old_state is not None:
        nnx.update(new_opt, old_state)
        
    gc.collect()
    return new_opt

def get_curriculum_weights(loader_step):
    CURRICULUM_STEPS = 10000.0
    step = float(loader_step)
    if step >= CURRICULUM_STEPS:
        return [0.35, 0.40, 0.25]
    fraction = step / CURRICULUM_STEPS
    w_web = 0.85 - 0.50 * fraction
    w_code = 0.10 + 0.30 * fraction
    w_math = 0.05 + 0.20 * fraction
    return [w_web, w_code, w_math]

def get_average_curriculum_weights(loader_step):
    step = float(loader_step)
    CURRICULUM_STEPS = 10000.0
    if step == 0:
        return [0.85, 0.10, 0.05]
    if step >= CURRICULUM_STEPS:
        curriculum_fraction = CURRICULUM_STEPS / step
        post_fraction = 1.0 - curriculum_fraction
        avg_web = 0.60 * curriculum_fraction + 0.35 * post_fraction
        avg_code = 0.25 * curriculum_fraction + 0.40 * post_fraction
        avg_math = 0.15 * curriculum_fraction + 0.25 * post_fraction
        return [avg_web, avg_code, avg_math]
    else:
        curr = get_curriculum_weights(step)
        return [
            (0.85 + curr[0]) / 2.0,
            (0.10 + curr[1]) / 2.0,
            (0.05 + curr[2]) / 2.0
        ]

def get_curriculum_steps(train_opt_step):
    if train_opt_step < 1000:
        return 1
    elif train_opt_step < 4000:
        return 2
    elif train_opt_step < 8000:
        return 4
    else:
        return 8

def setup_data_pipeline(start_step, sft_phase_event, sft_start_step=None):
    print("🚀 Initializing Dynamic Data Phases...")
    pretrain_sources = [
        TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/code_instructions"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/finemath"),
    ]
    pretrain_weights = [0.85, 0.10, 0.05]
    pretrain_mixer = DataMixer(pretrain_sources, pretrain_weights)
    
    sft_sources = [
        TextDataGenerator(f"{DATA_ROOT}/chat/ultrachat"),
        pretrain_sources[0],
        pretrain_sources[1],
        pretrain_sources[2],
    ]
    sft_weights = [0.70, 0.15, 0.10, 0.05]
    sft_mixer = DataMixer(sft_sources, sft_weights)

    if start_step > 1:
        start_opt_step = start_step // ACCUMULATION_STEPS
        if sft_start_step is None or start_step < sft_start_step:
            total_pretrain_seen = (start_step - 1)
            avg_weights = get_average_curriculum_weights(start_opt_step)
            for gen, weight in zip(pretrain_sources, avg_weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            # 1. Catch up pretrain sources to the point where pretraining ended
            sft_start_opt_step = sft_start_step // ACCUMULATION_STEPS
            total_pre_pretrain_seen = (sft_start_step - 1)
            avg_weights = get_average_curriculum_weights(sft_start_opt_step)
            for gen, weight in zip(pretrain_sources, avg_weights):
                gen.skip_count = int(total_pre_pretrain_seen * weight)
            
            # 2. Add SFT usage for all blended sources (Chat + Replay)
            total_sft_seen = (start_step - sft_start_step)
            for gen, weight in zip(sft_sources, sft_weights):
                gen.skip_count += int(total_sft_seen * weight)

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        loader_step = start_step
        while True:
            loader_opt_step = loader_step // ACCUMULATION_STEPS
            if not sft_phase_event.is_set():
                pretrain_mixer.weights = get_curriculum_weights(loader_opt_step)
                res = pretrain_mixer.get_batch(BATCH_SIZE)
            else:
                res = sft_mixer.get_batch(BATCH_SIZE)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)
            loader_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event, run_tracker):
    history_file = os.path.join(run_tracker.run_dir, "metrics.csv")
    logger = MetricsLogger(history_file)
    step = start_step
    
    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_forget_cost = 0.0
    accum_grad_norm = 0.0
    t_compute = 0.0
    
    try:
        while True:
            batch, should_truncate = data_queue.get() 
            if batch is None: 
                break
            
            t_compute_start = time.time()
            
            opt_step = step // ACCUMULATION_STEPS
            curr_steps = get_curriculum_steps(opt_step)
            
            loss, out, grads, grad_norm = compute_grad_step(
                model, batch, jnp.array(step), curr_steps, should_truncate=should_truncate
            )
            
            apply_grads(optimizer, grads, model)
            
            t_compute += (time.time() - t_compute_start)

            current_loss = float(loss)
            current_token_loss = float(out.halt_diag.get('token_loss', loss))
            current_forget = float(out.forget_cost)
            current_grad_norm = float(grad_norm)

            divisor = ACCUMULATION_STEPS * LOG_REAL_STEPS
            accum_loss += current_loss / divisor
            accum_token_loss += current_token_loss / divisor
            accum_forget_cost += current_forget / divisor
            accum_grad_norm += current_grad_norm / divisor
                
            if (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0:
                opt_step = (step + 1) // ACCUMULATION_STEPS
                
                logger.log(
                    opt_step, 
                    float(accum_token_loss),
                    float(accum_loss),
                    out,
                    t_compute,
                    grad_norm_avg=float(accum_grad_norm),
                    first_ce=float(out.halt_diag.get('ce1', 0))
                )
                
                if not sft_phase_event.is_set():
                    curr_weights = get_curriculum_weights(opt_step)
                    print(
                        f"📚 [Curriculum] Opt Step: {opt_step} | Reasoning Steps: {curr_steps} | "
                        f"Weights (Web/Code/Math): {curr_weights[0]:.3f} / {curr_weights[1]:.3f} / {curr_weights[2]:.3f}"
                    )
                else:
                    print(f"💬 [SFT Phase] Opt Step: {opt_step} | Reasoning Steps: {curr_steps} | Weights (Chat/Web/Code/Math): [0.70, 0.15, 0.10, 0.05]")
                
                # Periodically update session duration to capture active timings
                run_tracker.update_session_duration()
                
                if monitor.push(opt_step, float(accum_token_loss), float(accum_loss)): 
                    if not sft_phase_event.is_set():
                        sft_phase_event.set()
                        monitor.sft_start_step = step
                        
                        import gc
                        old_state = nnx.state(optimizer)
                        del optimizer
                        gc.collect()

                        optimizer = create_sft_optimizer(model, old_state)
                        del old_state
                        gc.collect()
                        
                        print("\n" + "🔄"*30)
                        print("🔄 CE Plateau Detected! Triggering SFT Chat Phase and decaying Learning Rate!")
                        print("🔄"*30 + "\n")
                        
                        monitor.ce_history = []
                        monitor.best_ce = float("inf")
                        monitor.best_loss = float("inf")
                        monitor.best_avg_ce = float("inf")
                        monitor.last_improvement_step = opt_step
                    else:
                        print("🛑 Training halted: No improvement in CE during SFT phase.")
                        break

                if monitor.is_new_best:
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
                                "sft_active": sft_phase_event.is_set(),
                                "sft_start_step": monitor.sft_start_step,
                                "run_id": run_tracker.run_id,  # Save run_id inside checkpoint metadata
                            }),
                            step=ocp.args.JsonSave(step),
                        ),
                    )
                    mngr.wait_until_finished()
                
                accum_loss = 0.0
                accum_token_loss = 0.0
                accum_forget_cost = 0.0
                accum_grad_norm = 0.0
                t_compute = 0.0

            step += 1
    finally:
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()

if __name__ == "__main__":
    import argparse

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
    
    # 2. Start/Resume Run Tracker session
    run_tracker = RunTracker()
    if checkpoint_run_id is None:
        # Starting a brand new run
        run_tracker.start_session(run_id=None)
        if active_checkpoint_path is None:
            active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")
    else:
        # Resuming existing run
        run_tracker.start_session(run_id=checkpoint_run_id)
        if active_checkpoint_path is None:
            active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")

    sft_phase_event = threading.Event()

    model, optimizer = init_model_and_optimizer()
    
    mngr, monitor, start_step, optimizer, checkpoint_run_id_from_meta = load_or_create_checkpoint(
        model, optimizer, active_checkpoint_path, force_new_run=args.new_run
    )
    
    # Set event if resuming in SFT phase
    if getattr(monitor, "sft_start_step", None) is not None:
        print(f"🔄 Resuming in SFT phase (started at step {monitor.sft_start_step})")
        sft_phase_event.set()
        
        import gc
        old_state = nnx.state(optimizer)
        del optimizer
        gc.collect()
        
        optimizer = create_sft_optimizer(model, old_state)
        del old_state
        gc.collect()

    data_queue = setup_data_pipeline(start_step, sft_phase_event, getattr(monitor, "sft_start_step", None))
    
    train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event, run_tracker)