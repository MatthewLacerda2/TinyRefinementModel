import os
import gc
import time
import threading
import queue

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
from dotenv import load_dotenv

import math

from config import BATCH_SIZE, ACCUMULATION_STEPS, LATENT_DIM, resolve_root
from model import UniversalReasoner
from grad_step import compute_grad_step, apply_grads
from schedules import (
    learning_schedule,
    weight_decay_schedule,
    get_curriculum_weights,
    get_average_curriculum_weights,
    get_curriculum_steps,
)
from metrics_logger import MetricsLogger
from data_loaders import TextDataGenerator, DataMixer

load_dotenv()

LOG_REAL_STEPS = 5
PREFETCH_SIZE = 128

# Abort training after this many consecutive non-finite micro-steps.
MAX_NONFINITE_STREAK = 50

DATA_ROOT = os.environ.get("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)
else:
    print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")


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

def init_model_and_optimizer():
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    return model, optimizer

def create_sft_optimizer(model, old_state=None):
    print("📉 Recreating optimizer with LR reduced to 10% for SFT phase...")

    def sft_lr_schedule(step):
        return learning_schedule(step) * 0.1

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

def setup_data_pipeline(start_step, sft_phase_event, sft_start_step=None):
    print("🚀 Initializing Dynamic Data Phases...")
    pretrain_sources = [
        TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/codeparrot"),
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
                pretrain_mixer.set_weights(get_curriculum_weights(loader_opt_step))
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
    # On resume, trim CSV rows the restored checkpoint will replay; a fresh run
    # (start_step == 1) appends to any existing CSV untouched.
    start_opt_step = start_step // ACCUMULATION_STEPS if start_step > 1 else None
    logger = MetricsLogger(history_file, start_opt_step=start_opt_step)
    step = start_step

    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_grad_norm = 0.0
    t_compute = 0.0
    nonfinite_streak = 0

    try:
        while True:
            batch, doc_boundary = data_queue.get()
            if batch is None:
                break

            t_compute_start = time.time()

            opt_step = step // ACCUMULATION_STEPS
            curr_steps = get_curriculum_steps(opt_step)

            loss, out, grads, grad_norm = compute_grad_step(
                model, batch, jnp.array(step), curr_steps, doc_boundary=doc_boundary
            )

            current_loss = float(loss)
            current_grad_norm = float(grad_norm)

            if not (math.isfinite(current_loss) and math.isfinite(current_grad_norm)):
                # Divergence must be loud and must not poison the optimizer state
                # or the carried hunch (which the grad step already overwrote).
                nonfinite_streak += 1
                print(
                    f"⚠️ Non-finite loss/grad at micro-step {step} "
                    f"(loss={current_loss}, grad_norm={current_grad_norm}, streak={nonfinite_streak}) — skipping update."
                )
                model.hunch_cache.value = jnp.zeros_like(model.hunch_cache.value)
                if nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {MAX_NONFINITE_STREAK} consecutive non-finite micro-steps "
                        f"(last at step {step})."
                    )
                step += 1
                continue
            nonfinite_streak = 0

            apply_grads(optimizer, grads, model)

            t_compute += (time.time() - t_compute_start)

            current_token_loss = float(out.halt_diag.get('token_loss', loss))

            divisor = ACCUMULATION_STEPS * LOG_REAL_STEPS
            accum_loss += current_loss / divisor
            accum_token_loss += current_token_loss / divisor
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
                accum_grad_norm = 0.0
                t_compute = 0.0

            step += 1
    finally:
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()
