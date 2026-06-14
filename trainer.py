import os
import gc
import time
import threading
import queue

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from dotenv import load_dotenv

import math

from config import (
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    LATENT_DIM,
    MAX_SEQ_LEN,
    PAD_TOKEN_ID,
    resolve_root,
)
from model import UniversalReasoner
from checkpoint_utils import save_checkpoint
from grad_step import compute_grad_step, apply_grads
from schedules import (
    learning_schedule,
    weight_decay_schedule,
    get_curriculum_weights,
    get_average_curriculum_weights,
    sample_reasoning_depth,
)
from metrics_logger import MetricsLogger
from data_loaders import TextDataGenerator, DataMixer

load_dotenv()

LOG_REAL_STEPS = 5
PREFETCH_SIZE = 128

# Abort training after this many consecutive non-finite micro-steps.
MAX_NONFINITE_STREAK = 50

# Held-out validation: the same fixed batches, scored the same deterministic way,
# every VAL_EVERY_OPT_STEPS optimizer steps. Train CE cannot see overfitting or
# data drift; this curve is the one decisions should read.
VAL_EVERY_OPT_STEPS = 64
VAL_BATCHES = 4
VAL_FIXED_DEPTH = 4
# Far past any plausible training consumption (an 8k-opt-step run consumes
# under 1M fineweb samples; fineweb holds 4.3M) so the slice stays held out.
VAL_SKIP_SAMPLES = 3_000_000

DATA_ROOT = os.environ.get("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)
else:
    print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")


@nnx.jit
def _val_ce_sums(model, batch):
    """Masked CE sums over both windows, mirroring the training segment structure
    (window 1 fresh, window 2 on the carried hunch) at a fixed depth."""
    seq1_in, seq1_out = batch[:, :MAX_SEQ_LEN], batch[:, 1:MAX_SEQ_LEN + 1]
    seq2_in, seq2_out = batch[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN], batch[:, MAX_SEQ_LEN + 1:2 * MAX_SEQ_LEN + 1]
    out1 = model(seq1_in, max_steps=VAL_FIXED_DEPTH, training=False, should_refresh=True)
    out2 = model(seq2_in, max_steps=VAL_FIXED_DEPTH, training=False, should_refresh=False)
    total = jnp.array(0.0)
    count = jnp.array(0)
    for logits, targets in ((out1.logits, seq1_out), (out2.logits, seq2_out)):
        mask = targets != PAD_TOKEN_ID
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        total += jnp.sum(ce * mask)
        count += jnp.sum(mask)
    return total, count


class ValidationProbe:
    """Loads VAL_BATCHES fixed held-out batches once, then scores them on demand.
    Restores the training stream's carried hunch afterwards, so validating never
    perturbs training."""

    def __init__(self):
        self._batches = None

    def _load(self):
        gen = TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu")
        gen.skip_count = VAL_SKIP_SAMPLES
        batches = []
        while len(batches) < VAL_BATCHES:
            batch, _ = gen.get_batch(BATCH_SIZE)
            if batch is None:
                break
            batches.append(batch)
        if not batches:
            print("⚠️ Validation disabled: no held-out data available past the skip range.")
        return batches

    def run(self, model):
        if self._batches is None:
            self._batches = self._load()
        if not self._batches:
            return None
        saved_hunch = model.hunch_cache.value
        total, count = 0.0, 0
        for batch in self._batches:
            model.hunch_cache.value = jnp.zeros_like(saved_hunch)
            ce_sum, ce_count = _val_ce_sums(model, batch)
            total += float(ce_sum)
            count += int(ce_count)
        model.hunch_cache.value = saved_hunch
        return total / max(count, 1)


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

def train_loop(model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, sft_phase_event, run_tracker):
    history_file = os.path.join(run_tracker.run_dir, "metrics.csv")
    # On resume, trim CSV rows the restored checkpoint will replay; a fresh run
    # (start_step == 1) appends to any existing CSV untouched.
    start_opt_step = start_step // ACCUMULATION_STEPS if start_step > 1 else None
    logger = MetricsLogger(history_file, start_opt_step=start_opt_step)
    val_probe = ValidationProbe()
    step = start_step

    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_grad_norm = 0.0
    accum_depth = 0.0
    t_compute = 0.0
    nonfinite_streak = 0
    # Latest held-out CE from the validation probe, carried so the (less frequent)
    # logging block can record it. None until the first probe fires.
    latest_val_ce = None

    try:
        while True:
            batch, doc_boundary = data_queue.get()
            if batch is None:
                break

            t_compute_start = time.time()

            depth = sample_reasoning_depth(step)

            loss, out, grads, grad_norm = compute_grad_step(
                model, batch, jnp.array(step), depth, doc_boundary=doc_boundary
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

            current_token_loss = float(out.diag.get('token_loss', loss))

            divisor = ACCUMULATION_STEPS * LOG_REAL_STEPS
            accum_loss += current_loss / divisor
            accum_token_loss += current_token_loss / divisor
            accum_grad_norm += current_grad_norm / divisor
            accum_depth += depth / divisor

            # Validation probe fires on its own cadence at the optimizer-step
            # boundary (every ACCUMULATION_STEPS micro-steps), independent of the
            # logging block — nesting it inside logging multiplied the effective
            # interval by LOG_REAL_STEPS.
            if (step + 1) % ACCUMULATION_STEPS == 0:
                opt_step = (step + 1) // ACCUMULATION_STEPS
                if opt_step % VAL_EVERY_OPT_STEPS == 0:
                    val_ce = val_probe.run(model)
                    if val_ce is not None:
                        latest_val_ce = val_ce
                        print(f"🧪 [Validation] Opt Step {opt_step} | held-out CE: {val_ce:.4f}")

            if (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0:
                opt_step = (step + 1) // ACCUMULATION_STEPS

                logger.log(
                    opt_step,
                    float(accum_token_loss),
                    float(accum_loss),
                    out,
                    t_compute,
                    grad_norm_avg=float(accum_grad_norm),
                    seg1_ce=float(out.diag.get('seg1_ce', 0)),
                    depth_avg=float(accum_depth),
                    val_ce=latest_val_ce,
                )
                # Logged once; clear so it isn't re-attributed to later opt-steps.
                latest_val_ce = None

                if not sft_phase_event.is_set():
                    curr_weights = get_curriculum_weights(opt_step)
                    print(
                        f"📚 [Curriculum] Opt Step: {opt_step} | Avg Sampled Depth: {accum_depth:.2f} | "
                        f"Weights (Web/Code/Math): {curr_weights[0]:.3f} / {curr_weights[1]:.3f} / {curr_weights[2]:.3f}"
                    )
                else:
                    print(f"💬 [SFT Phase] Opt Step: {opt_step} | Avg Sampled Depth: {accum_depth:.2f} | Weights (Chat/Web/Code/Math): [0.70, 0.15, 0.10, 0.05]")

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

                # Rolling-latest: always persist the true latest state so a resume
                # continues from where training actually left off (max_to_keep=3
                # by recency). The best-CE state is preserved separately under
                # best_mngr so it can never be evicted by the rolling retention.
                sft_active = sft_phase_event.is_set()
                save_checkpoint(mngr, step, model, optimizer, monitor, sft_active, run_tracker.run_id)
                if monitor.is_new_best:
                    save_checkpoint(best_mngr, step, model, optimizer, monitor, sft_active, run_tracker.run_id)

                accum_loss = 0.0
                accum_token_loss = 0.0
                accum_grad_norm = 0.0
                accum_depth = 0.0
                t_compute = 0.0

            step += 1
    finally:
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()
