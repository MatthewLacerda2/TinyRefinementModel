"""The training loop and its data pipeline. The pieces the loop *uses* live in
their own modules: held-out scoring in validation.py, the optimizer chains in
optimizers.py, schedules and mixture policies in schedules.py."""

import os
import gc
import time
import threading
import queue

import jax
import jax.numpy as jnp
from flax import nnx
from dotenv import load_dotenv

import math

from config import (
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    LATENT_DIM,
    MODEL_ARCH,
    REFINER_ENCODER_LAYERS,
    MAX_STEPS_LIMIT,
    DATA_SEED,
    MODEL_SEED,
    TOKENS_PER_OPT_STEP,
    TRAIN_TOKEN_BUDGET,
    resolve_root,
)
from model import UniversalReasoner
from checkpoint_utils import save_checkpoint
from grad_step import compute_grad_step, apply_grads, grad_zero_fractions, dense_zero_frac_max
from optimizers import optimizer_chain, create_sft_optimizer
from schedules import (
    CURRICULUM_START_WEIGHTS,
    SFT_MIX_WEIGHTS,
    DECAY_STEPS,
    WARMUP_STEPS,
    get_curriculum_weights,
    get_average_curriculum_weights,
    sample_reasoning_depth,
)
from validation import ValidationProbe
from metrics_logger import MetricsLogger
from data_loaders import TextDataGenerator, DataMixer

load_dotenv()

LOG_REAL_STEPS = 5
PREFETCH_SIZE = 128

# Abort training after this many consecutive non-finite micro-steps.
MAX_NONFINITE_STREAK = 50

# Cadences, in optimizer steps, both firing at the opt-step boundary — NOT
# nested in the logging block (nesting would multiply the interval by
# LOG_REAL_STEPS, the bug that hid the probe). The full-state checkpoint save
# blocks the loop (wait_until_finished), so it must stay rare.
VAL_EVERY_OPT_STEPS = 64
CHECKPOINT_EVERY_OPT_STEPS = 64

DATA_ROOT = os.environ.get("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)
else:
    print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")


def _param_count(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def split_samples(total_samples, weights):
    """Split a sample count across sources by mixture weight.

    The unit that resume correctness hangs on: TextDataGenerator's skip_count is
    in SAMPLES, while every step counter in this trainer is in MICRO-STEPS, and
    each micro-step draws BATCH_SIZE samples across the mixer (#24). Callers pass
    the SAMPLE total — read from the checkpoint, not re-derived from steps — so
    a run that resumes under a different batch size than it was trained at still
    seeks to the right place. Getting this wrong is silent either way: too small
    and the model re-trains on data it already saw, too large and it skips a
    slice of corpus it never read. No crash, no warning, just a bad run.

    Per-source truncation is deliberate: skip_count is an integer sample offset,
    so the total can fall short by at most one sample per source.
    """
    return [int(total_samples * w) for w in weights]


def samples_from_micro_steps(micro_steps, weights, batch_size=BATCH_SIZE):
    """split_samples for the case with no recorded sample count — a pre-#24
    checkpoint, or a fresh run. Converts micro-steps at the given batch size."""
    return split_samples(micro_steps * batch_size, weights)


def init_model_and_optimizer():
    if MODEL_ARCH == "refiner":
        # Imported lazily so the baseline path never touches Plan A code.
        from plan_a_trainer import RefinerForTraining
        print(f"🚀 Initializing Plan A CausalRefiner "
              f"(Dim={LATENT_DIM}, encoder_layers={REFINER_ENCODER_LAYERS}, max_depth={MAX_STEPS_LIMIT})...")
        model = RefinerForTraining(LATENT_DIM, nnx.Rngs(MODEL_SEED))
    else:
        print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
        model = UniversalReasoner(LATENT_DIM, nnx.Rngs(MODEL_SEED))

    print(f"📐 Architecture '{MODEL_ARCH}': {_param_count(model) / 1e6:.1f}M parameters "
          f"(MODEL_SEED={MODEL_SEED}, DATA_SEED={DATA_SEED})")
    # The resolved LR horizon must be visible at launch (#83): an anneal that
    # bottoms out before the budget ends is undertraining masquerading as an
    # architecture problem.
    budget_note = (f"TRAIN_TOKEN_BUDGET={TRAIN_TOKEN_BUDGET:,}" if TRAIN_TOKEN_BUDGET is not None
                   else "TRAIN_TOKEN_BUDGET unset — historical default")
    print(f"🗓️ LR horizon: DECAY_STEPS={DECAY_STEPS:,} opt steps "
          f"(warmup {WARMUP_STEPS:,}) ≈ {DECAY_STEPS * TOKENS_PER_OPT_STEP / 1e9:.2f}B "
          f"target tokens ({budget_note})")
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    return model, optimizer

def setup_data_pipeline(start_step, sft_phase_event, sft_start_step=None, samples_seen=None):
    print("🚀 Initializing Dynamic Data Phases...")
    pretrain_sources = [
        TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/codeparrot"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/finemath"),
    ]
    pretrain_mixer = DataMixer(pretrain_sources, CURRICULUM_START_WEIGHTS)

    sft_sources = [
        TextDataGenerator(f"{DATA_ROOT}/chat/ultrachat"),
        pretrain_sources[0],
        pretrain_sources[1],
        pretrain_sources[2],
    ]
    sft_mixer = DataMixer(sft_sources, SFT_MIX_WEIGHTS)

    if start_step > 1:
        start_opt_step = start_step // ACCUMULATION_STEPS
        # Prefer the recorded sample count over re-deriving it from micro-steps
        # (#24): only the recorded figure survives a change in BATCH_SIZE between
        # the run that wrote the checkpoint and the one resuming it.
        if sft_start_step is None or start_step < sft_start_step:
            avg_weights = get_average_curriculum_weights(start_opt_step)
            skips = (split_samples(samples_seen, avg_weights) if samples_seen is not None
                     else samples_from_micro_steps(start_step - 1, avg_weights))
            for gen, skip in zip(pretrain_sources, skips):
                gen.skip_count = skip
        else:
            # 1. Catch up pretrain sources to the point where pretraining ended.
            # The pretrain/SFT split is still counted in micro-steps: samples_seen
            # is a single total and does not say where the phase boundary fell.
            sft_start_opt_step = sft_start_step // ACCUMULATION_STEPS
            avg_weights = get_average_curriculum_weights(sft_start_opt_step)
            skips = samples_from_micro_steps(sft_start_step - 1, avg_weights)
            for gen, skip in zip(pretrain_sources, skips):
                gen.skip_count = skip

            # 2. Add SFT usage for all blended sources (Chat + Replay)
            sft_skips = samples_from_micro_steps(start_step - sft_start_step, SFT_MIX_WEIGHTS)
            for gen, skip in zip(sft_sources, sft_skips):
                gen.skip_count += skip

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
    val_probe = ValidationProbe(DATA_ROOT)
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

            # Count consumed samples as they are consumed (#24). Deriving this at
            # save time as step x BATCH_SIZE would be wrong for exactly the run
            # that needs it most: one resumed at a different batch size than it
            # was trained at, whose history spans both.
            monitor.samples_seen += batch.shape[0]

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
                model.hunch_cache[...] = jnp.zeros_like(model.hunch_cache[...])
                if nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {MAX_NONFINITE_STREAK} consecutive non-finite micro-steps "
                        f"(last at step {step})."
                    )
                step += 1
                continue
            nonfinite_streak = 0

            # Underflow instrument (#82) sampling happens BEFORE the update:
            # apply_grads donates the grad buffers (#128), so this micro-step's
            # raw grads are unreadable afterwards. Same tensors either way —
            # pre-accumulation grads are what the instrument wants.
            if (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0:
                zero_fracs = {k: float(v) for k, v in grad_zero_fractions(grads).items()}
                zero_frac_dense = dense_zero_frac_max(zero_fracs)

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

                # Rolling-latest: persist the true latest state on its own cadence
                # so a resume continues from where training actually left off
                # (max_to_keep=3 by recency). Kept out of the logging block — the
                # full-state save blocks, so it must stay rare. The best-CE state
                # is saved separately below, where monitor.is_new_best is computed.
                if opt_step % CHECKPOINT_EVERY_OPT_STEPS == 0:
                    save_checkpoint(mngr, step, model, optimizer, monitor,
                                    sft_phase_event.is_set(), run_tracker.run_id)

            if (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0:
                opt_step = (step + 1) // ACCUMULATION_STEPS

                # Underflow instrument (#82): zero_fracs / zero_frac_dense were
                # sampled just before apply_grads above (donation makes the raw
                # grads unreadable here). Interpretation caveats (time_embed row
                # sparsity, structural zeros behind the zero-init down_proj early
                # in training) live on grad_zero_fractions itself.
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
                    zero_frac_dense_max=zero_frac_dense,
                )
                # Logged once; clear so it isn't re-attributed to later opt-steps.
                latest_val_ce = None

                print(
                    f"🧊 [ZeroGrad] dense max: {zero_frac_dense:.4f} | "
                    + " ".join(f"{k}={v:.3f}" for k, v in zero_fracs.items())
                )

                if not sft_phase_event.is_set():
                    curr_weights = get_curriculum_weights(opt_step)
                    print(
                        f"📚 [Curriculum] Opt Step: {opt_step} | Avg Sampled Depth: {accum_depth:.2f} | "
                        f"Weights (Web/Code/Math): {curr_weights[0]:.3f} / {curr_weights[1]:.3f} / {curr_weights[2]:.3f}"
                    )
                else:
                    sft_w = " / ".join(f"{w:.2f}" for w in SFT_MIX_WEIGHTS)
                    print(f"💬 [SFT Phase] Opt Step: {opt_step} | Avg Sampled Depth: {accum_depth:.2f} | Weights (Chat/Web/Code/Math): {sft_w}")

                # Periodically update session duration to capture active timings
                run_tracker.update_session_duration()

                plateaued = monitor.push(opt_step, float(accum_token_loss), float(accum_loss))
                if plateaued:
                    if sft_phase_event.is_set():
                        print("🛑 Training halted: No improvement in CE during SFT phase.")
                        break

                    print("\n" + "🔄"*30)
                    print("🔄 CE Plateau Detected! Triggering SFT Chat Phase and decaying Learning Rate!")
                    print("🔄"*30 + "\n")
                    sft_phase_event.set()
                    monitor.sft_start_step = step

                    # OOM-critical ordering (#30): pull the optimizer moments to
                    # host, then free the old optimizer IN THIS SCOPE, before
                    # nnx.Optimizer eagerly allocates the new full-size mu/nu on
                    # the GPU — otherwise both states coexist for an instant, a
                    # ~2x spike that OOM'd the 6GB card. (This block must stay
                    # inline: a helper's `del` cannot release the loop's local
                    # reference.) Momentum survives via the host copy.
                    old_state = jax.device_get(nnx.state(optimizer))
                    del optimizer
                    gc.collect()
                    optimizer = create_sft_optimizer(model, old_state)
                    del old_state
                    gc.collect()

                    monitor.reset_for_new_phase(opt_step)

                # Best-CE checkpoint: saved here where monitor.is_new_best is
                # computed (over the windowed accumulators), in a sibling dir so
                # best-retention and the rolling-latest retention never evict each
                # other. The rolling-latest save runs separately, above, on
                # CHECKPOINT_EVERY_OPT_STEPS at the opt-step boundary.
                if monitor.is_new_best:
                    save_checkpoint(best_mngr, step, model, optimizer, monitor,
                                    sft_phase_event.is_set(), run_tracker.run_id)

                accum_loss = 0.0
                accum_token_loss = 0.0
                accum_grad_norm = 0.0
                accum_depth = 0.0
                t_compute = 0.0

            step += 1
    finally:
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()
