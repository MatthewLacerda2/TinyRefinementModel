"""The training loop and its data pipeline. The pieces the loop *uses* live in
their own modules: held-out scoring in validation.py, the optimizer chains in
optimizers.py, schedules and mixture policies in schedules.py."""

import os
import math
import time
import threading
import queue

import jax
import jax.numpy as jnp
from flax import nnx
from dotenv import load_dotenv

from trm.config import (
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    LATENT_DIM,
    MODEL_ARCH,
    REFINER_ENCODER_LAYERS,
    MAX_STEPS_LIMIT,
    DATA_SEED,
    MODEL_SEED,
    PLAIN_LAYERS,
    TOKENS_PER_OPT_STEP,
    TRAIN_TOKEN_BUDGET,
    TRM_OPTIMIZER,
    MUON_LR_MULT,
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    WEIGHT_DECAY,
    CLIP_NORM,
    MUON_BETA,
    MUON_NS_STEPS,
    resolve_root,
)
from trm.model import build_model
from trm.runtime.layout import CHECKPOINT_EVERY_OPT_STEPS, LOG_REAL_STEPS, VAL_EVERY_OPT_STEPS
from trm.runtime.checkpoints import (make_milestone_manager, milestone_due, save_checkpoint,
                                     wait_for_pending_saves)
from trm.train.grad_step import (compute_grad_step, apply_grads, applied_gradient_stats, grad_zero_fractions,
                                 dense_zero_frac_max)
from trm.train.grad_guard import GradientNormGuard
from trm.train.loss_scale import DynamicLossScale
from trm.train.optimizers import optimizer_chain
from trm.train.schedules import (
    CURRICULUM_START_WEIGHTS,
    DECAY_STEPS,
    SCHEDULE_HORIZONS,
    WARMUP_STEPS,
    get_curriculum_weights,
    get_average_curriculum_weights,
    sample_reasoning_depth,
)
from trm.train.validation import ValidationProbe
from trm.runtime.metrics import MetricsLogger
from trm.data.loaders import TextDataGenerator, DataMixer

load_dotenv()

PREFETCH_SIZE = 128

# Abort training after this many consecutive non-finite micro-steps.
MAX_NONFINITE_STREAK = 50

# Opt steps between "still plateaued" notices. `monitor.plateaued` stays True on
# every logging step until held-out CE improves, so an unthrottled notice would
# print on every one of them and bury the rest of the log.
PLATEAU_NOTICE_EVERY = 200

DATA_ROOT = os.environ.get("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)


# Data sources, in mixer order. One list feeds both the loaders and the `mix`
# column of metrics.csv, so the recorded mixture cannot name a source other than
# the one that was served.
PRETRAIN_SOURCES = ("pretrain/fineweb-edu", "pretrain/codeparrot", "pretrain/finemath")


def mixture_label(sources, weights):
    """`fineweb-edu=0.600 codeparrot=0.250 finemath=0.150` — the mixture a CE was
    measured on, readable without today's curriculum constants (#186)."""
    assert len(sources) == len(weights), "a mixture must name every source it weights"
    return " ".join(f"{src.rsplit('/', 1)[-1]}={w:.3f}" for src, w in zip(sources, weights))


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


class LogWindow:
    """Means over one logging window, divided by the micro-steps it actually holds.

    Dividing by the nominal window (ACCUMULATION_STEPS * LOG_REAL_STEPS) is right
    only for a window that starts on a boundary. A resume does not: its first
    window holds fewer micro-steps, so every metric came out scaled by the fraction
    held — CE 1.87 beside a real 3.2, depth_avg 2.76 on a uniform 1–8 draw (#194) —
    and the bogus CE also entered the plateau detector's running minimum. A fresh
    run's first window is one micro-step short too, since steps count from 1.
    """

    FIELDS = ("loss", "token_loss", "grad_norm", "depth")

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = dict.fromkeys(self.FIELDS, 0.0)
        self.count = 0

    def add(self, **values):
        for name in self.FIELDS:
            self.sums[name] += values[name]
        self.count += 1

    def means(self):
        return tuple(self.sums[name] / self.count for name in self.FIELDS)


def init_model_and_optimizer():
    if MODEL_ARCH == "plain":
        print(f"🚀 Initializing PlainTransformer (Dim={LATENT_DIM}, layers={PLAIN_LAYERS})...")
    elif MODEL_ARCH == "refiner":
        print(f"🚀 Initializing Plan A CausalRefiner "
              f"(Dim={LATENT_DIM}, encoder_layers={REFINER_ENCODER_LAYERS}, max_depth={MAX_STEPS_LIMIT})...")
    else:
        print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = build_model(MODEL_ARCH, LATENT_DIM, nnx.Rngs(MODEL_SEED))

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
    # Every schedule with its horizon and what it scales with (#362).
    print("🗓️ Schedules: " + " | ".join(
        f"{name} {steps:,} opt steps ({kind})" for name, (kind, steps) in SCHEDULE_HORIZONS.items()))
    # The whole optimizer at launch, every knob named (#358).
    muon = (f"muon on the matrices (LR x{MUON_LR_MULT:g}, beta {MUON_BETA:g}, "
            f"{MUON_NS_STEPS} Newton-Schulz steps), adamw on the rest"
            if TRM_OPTIMIZER == "muon" else "adamw")
    print(f"🎛️ Optimizer: {muon} | adam b1 {ADAM_B1:g} b2 {ADAM_B2:g} eps {ADAM_EPS:g} | "
          f"weight decay {WEIGHT_DECAY:g} | clip {CLIP_NORM:g}")
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    return model, optimizer

def setup_data_pipeline(start_step, samples_seen=None):
    # Warned here, where data is first needed, not at import: every importer of this
    # module (instruments that never load data included) used to print it.
    if not DATA_ROOT:
        print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")
    print("🚀 Initializing Dynamic Data Phases...")
    pretrain_sources = [TextDataGenerator(f"{DATA_ROOT}/{path}") for path in PRETRAIN_SOURCES]
    pretrain_mixer = DataMixer(pretrain_sources, CURRICULUM_START_WEIGHTS)

    if start_step > 1:
        start_opt_step = start_step // ACCUMULATION_STEPS
        # Prefer the recorded sample count over re-deriving it from micro-steps
        # (#24): only the recorded figure survives a change in BATCH_SIZE between
        # the run that wrote the checkpoint and the one resuming it.
        avg_weights = get_average_curriculum_weights(start_opt_step)
        skips = (split_samples(samples_seen, avg_weights) if samples_seen is not None
                 else samples_from_micro_steps(start_step - 1, avg_weights))
        for gen, skip in zip(pretrain_sources, skips):
            gen.skip_count = skip

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        loader_step = start_step
        while True:
            loader_opt_step = loader_step // ACCUMULATION_STEPS
            pretrain_mixer.set_weights(get_curriculum_weights(loader_opt_step))
            res = pretrain_mixer.get_batch(BATCH_SIZE)

            if res[0] is None:
                data_queue.put((None, None))
                break

            data_queue.put(res)
            loader_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, run_tracker):
    history_file = os.path.join(run_tracker.run_dir, "metrics.csv")
    # On resume, trim CSV rows the restored checkpoint will replay; a fresh run
    # (start_step == 1) appends to any existing CSV untouched.
    start_opt_step = start_step // ACCUMULATION_STEPS if start_step > 1 else None
    logger = MetricsLogger(history_file, start_opt_step=start_opt_step)
    val_probe = ValidationProbe(DATA_ROOT)
    step = start_step

    # f16 gradients underflow to exactly zero without this, which is what destroyed
    # the model at opt step 11,140 on 2026-08-18 (#199). Not restored from the
    # checkpoint: S re-finds its ceiling within a few hundred micro-steps, and a
    # stale value would be a worse starting guess than the standard one.
    loss_scaler = DynamicLossScale()
    print(f"🔍 [LossScale] dynamic f16 loss scaling active, starting at "
          f"{loss_scaler.value:g} (#199)")
    # The clip in the optimizer chain only ever sees the mean of ACCUMULATION_STEPS
    # micro-steps; this one sees each micro-step (#201). Also not checkpointed — the
    # EMA re-converges inside its warmup, and a stale estimate would be a worse
    # ceiling than a freshly measured one.
    grad_guard = GradientNormGuard()
    # How often the optimizer-level clip bit, over the logged opt steps (#358): the
    # micro-step guard above and this clip are different gates, reported apart.
    clip_logged, clip_bit = 0, 0
    print(f"🔍 [GradGuard] per-micro-step clipping at {grad_guard.multiplier:g}x the "
          f"running typical norm, after {grad_guard.warmup} warmup micro-steps (#201)")
    window = LogWindow()
    milestone_mngr = make_milestone_manager(mngr.directory)
    t_compute = 0.0
    nonfinite_streak = 0
    # Latest held-out CE from the validation probe, carried so the (less frequent)
    # logging block can record it. None until the first probe fires.
    latest_val_ce = None
    # Opt step of the last plateau notice, so a persistent plateau reports
    # periodically instead of on every step. Negative so the first one always prints.
    last_plateau_notice = -PLATEAU_NOTICE_EVERY

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
            # The logging micro-step: its gradient stats are sampled before the
            # update below and logged after it, so both blocks read this one flag.
            is_log_step = (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0

            # The ceiling is built from the micro-steps already seen, so it is
            # known before this one runs — an outlier cannot widen the gate it is
            # about to be measured against.
            ceiling = grad_guard.threshold
            loss, out, grads, grad_norm = compute_grad_step(
                model, batch, jnp.array(step), depth, doc_boundary=doc_boundary,
                loss_scale=jnp.float32(loss_scaler.value),
                clip_norm=jnp.float32(jnp.inf if ceiling is None else ceiling),
            )

            current_loss = float(loss)
            current_grad_norm = float(grad_norm)
            grad_guard.observe(current_grad_norm)

            if not (math.isfinite(current_loss) and math.isfinite(current_grad_norm)):
                # Divergence must be loud and must not poison the optimizer state
                # or any state the model carries (which the grad step already
                # overwrote).
                nonfinite_streak += 1
                # With loss scaling on, the overwhelmingly likely cause is that S
                # climbed past what the f16 backward can hold (#199), so back it off
                # and let the next micro-step try again. A genuine divergence keeps
                # producing non-finite steps as S falls, and the streak abort below
                # still ends the run.
                scale_now = loss_scaler.record_overflow()
                print(
                    f"⚠️ Non-finite loss/grad at micro-step {step} "
                    f"(loss={current_loss}, grad_norm={current_grad_norm}, streak={nonfinite_streak}) — "
                    f"skipping update, loss scale → {scale_now:g}."
                )
                model.reset_state()
                if nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {MAX_NONFINITE_STREAK} consecutive non-finite micro-steps "
                        f"(last at step {step})."
                    )
                step += 1
                continue
            nonfinite_streak = 0
            if loss_scaler.record_good_step():
                print(f"🔍 [LossScale] raised to {loss_scaler.value:g} after "
                      f"{loss_scaler.growth_interval} clean micro-steps (#199)")

            # Underflow instrument (#82), sampled BEFORE the update: apply_grads
            # donates the grad buffers and the accumulator (#128). This logging
            # micro-step is always an apply boundary, so two readings exist and are
            # kept apart (#191): the gradient that actually updates the weights (the
            # window's mean), and this one micro-step's, which carries per-draw
            # artifacts that never reach the weights.
            if is_log_step:
                applied_fracs, applied_norm = applied_gradient_stats(optimizer, grads)
                zero_fracs = {k: float(v) for k, v in applied_fracs.items()}
                # The norm the clip actually sees (#180). grad_norm_avg is per-micro-step
                # and cannot be read against CLIP_NORM; this can: above it, the clip, not
                # the LR schedule, is setting the step size.
                applied_grad_norm = float(applied_norm)
                clip_active = int(applied_grad_norm > CLIP_NORM)
                clip_logged, clip_bit = clip_logged + 1, clip_bit + clip_active
                zero_frac_dense = dense_zero_frac_max(zero_fracs)
                zero_frac_dense_microstep = float(dense_zero_frac_max(grad_zero_fractions(grads)))

            apply_grads(optimizer, grads, model)

            t_compute += (time.time() - t_compute_start)

            current_token_loss = float(out.diag.get('token_loss', loss))

            window.add(loss=current_loss, token_loss=current_token_loss,
                       grad_norm=current_grad_norm, depth=depth)

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
                        # Best checkpoint: selected on held-out CE (#222), in a
                        # sibling dir so best-retention and rolling-latest
                        # retention never evict each other.
                        if monitor.push_val(val_ce, opt_step):
                            save_checkpoint(best_mngr, step, model, optimizer, monitor,
                                            run_tracker.run_id, wait=False)

                # Rolling-latest: persist the true latest state on its own cadence
                # so a resume continues from where training actually left off
                # (ROLLING_KEEP by recency, trm/runtime/layout.py). Kept out of the
                # logging block — the full-state save blocks, so it must stay rare.
                # The best-CE state is saved on the validation probe, above.
                if opt_step % CHECKPOINT_EVERY_OPT_STEPS == 0:
                    save_checkpoint(mngr, step, model, optimizer, monitor,
                                    run_tracker.run_id, wait=False)
                    # Milestones: never evicted by recency (#187), in their own dir.
                    if milestone_due(opt_step, CHECKPOINT_EVERY_OPT_STEPS, TOKENS_PER_OPT_STEP):
                        save_checkpoint(milestone_mngr, step, model, optimizer, monitor,
                                        run_tracker.run_id, wait=False)

            if is_log_step:
                opt_step = (step + 1) // ACCUMULATION_STEPS
                accum_loss, accum_token_loss, accum_grad_norm, accum_depth = window.means()

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
                    zero_frac_dense_max=zero_frac_dense_microstep,
                    applied_zero_frac_dense_max=zero_frac_dense,
                    applied_grad_norm=applied_grad_norm,
                    clip_active=clip_active,
                    mix=mixture_label(PRETRAIN_SOURCES, get_curriculum_weights(opt_step)),
                )
                # Logged once; clear so it isn't re-attributed to later opt-steps.
                latest_val_ce = None

                print(
                    f"🧊 [ZeroGrad] dense max: {zero_frac_dense:.4f} | "
                    + " ".join(f"{k}={v:.3f}" for k, v in zero_fracs.items())
                )
                # The clip rate is the guard's own vital sign (#201): a few percent
                # means it is catching the tail it was built for, ~0% means the
                # ceiling has drifted too high to catch anything, and a large
                # fraction means it is clipping ordinary steps and reshaping training.
                ceiling_now = grad_guard.threshold
                print(
                    f"✂️ [GradGuard] clipped {grad_guard.clip_rate:.2%} of "
                    f"{grad_guard.observed:,} micro-steps | typical norm "
                    f"{grad_guard.ema:.1f} | ceiling "
                    + ("warming up" if ceiling_now is None else f"{ceiling_now:.1f}")
                    + f" | opt-level clip bit on {clip_bit / max(clip_logged, 1):.0%} "
                      f"of {clip_logged:,} logged steps"
                )

                curr_weights = get_curriculum_weights(opt_step)
                print(
                    f"📚 [Curriculum] Opt Step: {opt_step} | Avg Sampled Depth: {accum_depth:.2f} | "
                    f"Weights (Web/Code/Math): {curr_weights[0]:.3f} / {curr_weights[1]:.3f} / {curr_weights[2]:.3f}"
                )

                # Periodically update session duration to capture active timings
                run_tracker.update_session_duration()

                monitor.push(opt_step, float(accum_token_loss), float(accum_loss))
                # Plateau is a property of HELD-OUT CE (#184), advanced by the
                # validation probe above on its own cadence. It is a report, never an
                # action: a flat curve at a high LR usually means "cannot descend
                # further yet", not "converged" — the in-run SFT flip that treated it
                # as the latter killed #157 and was removed (#323).
                if monitor.plateaued and opt_step - last_plateau_notice >= PLATEAU_NOTICE_EVERY:
                    last_plateau_notice = opt_step
                    print(f"📉 [Plateau] held-out CE flat for >{monitor.patience} opt steps "
                          f"(best windowed val CE {monitor.best_avg_ce:.4f}); pretraining continues.")

                window.reset()
                t_compute = 0.0

            step += 1
    finally:
        # An asynchronous checkpoint write may still be landing (#218) — a crash, a
        # budget stop's TERM, or a divergence kill must not cut the last one short.
        wait_for_pending_saves()
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()
