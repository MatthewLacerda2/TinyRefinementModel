"""The training loop and its data pipeline. The pieces the loop *uses* live in
their own modules: held-out scoring in validation.py, the optimizer chains in
optimizers.py, schedules and mixture policies in schedules.py.

Every knob the loop reads comes from the Config it is handed (#475):
trm.train.start builds it, with a resumed run's own budget, and passes it down."""

import os
import math
import time
import threading
import queue

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.config import resolve_root
from trm.model import build_model
from trm.settings import DEFAULT_DATA_MIXTURE, load_env, location
from trm.runtime.layout import LOG_REAL_STEPS
from trm.runtime.checkpoints import (make_milestone_manager, milestone_due, milestone_thresholds,
                                     save_checkpoint, save_milestone, wait_for_pending_saves)
from trm.train.grad_step import (HotPath, applied_gradient_stats, grad_zero_fractions,
                                 dense_zero_frac_max)
from trm.train.grad_guard import GradientNormGuard
from trm.train.loss_scale import DynamicLossScale
from trm.train.optimizers import optimizer_chain
from trm.train.schedules import Schedules, parse_mixture
from trm.train.validation import VAL_BY_SOURCE, ValidationProbe
from trm.runtime.metrics import MetricsLogger
from trm.data.loaders import TextDataGenerator, DataMixer

load_env()

PREFETCH_SIZE = 128

# Abort training after this many consecutive non-finite micro-steps.
MAX_NONFINITE_STREAK = 50

# Opt steps between "still plateaued" notices. `monitor.plateaued` stays True on
# every logging step until held-out CE improves, so an unthrottled notice would
# print on every one of them and bury the rest of the log.
PLATEAU_NOTICE_EVERY = 200

# Host spans of the micro-step, named for a profiler trace (instruments/profile_step.py,
# #473). A span records only while a trace runs; otherwise it costs ~0.4 µs.
SPAN_PREFIX = "trm/"


def span(name):
    return jax.profiler.TraceAnnotation(SPAN_PREFIX + name)

DATA_ROOT = location("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)




def mixture_label(sources, weights):
    """`fineweb-edu=0.600 codeparrot=0.250 finemath=0.150` — the mixture a CE was
    measured on, readable without today's curriculum constants (#186)."""
    assert len(sources) == len(weights), "a mixture must name every source it weights"
    return " ".join(f"{src.rsplit('/', 1)[-1]}={w:.3f}" for src, w in zip(sources, weights))


class SourceGrads:
    """Per-source micro-step gradient norms over one logging window (#364).

    The micro-step norms are heavy-tailed (p50 11, p99 603, max 75,331; #201), and
    nothing said which data produces the tail. Each micro-step is one source's
    chunk at batch 1, so the norm can be filed under the source it came from. A
    batch that mixed sources (batch > 1) is filed as `mixed`."""

    def __init__(self, sources):
        self.names = [src.rsplit("/", 1)[-1] for src in sources]
        self.reset()

    def reset(self):
        self.seen = {}

    def add(self, source, norm, clipped):
        name = "mixed" if source is None else self.names[source]
        n, total, peak, clips = self.seen.get(name, (0, 0.0, 0.0, 0))
        self.seen[name] = (n + 1, total + norm, max(peak, norm), clips + int(clipped))

    def label(self):
        """`fineweb-edu=11.2/603.5/0/534 ...`: mean / max / guard-clipped / micro-steps."""
        return " ".join(f"{name}={total / n:.1f}/{peak:.1f}/{clips}/{n}"
                        for name, (n, total, peak, clips) in self.seen.items())


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


def samples_from_micro_steps(micro_steps, weights, batch_size):
    """split_samples for the case with no recorded sample count — a pre-#24
    checkpoint, or a fresh run. Converts micro-steps at the given batch size."""
    return split_samples(micro_steps * batch_size, weights)


class LogWindow:
    """Means over one logging window, divided by the micro-steps it actually holds.

    Dividing by the nominal window (ACCUMULATION_STEPS * LOG_REAL_STEPS) is right
    only for a window that starts on a boundary. A resume does not: its first
    window holds fewer micro-steps, so every metric came out scaled by the fraction
    held — CE 1.87 beside a real 3.2 (#194) —
    and the bogus CE also entered the plateau detector's running minimum. A fresh
    run's first window is one micro-step short too, since steps count from 1.
    """

    FIELDS = ("loss", "token_loss", "grad_norm")

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


def init_model_and_optimizer(config):
    print(f"🚀 Initializing PlainTransformer (Dim={config.LATENT_DIM}, layers={config.PLAIN_LAYERS})...")
    model = build_model(config, nnx.Rngs(config.MODEL_SEED))

    print(f"📐 {_param_count(model) / 1e6:.1f}M parameters "
          f"(MODEL_SEED={config.MODEL_SEED}, DATA_SEED={config.DATA_SEED})")
    schedules = Schedules.of(config)
    # The resolved LR horizon must be visible at launch (#83): an anneal that
    # bottoms out before the budget ends is undertraining masquerading as an
    # architecture problem.
    budget = config.TRAIN_TOKEN_BUDGET
    budget_note = (f"TRAIN_TOKEN_BUDGET={budget:,}" if budget is not None
                   else "TRAIN_TOKEN_BUDGET unset — historical default")
    print(f"🗓️ LR horizon: DECAY_STEPS={schedules.decay_steps:,} opt steps "
          f"(warmup {config.WARMUP_STEPS:,}) ≈ "
          f"{schedules.decay_steps * config.TOKENS_PER_OPT_STEP / 1e9:.2f}B "
          f"target tokens ({budget_note})")
    # Every schedule with its horizon and what it scales with (#362).
    print("🗓️ Schedules: " + " | ".join(
        f"{name} {steps:,} opt steps ({kind})" for name, (kind, steps) in schedules.horizons.items()))
    # The mixture at both ends of its ramp, by bucket (#439): what this run reads.
    print(f"🥣 Mixture: {mixture_label(schedules.sources, schedules.start_weights)} -> "
          f"{mixture_label(schedules.sources, schedules.end_weights)}")
    # The whole optimizer at launch, every knob named (#358).
    muon = (f"muon on the matrices (LR x{config.MUON_LR_MULT:g}, beta {config.MUON_BETA:g}, "
            f"{config.MUON_NS_STEPS} Newton-Schulz steps, {config.MUON_NS_COEFFS}), adamw on the rest"
            if config.TRM_OPTIMIZER == "muon" else "adamw")
    print(f"🎛️ Optimizer: {muon} | adam b1 {config.ADAM_B1:g} b2 {config.ADAM_B2:g} "
          f"eps {config.ADAM_EPS:g} | weight decay {config.WEIGHT_DECAY:g} x LR"
          + (f", embedding {config.EMBED_WEIGHT_DECAY:g}/step at peak" if config.TRM_OPTIMIZER == "muon" else "")
          + f" | clip {config.CLIP_NORM:g}")
    # What the loss and the stream are, beside the optimizer (#357, #369).
    print(f"🧮 Loss: z-loss {config.Z_LOSS_WEIGHT:g} x log^2 Z | residual stream "
          f"{config.RESIDUAL_DTYPE}")
    optimizer = nnx.Optimizer(model, optimizer_chain(config, schedules.learning_rate), wrt=nnx.Param)

    return model, optimizer

def setup_data_pipeline(config, start_step, samples_seen=None, data_state=None):
    # Warned here, where data is first needed, not at import: every importer of this
    # module (instruments that never load data included) used to print it.
    if not DATA_ROOT:
        print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")
    print("🚀 Initializing Dynamic Data Phases...")
    schedules = Schedules.of(config)
    # Every reader and the mixer seeded from DATA_SEED, each its own stream.
    pretrain_sources = [TextDataGenerator(f"{DATA_ROOT}/{path}", max_seq_len=config.MAX_SEQ_LEN,
                                          rng=np.random.default_rng(config.DATA_SEED))
                        for path in schedules.sources]
    pretrain_mixer = DataMixer(pretrain_sources, schedules.start_weights,
                               rng=np.random.default_rng(config.DATA_SEED), names=schedules.sources)

    if config.DATA_BRANCH and start_step == 0:
        # A branch that found no checkpoint would train a fresh model on the arm's
        # mixture and report it as the branch.
        raise SystemExit("DATA_BRANCH=1 but no checkpoint was restored: a branch starts from one (#489)")
    if start_step > 0 and config.DATA_BRANCH:
        # A branch onto this run's mixture (#489): the weights continue, the stream
        # is rebuilt, and each seed reads its own rows.
        if data_state is None:
            raise SystemExit("DATA_BRANCH=1 needs a checkpoint that saved its data state (#424)")
        pretrain_mixer.branch_state(data_state, parse_mixture(DEFAULT_DATA_MIXTURE)[0],
                                    skip=config.DATA_SEED * config.DATA_BRANCH_SEED_STRIDE)
        print(f"🌿 Data stream branched onto {', '.join(schedules.sources)}: seed {config.DATA_SEED} "
              f"skips {config.DATA_SEED * config.DATA_BRANCH_SEED_STRIDE:,} rows per bucket (#489)")
    elif start_step > 0 and data_state is not None:
        # Exact (#424): the reader and mixer state saved with the last batch the
        # checkpointed run consumed, so the next row is the one it would have read.
        pretrain_mixer.load_state(data_state)
        print("📍 Data stream restored exactly from the checkpoint (#424)")
    elif start_step > 0:
        # A checkpoint from before #424: the position is estimated, and the stream
        # after it is not the one the run would have read.
        print("⚠️ Data position estimated from the sample count: this checkpoint predates "
              "the saved data state (#424)")
        start_opt_step = start_step // config.ACCUMULATION_STEPS
        # Prefer the recorded sample count over re-deriving it from micro-steps
        # (#24): only the recorded figure survives a change in BATCH_SIZE between
        # the run that wrote the checkpoint and the one resuming it.
        avg_weights = schedules.average_curriculum_weights(start_opt_step)
        skips = (split_samples(samples_seen, avg_weights) if samples_seen is not None
                 # Pre-#24 checkpoints come from runs that counted from 1 (#355).
                 else samples_from_micro_steps(start_step - 1, avg_weights, config.BATCH_SIZE))
        for gen, skip in zip(pretrain_sources, skips):
            gen.skip_count = skip

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        loader_step = start_step
        while True:
            loader_opt_step = loader_step // config.ACCUMULATION_STEPS
            pretrain_mixer.set_weights(schedules.curriculum_weights(loader_opt_step))
            batch = pretrain_mixer.get_batch(config.BATCH_SIZE)

            if batch is None:
                data_queue.put((None, None, None))
                break

            # The state AFTER this batch travels with it, so the trainer can save
            # the one for the last batch it actually consumed, not the prefetched ones.
            data_queue.put((batch, pretrain_mixer.last_source, pretrain_mixer.state()))
            loader_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(config, model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, run_tracker):
    history_file = os.path.join(run_tracker.run_dir, "metrics.csv")
    # On resume, trim CSV rows the restored checkpoint will replay; a fresh run
    # (start_step == 0) appends to any existing CSV untouched.
    # The first opt step this run will log. A checkpoint at micro-step k holds every
    # opt step up to (k + 1) // ACCUMULATION_STEPS, and its row was logged before
    # the save, so it stays: trimming from start_step // ACCUMULATION_STEPS dropped
    # that row on every resume (found by tests/apparatus/test_trainer_end_to_end.py).
    accumulation_steps = config.ACCUMULATION_STEPS
    schedules = Schedules.of(config)
    start_opt_step = start_step // accumulation_steps + 1 if start_step > 0 else None
    logger = MetricsLogger(history_file, start_opt_step=start_opt_step)
    val_probe = ValidationProbe.of(config, DATA_ROOT)
    source_probes = [ValidationProbe.of(config, DATA_ROOT, source=s) for s in VAL_BY_SOURCE]
    step = start_step

    # f16 gradients underflow to exactly zero without this, which is what destroyed
    # the model at opt step 11,140 on 2026-08-18 (#199). Not restored from the
    # checkpoint: S re-finds its ceiling within a few hundred micro-steps, and a
    # stale value would be a worse starting guess than the standard one.
    loss_scaler = DynamicLossScale(growth_interval=config.LOSS_SCALE_GROWTH_INTERVAL)
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
    source_grads = SourceGrads(schedules.sources)
    milestone_mngr = make_milestone_manager(mngr.directory)
    milestones = milestone_thresholds(config.MILESTONE_FIRST_TOKENS, config.MILESTONE_RATIO,
                                      config.MILESTONE_MAX_COUNT)
    t_compute = 0.0
    nonfinite_streak = 0
    # Latest held-out CE from the validation probe, carried so the (less frequent)
    # logging block can record it. None until the first probe fires.
    latest_val_ce = None
    latest_val_step = None
    latest_val_by_source = None
    # Opt step of the last plateau notice, so a persistent plateau reports
    # periodically instead of on every step. Negative so the first one always prints.
    last_plateau_notice = -PLATEAU_NOTICE_EVERY
    # The grad step and the optimizer apply with the NNX graph walked once (#474).
    hot = HotPath(model, optimizer, z_loss_weight=config.Z_LOSS_WEIGHT)

    try:
        while True:
            with span("data_get"):
                batch, source, data_state = data_queue.get()
            if batch is None:
                break
            monitor.data_state = data_state

            # Count consumed samples as they are consumed (#24). Deriving this at
            # save time as step x BATCH_SIZE would be wrong for exactly the run
            # that needs it most: one resumed at a different batch size than it
            # was trained at, whose history spans both.
            monitor.samples_seen += batch.shape[0]

            t_compute_start = time.time()

            # The logging micro-step: its gradient stats are sampled before the
            # update below and logged after it, so both blocks read this one flag.
            is_log_step = (step + 1) % (accumulation_steps * LOG_REAL_STEPS) == 0

            # The ceiling is built from the micro-steps already seen, so it is
            # known before this one runs — an outlier cannot widen the gate it is
            # about to be measured against.
            ceiling = grad_guard.threshold
            with span("step_scalars"):
                loss_scale = jnp.float32(loss_scaler.value)
                clip_norm = jnp.float32(jnp.inf if ceiling is None else ceiling)
            with span("grad_step"):
                loss, out, grads, grad_norm = hot.grad_step(
                    batch, loss_scale=loss_scale, clip_norm=clip_norm)

            with span("loss_readback"):
                current_loss = float(loss)
                current_grad_norm = float(grad_norm)
            with span("guard"):
                grad_guard.observe(current_grad_norm)
                if math.isfinite(current_grad_norm):
                    source_grads.add(source, current_grad_norm,
                                     clipped=ceiling is not None and current_grad_norm > ceiling)

            if not (math.isfinite(current_loss) and math.isfinite(current_grad_norm)):
                # Divergence must be loud and must not poison the optimizer state.
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
                if nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {MAX_NONFINITE_STREAK} consecutive non-finite micro-steps "
                        f"(last at step {step})."
                    )
                # `step` does NOT advance (#355): the skipped batch is discarded and the
                # next one takes its place. The optimizer never counts a skipped
                # micro-step, so advancing here drifted every boundary the trainer keys
                # on `step` (logging, the probe, checkpoints, the applied-gradient
                # telemetry) off the optimizer's real windows, by one micro-step per
                # skip: ~200 in a 512-step pair. Consumed data is counted apart, in
                # monitor.samples_seen, so a resume still skips the right amount.
                continue
            nonfinite_streak = 0
            with span("loss_scale"):
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
                applied_fracs, applied_norm = applied_gradient_stats(hot.optimizer, grads)
                zero_fracs = {k: float(v) for k, v in applied_fracs.items()}
                # The norm the clip actually sees (#180). grad_norm_avg is per-micro-step
                # and cannot be read against CLIP_NORM; this can: above it, the clip, not
                # the LR schedule, is setting the step size.
                applied_grad_norm = float(applied_norm)
                clip_active = int(applied_grad_norm > config.CLIP_NORM)
                clip_logged, clip_bit = clip_logged + 1, clip_bit + clip_active
                zero_frac_dense = dense_zero_frac_max(zero_fracs)
                zero_frac_dense_microstep = float(dense_zero_frac_max(grad_zero_fractions(grads)))

            with span("apply_grads"):
                hot.apply(grads)
            if is_log_step:
                hot.check_counter()

            t_compute += (time.time() - t_compute_start)

            with span("token_loss_readback"):
                current_token_loss = float(out.diag.get('token_loss', loss))

            window.add(loss=current_loss, token_loss=current_token_loss,
                       grad_norm=current_grad_norm)

            # Validation probe fires on its own cadence at the optimizer-step
            # boundary (every ACCUMULATION_STEPS micro-steps), independent of the
            # logging block — nesting it inside logging multiplied the effective
            # interval by LOG_REAL_STEPS.
            if (step + 1) % accumulation_steps == 0:
                opt_step = (step + 1) // accumulation_steps
                if opt_step % config.VAL_EVERY_OPT_STEPS == 0:
                    val_ce = val_probe.run(hot.model)
                    if val_ce is not None:
                        latest_val_ce, latest_val_step = val_ce, opt_step
                        print(f"🧪 [Validation] Opt Step {opt_step} | held-out CE: {val_ce:.4f}")
                        # Best checkpoint: selected on held-out CE (#222), in a
                        # sibling dir so best-retention and rolling-latest
                        # retention never evict each other.
                        if monitor.push_val(val_ce, opt_step):
                            save_checkpoint(best_mngr, step, hot.model, hot.optimizer, monitor,
                                            run_tracker.run_id, wait=False)
                    # The other corpora (#363), on a rarer cadence of the same steps.
                    if opt_step % config.VAL_BY_SOURCE_EVERY_OPT_STEPS == 0:
                        readings = {p.source: p.run(hot.model) for p in source_probes}
                        latest_val_by_source = ";".join(
                            f"{s}={ce:.4f}" for s, ce in readings.items() if ce is not None) or None
                        if latest_val_by_source:
                            print(f"🧪 [Validation] Opt Step {opt_step} | by source: {latest_val_by_source}")

                # Rolling-latest: persist the true latest state on its own cadence
                # so a resume continues from where training actually left off
                # (ROLLING_KEEP by recency, trm/runtime/layout.py). Kept out of the
                # logging block — the full-state save blocks, so it must stay rare.
                # The best-CE state is saved on the validation probe, above.
                if opt_step % config.CHECKPOINT_EVERY_OPT_STEPS == 0:
                    save_checkpoint(mngr, step, hot.model, hot.optimizer, monitor,
                                    run_tracker.run_id, wait=False)

                # Milestones: never evicted by recency (#187), in their own dir, at
                # doubling token counts and weights-only (#394). Checked every
                # optimizer step rather than at the rolling boundary, because the
                # early ones are closer together than that boundary.
                if milestone_due(opt_step, opt_step - 1, config.TOKENS_PER_OPT_STEP, milestones):
                    save_milestone(milestone_mngr, step, hot.model, monitor,
                                   run_tracker.run_id, wait=False)

            if is_log_step:
                opt_step = (step + 1) // accumulation_steps
                accum_loss, accum_token_loss, accum_grad_norm = window.means()

                # Underflow instrument (#82): zero_fracs / zero_frac_dense were
                # sampled just before apply_grads above (donation makes the raw
                # grads unreadable here). Interpretation caveats (structural zeros
                # behind the zero-init down_proj early in training) live on
                # grad_zero_fractions itself.
                logger.log(
                    opt_step,
                    float(accum_token_loss),
                    float(accum_loss),
                    out,
                    t_compute,
                    grad_norm_avg=float(accum_grad_norm),
                    seg1_ce=float(out.diag.get('seg1_ce', 0)),
                    val_ce=latest_val_ce,
                    val_step=latest_val_step,
                    val_by_source=latest_val_by_source,
                    zero_frac_dense_max=zero_frac_dense_microstep,
                    applied_zero_frac_dense_max=zero_frac_dense,
                    applied_grad_norm=applied_grad_norm,
                    clip_active=clip_active,
                    mix=mixture_label(schedules.sources, schedules.curriculum_weights(opt_step)),
                    grad_by_source=source_grads.label(),
                    loss_scale=f"{loss_scaler.value:g}",
                    skipped_micro_steps=loss_scaler.overflows,
                )
                # Logged once; clear so it isn't re-attributed to later opt-steps.
                latest_val_ce = latest_val_step = latest_val_by_source = None

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
                print(f"📐 [GradBySource] mean/max/guard-clipped/micro-steps: {source_grads.label()}")

                curr_weights = schedules.curriculum_weights(opt_step)
                print(
                    f"📚 [Curriculum] Opt Step: {opt_step} | "
                    f"Weights: {mixture_label(schedules.sources, curr_weights)}"
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
                source_grads.reset()
                t_compute = 0.0

            step += 1
    finally:
        # The module objects get the loop's last state, for whoever reads them after
        # this returns (#474: the live state is the hot path's while the loop runs).
        hot.model
        # An asynchronous checkpoint write may still be landing (#218) — a crash, a
        # budget stop's TERM, or a divergence kill must not cut the last one short.
        wait_for_pending_saves()
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()
