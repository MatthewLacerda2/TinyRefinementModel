"""The training loop: a micro-step, plus what runs every N optimizer steps (#476).

`TrainLoop.run` reads a batch, runs `micro_step`, and when the micro-step closes an
optimizer window, runs each hook of `cadence()` that is due, in the order listed. A
hook is a small method that reads what it needs: validation, the rolling checkpoint,
the milestone checkpoint, the log row.

Every knob the loop reads comes from the Config it is handed (#475)."""

import math
import os
import time
from dataclasses import dataclass

import jax
import numpy as np

from trm.runtime.checkpoints import (make_milestone_manager, milestone_due, milestone_thresholds,
                                     save_checkpoint, save_milestone, wait_for_pending_saves)
from trm.runtime.layout import LOG_REAL_STEPS
from trm.runtime.metrics import MetricsLogger
from trm.train.grad_guard import GradientNormGuard
from trm.train.grad_step import HotPath, applied_gradient_stats, dense_zero_frac_max, grad_zero_fractions
from trm.train.loss_scale import DynamicLossScale
from trm.train.schedules import Schedules, mixture_label
from trm.train.validation import VAL_BY_SOURCE, ValidationProbe

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


@dataclass(frozen=True)
class AppliedStats:
    """The underflow instrument (#82) and the norm the clip sees (#180), read on a
    logging micro-step before the update donates the gradient buffers (#128). Two
    readings are kept apart (#191): the window's mean, which updates the weights, and
    this one micro-step's, which carries per-draw artifacts that never reach them."""
    zero_fracs: dict
    zero_frac_dense: float
    zero_frac_dense_microstep: float
    grad_norm: float
    clip_active: int


@dataclass(frozen=True)
class Boundary:
    """The micro-step that closed an optimizer window, as the hooks see it."""
    step: int
    opt_step: int
    out: object
    applied: AppliedStats | None


class TrainLoop:
    def __init__(self, config, model, optimizer, monitor, run_tracker, mngr, best_mngr, start_step, data_root):
        self.config, self.monitor, self.run_tracker = config, monitor, run_tracker
        self.mngr, self.best_mngr, self.start_step = mngr, best_mngr, start_step
        self.schedules = Schedules.of(config)
        # On resume, trim CSV rows the restored checkpoint will replay; a fresh run
        # (start_step == 0) appends to any existing CSV untouched. A checkpoint at
        # micro-step k holds every opt step up to (k + 1) // ACCUMULATION_STEPS, and its
        # row was logged before the save, so it stays: trimming from
        # start_step // ACCUMULATION_STEPS dropped that row on every resume (found by
        # tests/apparatus/test_trainer_end_to_end.py).
        start_opt_step = start_step // config.ACCUMULATION_STEPS + 1 if start_step > 0 else None
        self.logger = MetricsLogger(os.path.join(run_tracker.run_dir, "metrics.csv"), start_opt_step=start_opt_step)
        self.val_probe = ValidationProbe.of(config, data_root)
        self.source_probes = [ValidationProbe.of(config, data_root, source=s) for s in VAL_BY_SOURCE]

        # Not restored from the checkpoint: S re-finds its ceiling within a few hundred
        # micro-steps, and a stale value would be a worse starting guess (#199).
        self.loss_scaler = DynamicLossScale(growth_interval=config.LOSS_SCALE_GROWTH_INTERVAL)
        print(f"🔍 [LossScale] dynamic f16 loss scaling active, starting at "
              f"{self.loss_scaler.value:g} (#199)")
        # The clip in the optimizer chain only ever sees the mean of ACCUMULATION_STEPS
        # micro-steps; this one sees each micro-step (#201). Also not checkpointed — the
        # EMA re-converges inside its warmup, and a stale estimate would be a worse
        # ceiling than a freshly measured one.
        self.grad_guard = GradientNormGuard()
        print(f"🔍 [GradGuard] per-micro-step clipping at {self.grad_guard.multiplier:g}x the "
              f"running typical norm, after {self.grad_guard.warmup} warmup micro-steps (#201)")
        # How often the optimizer-level clip bit, over the logged opt steps (#358): the
        # micro-step guard above and this clip are different gates, reported apart.
        self.clip_logged, self.clip_bit = 0, 0
        self.window = LogWindow()
        self.source_grads = SourceGrads(self.schedules.sources)
        self.milestone_mngr = make_milestone_manager(mngr.directory)
        self.milestones = milestone_thresholds(config.MILESTONE_FIRST_TOKENS, config.MILESTONE_RATIO,
                                               config.MILESTONE_MAX_COUNT)
        self.t_compute = 0.0
        self.nonfinite_streak = 0
        # The latest held-out readings, carried to the (less frequent) log row and
        # cleared once logged. None until the first probe fires.
        self.latest_val_ce = self.latest_val_step = self.latest_val_by_source = None
        # Negative so the first plateau notice always prints.
        self.last_plateau_notice = -PLATEAU_NOTICE_EVERY
        # The grad step and the optimizer apply with the NNX graph walked once (#474).
        self.hot = HotPath(model, optimizer, z_loss_weight=config.Z_LOSS_WEIGHT)

    def cadence(self):
        """(every N opt steps, hook), run in this order at each window boundary. The log
        row comes last so it carries the validation that fired at the same step."""
        return ((self.config.VAL_EVERY_OPT_STEPS, self.validate),
                (self.config.CHECKPOINT_EVERY_OPT_STEPS, self.save_rolling),
                (1, self.save_milestone_if_due),
                (LOG_REAL_STEPS, self.log_row))

    def run(self, data_queue):
        step = self.start_step
        accumulation_steps = self.config.ACCUMULATION_STEPS
        hooks = self.cadence()
        try:
            while True:
                with span("data_get"):
                    batch, source, data_state = data_queue.get()
                if batch is None:
                    break
                self.monitor.data_state = data_state
                # Count consumed samples as they are consumed (#24). Deriving this at
                # save time as step x BATCH_SIZE would be wrong for exactly the run
                # that needs it most: one resumed at a different batch size than it
                # was trained at, whose history spans both.
                self.monitor.samples_seen += batch.shape[0]

                closes_window = (step + 1) % accumulation_steps == 0
                opt_step = (step + 1) // accumulation_steps
                is_log_step = closes_window and opt_step % LOG_REAL_STEPS == 0
                outcome = self.micro_step(step, batch, source, sample_applied=is_log_step)
                if outcome is None:
                    # `step` does NOT advance (#355): the skipped batch is discarded and
                    # the next one takes its place. The optimizer never counts a skipped
                    # micro-step, so advancing here drifted every boundary keyed on
                    # `step` off the optimizer's real windows, by one micro-step per
                    # skip. Consumed data is counted apart, in monitor.samples_seen.
                    continue
                if closes_window:
                    at = Boundary(step, opt_step, *outcome)
                    for every, hook in hooks:
                        if opt_step % every == 0:
                            hook(at)
                step += 1
        finally:
            # The module objects get the loop's last state, for whoever reads them after
            # this returns (#474: the live state is the hot path's while the loop runs).
            self.hot.sync()
            # An asynchronous checkpoint write may still be landing (#218) — a crash, a
            # budget stop's TERM, or a divergence kill must not cut the last one short.
            wait_for_pending_saves()
            # Guarantee run metadata is finalized on exit
            self.run_tracker.update_session_duration()

    # ── the micro-step ───────────────────────────────────────────────────────

    def micro_step(self, step, batch, source, sample_applied):
        """One batch: the gradient, the guards around it, the update. Returns
        (model output, AppliedStats or None), or None for a skipped non-finite step."""
        t_compute_start = time.time()
        # The ceiling is built from the micro-steps already seen, so it is known before
        # this one runs — an outlier cannot widen the gate it is about to be measured against.
        ceiling = self.grad_guard.threshold
        # numpy scalars (#411): they ride in the jitted call's own transfer instead
        # of each dispatching a device array first.
        with span("step_scalars"):
            loss_scale = np.float32(self.loss_scaler.value)
            clip_norm = np.float32(np.inf if ceiling is None else ceiling)
        with span("grad_step"):
            loss, out, grads, grad_norm = self.hot.grad_step(batch, loss_scale=loss_scale, clip_norm=clip_norm)

        with span("loss_readback"):  # one blocking read for both scalars, not two (#411)
            current_loss, current_grad_norm = map(float, jax.device_get((loss, grad_norm)))
        with span("guard"):
            self.grad_guard.observe(current_grad_norm)
            if math.isfinite(current_grad_norm):
                self.source_grads.add(source, current_grad_norm,
                                      clipped=ceiling is not None and current_grad_norm > ceiling)

        if not (math.isfinite(current_loss) and math.isfinite(current_grad_norm)):
            self.skip_nonfinite(step, current_loss, current_grad_norm)
            return None
        self.nonfinite_streak = 0
        with span("loss_scale"):
            if self.loss_scaler.record_good_step():
                print(f"🔍 [LossScale] raised to {self.loss_scaler.value:g} after "
                      f"{self.loss_scaler.growth_interval} clean micro-steps (#199)")

        # Sampled BEFORE the update: apply donates the grad buffers and the accumulator.
        applied = self.sample_applied(grads) if sample_applied else None
        with span("apply_grads"):
            self.hot.apply(grads)
        if sample_applied:
            self.hot.check_counter()
        self.t_compute += time.time() - t_compute_start

        with span("token_loss_readback"):
            current_token_loss = float(out.diag.get('token_loss', loss))
        self.window.add(loss=current_loss, token_loss=current_token_loss, grad_norm=current_grad_norm)
        return out, applied

    def skip_nonfinite(self, step, loss, grad_norm):
        """Divergence must be loud and must not poison the optimizer state. With loss
        scaling on, the overwhelmingly likely cause is that S climbed past what the f16
        backward can hold (#199), so back it off and let the next micro-step try again.
        A genuine divergence keeps producing non-finite steps as S falls, and the streak
        abort still ends the run."""
        self.nonfinite_streak += 1
        scale_now = self.loss_scaler.record_overflow()
        print(f"⚠️ Non-finite loss/grad at micro-step {step} "
              f"(loss={loss}, grad_norm={grad_norm}, streak={self.nonfinite_streak}) — "
              f"skipping update, loss scale → {scale_now:g}.")
        if self.nonfinite_streak >= MAX_NONFINITE_STREAK:
            raise RuntimeError(f"Training diverged: {MAX_NONFINITE_STREAK} consecutive non-finite "
                               f"micro-steps (last at step {step}).")

    def sample_applied(self, grads):
        applied_fracs, applied_norm = applied_gradient_stats(self.hot.optimizer, grads)
        zero_fracs = {k: float(v) for k, v in applied_fracs.items()}
        # The norm the clip actually sees (#180). grad_norm_avg is per-micro-step and
        # cannot be read against CLIP_NORM; this can: above it, the clip, not the LR
        # schedule, is setting the step size.
        applied_grad_norm = float(applied_norm)
        clip_active = int(applied_grad_norm > self.config.CLIP_NORM)
        self.clip_logged, self.clip_bit = self.clip_logged + 1, self.clip_bit + clip_active
        return AppliedStats(
            zero_fracs=zero_fracs,
            zero_frac_dense=dense_zero_frac_max(zero_fracs),
            zero_frac_dense_microstep=float(dense_zero_frac_max(grad_zero_fractions(grads))),
            grad_norm=applied_grad_norm,
            clip_active=clip_active)

    # ── the hooks ────────────────────────────────────────────────────────────

    def validate(self, at):
        """Held-out CE on its own cadence of optimizer steps, independent of the log
        row: nesting it inside logging multiplied the interval by LOG_REAL_STEPS."""
        val_ce = self.val_probe.run(self.hot.model)
        if val_ce is not None:
            self.latest_val_ce, self.latest_val_step = val_ce, at.opt_step
            print(f"🧪 [Validation] Opt Step {at.opt_step} | held-out CE: {val_ce:.4f}")
            # Best checkpoint: selected on held-out CE (#222), in a sibling dir so
            # best-retention and rolling-latest retention never evict each other.
            if self.monitor.push_val(val_ce, at.opt_step):
                save_checkpoint(self.best_mngr, at.step, self.hot.model, self.hot.optimizer, self.monitor,
                                self.run_tracker.run_id, wait=False)
        # The other corpora (#363), on a rarer cadence of the same steps.
        if at.opt_step % self.config.VAL_BY_SOURCE_EVERY_OPT_STEPS == 0:
            readings = {p.source: p.run(self.hot.model) for p in self.source_probes}
            self.latest_val_by_source = ";".join(
                f"{s}={ce:.4f}" for s, ce in readings.items() if ce is not None) or None
            if self.latest_val_by_source:
                print(f"🧪 [Validation] Opt Step {at.opt_step} | by source: {self.latest_val_by_source}")

    def save_rolling(self, at):
        """Rolling-latest: the true latest state, so a resume continues from where
        training left off (ROLLING_KEEP by recency, trm/runtime/layout.py). The
        full-state save blocks, so it keeps its own, rare cadence."""
        save_checkpoint(self.mngr, at.step, self.hot.model, self.hot.optimizer, self.monitor,
                        self.run_tracker.run_id, wait=False)

    def save_milestone_if_due(self, at):
        """Milestones: never evicted by recency (#187), in their own dir, at doubling
        token counts and weights-only (#394). Checked every optimizer step, because the
        early ones are closer together than the rolling cadence."""
        if milestone_due(at.opt_step, at.opt_step - 1, self.config.TOKENS_PER_OPT_STEP, self.milestones):
            save_milestone(self.milestone_mngr, at.step, self.hot.model, self.monitor,
                           self.run_tracker.run_id, wait=False)

    def log_row(self, at):
        """The metrics row and the console report for one logging window, then a fresh
        window. Grad-zero caveats (structural zeros behind the zero-init down_proj early
        in training) live on grad_zero_fractions itself."""
        accum_loss, accum_token_loss, accum_grad_norm = self.window.means()
        applied = at.applied
        self.logger.log(
            at.opt_step, float(accum_token_loss), float(accum_loss), at.out, self.t_compute,
            grad_norm_avg=float(accum_grad_norm),
            seg1_ce=float(at.out.diag.get('seg1_ce', 0)),
            val_ce=self.latest_val_ce,
            val_step=self.latest_val_step,
            val_by_source=self.latest_val_by_source,
            zero_frac_dense_max=applied.zero_frac_dense_microstep,
            applied_zero_frac_dense_max=applied.zero_frac_dense,
            applied_grad_norm=applied.grad_norm,
            clip_active=applied.clip_active,
            mix=mixture_label(self.schedules.sources, self.schedules.curriculum_weights(at.opt_step)),
            grad_by_source=self.source_grads.label(),
            loss_scale=f"{self.loss_scaler.value:g}",
            skipped_micro_steps=self.loss_scaler.overflows,
        )
        # Logged once; clear so it isn't re-attributed to later opt-steps.
        self.latest_val_ce = self.latest_val_step = self.latest_val_by_source = None
        self.print_report(at.opt_step, applied)

        # Periodically update session duration to capture active timings
        self.run_tracker.update_session_duration()
        self.monitor.push(at.opt_step, float(accum_token_loss), float(accum_loss))
        self.report_plateau(at.opt_step)

        self.window.reset()
        self.source_grads.reset()
        self.t_compute = 0.0

    def print_report(self, opt_step, applied):
        print(f"🧊 [ZeroGrad] dense max: {applied.zero_frac_dense:.4f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in applied.zero_fracs.items()))
        # The clip rate is the guard's own vital sign (#201): a few percent means it is
        # catching the tail it was built for, ~0% means the ceiling has drifted too high
        # to catch anything, and a large fraction means it is clipping ordinary steps
        # and reshaping training.
        guard, ceiling = self.grad_guard, self.grad_guard.threshold
        print(f"✂️ [GradGuard] clipped {guard.clip_rate:.2%} of {guard.observed:,} micro-steps | "
              f"typical norm {guard.ema:.1f} | ceiling "
              + ("warming up" if ceiling is None else f"{ceiling:.1f}")
              + f" | opt-level clip bit on {self.clip_bit / max(self.clip_logged, 1):.0%} "
                f"of {self.clip_logged:,} logged steps")
        print(f"📐 [GradBySource] mean/max/guard-clipped/micro-steps: {self.source_grads.label()}")
        weights = self.schedules.curriculum_weights(opt_step)
        print(f"📚 [Curriculum] Opt Step: {opt_step} | Weights: {mixture_label(self.schedules.sources, weights)}")

    def report_plateau(self, opt_step):
        """Plateau is a property of HELD-OUT CE (#184), advanced by the validation hook.
        It is a report, never an action: a flat curve at a high LR usually means "cannot
        descend further yet", not "converged" — the in-run SFT flip that treated it as
        the latter killed #157 and was removed (#323)."""
        if self.monitor.plateaued and opt_step - self.last_plateau_notice >= PLATEAU_NOTICE_EVERY:
            self.last_plateau_notice = opt_step
            print(f"📉 [Plateau] held-out CE flat for >{self.monitor.patience} opt steps "
                  f"(best windowed val CE {self.monitor.best_avg_ce:.4f}); pretraining continues.")


def train_loop(config, model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, run_tracker, data_root):
    TrainLoop(config, model, optimizer, monitor, run_tracker, mngr, best_mngr, start_step, data_root).run(data_queue)
