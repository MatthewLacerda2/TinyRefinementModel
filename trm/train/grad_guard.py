"""Clip the rare micro-step that would otherwise steer the whole optimizer step (#201).

`optax.MultiSteps` wraps the optimizer chain, so `clip_by_global_norm` runs once per
*optimizer* step — on the mean of ACCUMULATION_STEPS micro-steps, never on a
micro-step. That is fine when the per-micro-step gradient norms are tightly
distributed. They are not. Measured over 9,000 micro-steps of the #157 run:

    p50 11.4 · p90 24.7 · p99 603.5 · p99.9 7,617 · max 75,331

The top 1% of micro-steps carry 64.5% of the total gradient mass. One micro-step at
75,331 among 127 at ~11 produces an accumulated mean pointing ~99% in the outlier's
direction; the clip then rescales that to norm 1.0 and hands the optimizer a
correctly-sized step aimed almost entirely where one pathological batch wanted to
go. Every reading looks healthy. The model dies anyway, as it did at opt step
11,650.

The threshold is deliberately NOT a constant. Gradient magnitudes shrink as a run
converges, so a number calibrated against today's p50 of 11.4 is miscalibrated by
step 25,000 — and a stale threshold either stops clipping or starts clipping
everything, both silently. Instead we track what a typical micro-step looks like and
clip at a multiple of that, so the guard follows the run instead of a memory of it.

Two details that matter more than they look:

  - the value folded into the running scale is capped at EMA_UPDATE_CAP times the
    scale itself. Feeding the raw norm lets a single 75,331 drag the ceiling up for
    hundreds of steps afterwards, precisely when the guard is most needed. Feeding
    the *ceiling* is worse and was the first thing tried: the ceiling is already a
    multiple of the scale, so every clipped step raises the scale by 7% and 200 of
    them run it up 800,000-fold — positive feedback that disables the guard exactly
    when it is firing. Capping the update keeps rare outliers nearly powerless while
    still letting a genuine, sustained shift upward carry the ceiling with it.
  - nothing is clipped during warmup. An EMA seeded from one micro-step is not an
    estimate of anything, and clipping against it would corrupt the very early steps
    of a fresh run.
"""

import math

# ~100 micro-steps of memory: long enough to be a stable estimate of "typical",
# short enough to track a converging model rather than its distant past.
NORM_EMA_DECAY = 0.99
# Clip at this multiple of the typical norm. At the measured p50 of 11.4 this lands
# near 91, between p95 (62.8) and p99 (603.5): it touches ~3.6% of micro-steps and
# leaves the other 96.4% bit-identical, which is what makes it safe to switch on
# partway through a run.
OUTLIER_MULTIPLIER = 8.0
# Micro-steps observed before the guard is allowed to clip anything.
WARMUP_MICRO_STEPS = 256
# Ceiling on how much a single micro-step may move the running scale, as a multiple
# of the scale. At the ~3.6% clip rate this is built for, outliers move it by well
# under 1% per occurrence; if instead EVERY micro-step were large — a real regime
# change rather than a tail — the scale still doubles in ~70 micro-steps and the
# guard follows the run rather than strangling it.
EMA_UPDATE_CAP = 2.0


class GradientNormGuard:
    """The running estimate of a typical micro-step, and the ceiling it implies.

    Plain Python on purpose: this is a policy, and keeping it out of jax means the
    rule can be tested against a recorded distribution without a GPU or a trace.
    """

    def __init__(self, decay=NORM_EMA_DECAY, multiplier=OUTLIER_MULTIPLIER,
                 warmup=WARMUP_MICRO_STEPS, update_cap=EMA_UPDATE_CAP):
        self.decay = float(decay)
        self.multiplier = float(multiplier)
        self.warmup = int(warmup)
        self.update_cap = float(update_cap)
        self.ema = None
        self.observed = 0
        self.clipped = 0

    @property
    def threshold(self):
        """The norm above which a micro-step is an outlier, or None while warming up.

        None means "do not clip", and the caller must pass it through as an infinite
        ceiling rather than as zero — a guard that clips everything is worse than no
        guard at all.
        """
        if self.ema is None or self.observed < self.warmup:
            return None
        return self.ema * self.multiplier

    def observe(self, norm):
        """Fold one micro-step's raw gradient norm in; returns the ceiling for it.

        Called BEFORE the gradients are used, so the returned ceiling applies to the
        micro-step just measured. Non-finite norms are ignored entirely: they are
        the loss scaler's business (#199), and letting inf touch the EMA would
        destroy it permanently.
        """
        if not math.isfinite(norm):
            return self.threshold

        ceiling = self.threshold
        self.observed += 1
        if ceiling is not None and norm > ceiling:
            self.clipped += 1
        if self.ema is None:
            self.ema = norm
        else:
            # Capped so no single micro-step — least of all one being clipped — can
            # walk the scale up toward its own magnitude.
            self.ema = (self.decay * self.ema
                        + (1.0 - self.decay) * min(norm, self.update_cap * self.ema))
        return ceiling

    @property
    def clip_rate(self):
        return self.clipped / self.observed if self.observed else 0.0

    def __repr__(self):
        ema = "unset" if self.ema is None else f"{self.ema:.2f}"
        return (f"GradientNormGuard(ema={ema}, threshold={self.threshold}, "
                f"clipped={self.clipped}/{self.observed})")
