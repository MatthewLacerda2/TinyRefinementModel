"""Every schedule a run follows — the learning rate, the data mixture's ramp — resolved
from that run's Config (#475).

`Schedules.of(config)` is the one place a horizon is computed. Nothing here is fixed at
import: the horizon used to be a module constant, resolved from whatever
TRAIN_TOKEN_BUDGET the importing shell held, which is why a resume had to write its
own budget into the environment before anything imported this module (#197).
"""

from dataclasses import dataclass

import optax

# The LR anneal's horizon must match the run length (#83): it derives from the
# planned token budget (TRAIN_TOKEN_BUDGET). With no budget set it stays at the
# historical 15000 opt steps, so existing configs and the golden run resolve unchanged.
DEFAULT_DECAY_STEPS = 15000


def mixture_label(sources, weights):
    """`fineweb-edu=0.600 codeparrot=0.250 finemath=0.150` — the mixture a CE was
    measured on, readable without today's curriculum constants (#186)."""
    assert len(sources) == len(weights), "a mixture must name every source it weights"
    return " ".join(f"{src.rsplit('/', 1)[-1]}={w:.3f}" for src, w in zip(sources, weights))


def resolve_decay_steps(token_budget, tokens_per_opt_step, warmup_steps):
    """Opt-step horizon for the run: token budget / tokens per opt step
    (None → the historical default). A budget that doesn't clear warmup is a
    config error, not a run worth starting — fail loud."""
    if token_budget is None:
        return DEFAULT_DECAY_STEPS
    steps = round(token_budget / tokens_per_opt_step)
    if steps <= warmup_steps:
        raise ValueError(
            f"TRAIN_TOKEN_BUDGET={token_budget} resolves to {steps} opt steps, "
            f"inside the {warmup_steps}-step warmup — the cosine would never decay."
        )
    return steps


def build_learning_schedule(decay_steps, *, warmup_steps, peak_lr):
    """The cosine: warmup from peak/10 to the peak, then anneal to peak/100 by
    `decay_steps`. Every argument is explicit so a caller can rebuild the schedule
    some other run trained under (instruments.report reads them from its metadata)."""
    return optax.warmup_cosine_decay_schedule(
        init_value=peak_lr / 10.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=peak_lr / 100.0,
    )


# ── Warmup-Stable-Decay (#386) ────────────────────────────────────────────────
# The cosine needs the run's length before step 1: it only reaches its low tail (where
# much of the final gain lands) at the budget, so stopping early leaves an un-annealed
# model. WSD holds the peak and anneals only at the end, so a decay can be BRANCHED
# from any checkpoint to read what the model would score if it stopped there. That is
# the owner's stop rule for the base run as a mechanism (val CE < 3.6 or 10 days).
# Adopted as the default for the base run without its matched pair (owner,
# 2026-09-20): WSD matching the cosine at a fixed budget is settled outside (MiniCPM,
# DeepSeek), so the pair would have asked how much it gains HERE, and what the run
# needs is the flexibility, not the number. #386's spec stays on disk, pre-empted.
#   LR_SCHEDULE          wsd (the default since 2026-09-20) | cosine
#   WSD_DECAY_FRACTION   the share of the horizon the final decay takes (0.2: SmolLM2's)
#   WSD_DECAY_START      an explicit opt step to start the decay at, for a branch
#                        resumed from a checkpoint; unset, the decay starts at
#                        (1 - WSD_DECAY_FRACTION) x the horizon.

def wsd_decay_start(decay_steps, decay_fraction, decay_start=None):
    """The opt step WSD's final decay starts at: the explicit one, else the fraction's."""
    return decay_start if decay_start is not None else round((1 - decay_fraction) * decay_steps)


def build_wsd_schedule(decay_steps, *, warmup_steps, peak_lr, decay_fraction, decay_start=None):
    """Warmup from peak/10 to the peak, hold it, then decay linearly to peak/100 by
    `decay_steps`, the same two ends the cosine has, so the pair against it moves
    only the shape between them."""
    start = wsd_decay_start(decay_steps, decay_fraction, decay_start)
    if not warmup_steps <= start < decay_steps:
        raise ValueError(f"WSD decay start {start} must sit between the warmup ({warmup_steps}) "
                         f"and the horizon ({decay_steps})")
    return optax.join_schedules(
        [optax.linear_schedule(peak_lr / 10.0, peak_lr, warmup_steps),
         optax.constant_schedule(peak_lr),
         optax.linear_schedule(peak_lr, peak_lr / 100.0, decay_steps - start)],
        boundaries=[warmup_steps, start])


def build_schedule(decay_steps, kind, *, warmup_steps, peak_lr, decay_fraction=None, decay_start=None):
    """The LR schedule by name, at an explicit horizon. The cosine has no decay start."""
    if kind == "wsd":
        return build_wsd_schedule(decay_steps, warmup_steps=warmup_steps, peak_lr=peak_lr,
                                  decay_fraction=decay_fraction, decay_start=decay_start)
    return build_learning_schedule(decay_steps, warmup_steps=warmup_steps, peak_lr=peak_lr)


# ── Data curriculum ──────────────────────────────────────────────────────────
# Mixture weights ramp linearly from web-heavy toward a code/math-heavy blend
# over the first `curriculum_steps` optimizer steps, then hold steady.
#
# The ramp scales with the run (#362): a short pair inherits the shape of the run it
# informs (CLAUDE.md). It ends a third of the way in, the shape the 4B champion
# trained with (10,000 of 30,518 steps), so a 4B base run resolves to exactly the
# ramp it always had, while a 512-step pair now sees the same web-to-code/math turn
# instead of barely leaving 85% web. Warmup stays absolute: it settles the optimizer
# state, not the recipe. With no budget set, the historical 10,000 steps. The
# fraction itself is MIXTURE_RAMP_FRACTION (trm/settings.py, #439).
DEFAULT_CURRICULUM_STEPS = 10000


def resolve_curriculum_steps(token_budget, decay_steps, ramp_fraction):
    if token_budget is None:
        return DEFAULT_CURRICULUM_STEPS
    return max(1, round(ramp_fraction * decay_steps))


def parse_mixture(text):
    """`DATA_MIXTURE` (trm/settings.py) -> (buckets, start weights, end weights), or a
    refusal that says what is wrong. Refusing before a model is built is the point:
    a typo here would otherwise cost the hours it takes to notice the mix is odd."""
    buckets, starts, ends = [], [], []
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        bucket, sep, weights = entry.partition("=")
        bucket = bucket.strip()
        if not sep or not bucket:
            raise SystemExit(f"DATA_MIXTURE entry {entry!r}: write it as bucket=start:end or bucket=weight")
        start, _, end = weights.partition(":")
        try:
            start = float(start)
            end = float(end) if end.strip() else start
        except ValueError:
            raise SystemExit(f"DATA_MIXTURE entry {entry!r}: {weights!r} is not a weight") from None
        if start < 0 or end < 0:
            raise SystemExit(f"DATA_MIXTURE entry {entry!r}: a weight cannot be negative")
        if bucket in buckets:
            raise SystemExit(f"DATA_MIXTURE names {bucket!r} twice")
        buckets.append(bucket)
        starts.append(start)
        ends.append(end)
    if not buckets:
        raise SystemExit("DATA_MIXTURE names no bucket")
    for column, weights in (("start", starts), ("end", ends)):
        if abs(sum(weights) - 1.0) > 1e-9:
            raise SystemExit(f"DATA_MIXTURE {column} weights sum to {sum(weights):.6g}, not 1: {text!r}")
    return tuple(buckets), starts, ends


@dataclass(frozen=True)
class Schedules:
    """One run's schedules, resolved from its Config.

    `sources` are the mixture's buckets in DataMixer source order. One tuple feeds the
    loaders, the `mix` column of metrics.csv and the per-source gradient telemetry, so
    the recorded mixture cannot name a source other than the one that was served."""

    decay_steps: int                # the LR anneal's horizon, in opt steps (#83)
    curriculum_steps: int           # where the mixture ramp ends (#362)
    learning_rate: object           # the optax schedule, by opt step
    horizons: dict                  # every schedule's horizon and what it scales with
    sources: tuple
    start_weights: list
    end_weights: list

    @classmethod
    def of(cls, config):
        decay_steps = resolve_decay_steps(config.TRAIN_TOKEN_BUDGET, config.TOKENS_PER_OPT_STEP,
                                          config.WARMUP_STEPS)
        curriculum_steps = resolve_curriculum_steps(config.TRAIN_TOKEN_BUDGET, decay_steps,
                                                    config.MIXTURE_RAMP_FRACTION)
        learning_rate = build_schedule(decay_steps, config.LR_SCHEDULE, warmup_steps=config.WARMUP_STEPS,
                                       peak_lr=config.PEAK_LR, decay_fraction=config.WSD_DECAY_FRACTION,
                                       decay_start=config.WSD_DECAY_START)
        # Every schedule's horizon and what it scales with, in one place (#362):
        # "absolute" is a fixed number of opt steps whatever the run's length; "budget"
        # follows the run's token budget. The launch banner prints this; a test holds
        # it complete.
        horizons = {
            "warmup": ("absolute", config.WARMUP_STEPS),
            f"lr {config.LR_SCHEDULE}": ("budget", decay_steps),
            "mixture ramp": ("budget", curriculum_steps),
        }
        if config.LR_SCHEDULE == "wsd":
            horizons["wsd decay start"] = (
                "absolute" if config.WSD_DECAY_START is not None else "budget",
                wsd_decay_start(decay_steps, config.WSD_DECAY_FRACTION, config.WSD_DECAY_START))
        sources, start_weights, end_weights = parse_mixture(config.DATA_MIXTURE)
        return cls(decay_steps, curriculum_steps, learning_rate, horizons,
                   sources, start_weights, end_weights)

    def curriculum_weights(self, loader_step):
        step = float(loader_step)
        if step >= self.curriculum_steps:
            return list(self.end_weights)
        fraction = step / self.curriculum_steps
        return [
            start + (end - start) * fraction
            for start, end in zip(self.start_weights, self.end_weights)
        ]

    def average_curriculum_weights(self, loader_step):
        """Average mixture weights over steps [0, loader_step] — used on resume to
        estimate how many samples each source has already served."""
        step = float(loader_step)
        if step == 0:
            return list(self.start_weights)
        if step >= self.curriculum_steps:
            # During the linear ramp the average weight is the start/end midpoint;
            # blend that with the post-ramp plateau proportionally to time spent in each.
            ramp_fraction = self.curriculum_steps / step
            post_fraction = 1.0 - ramp_fraction
            return [
                (start + end) / 2.0 * ramp_fraction + end * post_fraction
                for start, end in zip(self.start_weights, self.end_weights)
            ]
        curr = self.curriculum_weights(step)
        return [
            (start + current) / 2.0
            for start, current in zip(self.start_weights, curr)
        ]
