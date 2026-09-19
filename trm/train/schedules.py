import os

import numpy as np
import optax

from trm.config import MAX_STEPS_LIMIT, DATA_SEED, TOKENS_PER_OPT_STEP, TRAIN_TOKEN_BUDGET, WEIGHT_DECAY

# Warmup is absolute: it stabilizes the optimizer's first moments, a fixed-cost
# phase that does not grow with the run. Env-overridable for one purpose: a
# short matched pair (#26 stage 2, ~500 opt steps) cannot spend 1000 of them
# warming up. A base run leaves it alone.
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "1000"))

# The peak the schedule warms up to. 6e-4 since 2026-09-19 (#388): 1e-4 was chosen once
# and never compared against anything; #287 measured 3e-4 at 2.25x fewer tokens and
# 6e-4 at 1.24x fewer again (docs/findings/2026-09-18-the-lr-curve-flattens-6e-4-beats-3e-4-by-1-24x.md).
# Env-overridable so a matched pair can sweep it. The golden run has its own optimizer
# and LR, so it does not move with this.
PEAK_LR = float(os.environ.get("PEAK_LR", "6e-4"))

# The LR anneal's horizon must match the run length (#83). DECAY_STEPS derives
# from the planned token budget (config.TRAIN_TOKEN_BUDGET); with no budget set
# it stays at the historical 15000 opt steps, so existing configs and the golden
# run resolve unchanged.
_DEFAULT_DECAY_STEPS = 15000


def resolve_decay_steps(token_budget, tokens_per_opt_step=TOKENS_PER_OPT_STEP,
                        warmup_steps=WARMUP_STEPS):
    """Opt-step horizon for the run: token budget / tokens per opt step
    (None → the historical default). A budget that doesn't clear warmup is a
    config error, not a run worth starting — fail loud."""
    if token_budget is None:
        return _DEFAULT_DECAY_STEPS
    steps = round(token_budget / tokens_per_opt_step)
    if steps <= warmup_steps:
        raise ValueError(
            f"TRAIN_TOKEN_BUDGET={token_budget} resolves to {steps} opt steps, "
            f"inside the {warmup_steps}-step warmup — the cosine would never decay."
        )
    return steps


def build_learning_schedule(decay_steps, warmup_steps=WARMUP_STEPS, peak_lr=PEAK_LR):
    """The run's LR schedule at an explicit horizon; module-level
    learning_schedule is this at the resolved DECAY_STEPS.

    `warmup_steps` and `peak_lr` are explicit so a caller can build the schedule
    some other run trained under: both are env knobs read at import, so the
    module-level defaults describe this process, not that run."""
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
#   LR_SCHEDULE          cosine (the historical default) | wsd
#   WSD_DECAY_FRACTION   the share of the horizon the final decay takes (0.2: SmolLM2's)
#   WSD_DECAY_START      an explicit opt step to start the decay at, for a branch
#                        resumed from a checkpoint; unset, the decay starts at
#                        (1 - WSD_DECAY_FRACTION) x DECAY_STEPS.
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
if LR_SCHEDULE not in ("cosine", "wsd"):
    raise SystemExit(f"LR_SCHEDULE={LR_SCHEDULE!r}: use cosine or wsd (#386)")
WSD_DECAY_FRACTION = float(os.environ.get("WSD_DECAY_FRACTION", "0.2"))
_WSD_DECAY_START_ENV = os.environ.get("WSD_DECAY_START")
WSD_DECAY_START = int(_WSD_DECAY_START_ENV) if _WSD_DECAY_START_ENV else None


def build_wsd_schedule(decay_steps, warmup_steps=WARMUP_STEPS, peak_lr=PEAK_LR,
                       decay_fraction=WSD_DECAY_FRACTION, decay_start=WSD_DECAY_START):
    """Warmup from peak/10 to the peak, hold it, then decay linearly to peak/100 by
    `decay_steps`, the same two ends the cosine has, so the pair against it moves
    only the shape between them."""
    start = decay_start if decay_start is not None else round((1 - decay_fraction) * decay_steps)
    if not warmup_steps <= start < decay_steps:
        raise ValueError(f"WSD decay start {start} must sit between the warmup ({warmup_steps}) "
                         f"and the horizon ({decay_steps})")
    return optax.join_schedules(
        [optax.linear_schedule(peak_lr / 10.0, peak_lr, warmup_steps),
         optax.constant_schedule(peak_lr),
         optax.linear_schedule(peak_lr, peak_lr / 100.0, decay_steps - start)],
        boundaries=[warmup_steps, start])


def build_schedule(decay_steps, kind=LR_SCHEDULE, **overrides):
    """The run's LR schedule by name, at an explicit horizon."""
    if kind == "wsd":
        return build_wsd_schedule(decay_steps, **overrides)
    return build_learning_schedule(decay_steps, **{k: v for k, v in overrides.items()
                                                   if k in ("warmup_steps", "peak_lr")})


DECAY_STEPS = resolve_decay_steps(TRAIN_TOKEN_BUDGET)
learning_schedule = build_schedule(DECAY_STEPS)

weight_decay_schedule = optax.constant_schedule(WEIGHT_DECAY)


# ── Data curriculum ──────────────────────────────────────────────────────────
# Mixture weights ramp linearly from web-heavy toward a code/math-heavy blend
# over the first CURRICULUM_STEPS optimizer steps, then hold steady.
#
# The ramp scales with the run (#362): a short pair inherits the shape of the run it
# informs (CLAUDE.md). It ends a third of the way in, the shape the 4B champion
# trained with (10,000 of 30,518 steps), so a 4B base run resolves to exactly the
# ramp it always had, while a 512-step pair now sees the same web-to-code/math turn
# instead of barely leaving 85% web. Warmup stays absolute: it settles the optimizer
# state, not the recipe. With no budget set, the historical 10,000 steps.
CURRICULUM_RAMP_FRACTION = 10000 / 30518
_DEFAULT_CURRICULUM_STEPS = 10000


def resolve_curriculum_steps(token_budget, decay_steps):
    if token_budget is None:
        return _DEFAULT_CURRICULUM_STEPS
    return max(1, round(CURRICULUM_RAMP_FRACTION * decay_steps))


CURRICULUM_STEPS = resolve_curriculum_steps(TRAIN_TOKEN_BUDGET, DECAY_STEPS)

# Every schedule's horizon and what it scales with, in one place (#362): "absolute"
# is a fixed number of opt steps whatever the run's length; "budget" follows the
# run's token budget. The launch banner prints this; a test holds it complete.
SCHEDULE_HORIZONS = {
    "warmup": ("absolute", WARMUP_STEPS),
    f"lr {LR_SCHEDULE}": ("budget", DECAY_STEPS),
    "mixture ramp": ("budget", CURRICULUM_STEPS),
}
if LR_SCHEDULE == "wsd":
    SCHEDULE_HORIZONS["wsd decay start"] = (
        "absolute" if WSD_DECAY_START is not None else "budget",
        WSD_DECAY_START if WSD_DECAY_START is not None else round((1 - WSD_DECAY_FRACTION) * DECAY_STEPS))
# Endpoints over the (web, code, math) sources, in DataMixer source order.
CURRICULUM_START_WEIGHTS = [0.85, 0.10, 0.05]
CURRICULUM_END_WEIGHTS = [0.35, 0.40, 0.25]

def get_curriculum_weights(loader_step):
    step = float(loader_step)
    if step >= CURRICULUM_STEPS:
        return list(CURRICULUM_END_WEIGHTS)
    fraction = step / CURRICULUM_STEPS
    return [
        start + (end - start) * fraction
        for start, end in zip(CURRICULUM_START_WEIGHTS, CURRICULUM_END_WEIGHTS)
    ]

def get_average_curriculum_weights(loader_step):
    """Average mixture weights over steps [0, loader_step] — used on resume to
    estimate how many samples each source has already served."""
    step = float(loader_step)
    if step == 0:
        return list(CURRICULUM_START_WEIGHTS)
    if step >= CURRICULUM_STEPS:
        # During the linear ramp the average weight is the start/end midpoint;
        # blend that with the post-ramp plateau proportionally to time spent in each.
        ramp_fraction = CURRICULUM_STEPS / step
        post_fraction = 1.0 - ramp_fraction
        return [
            (start + end) / 2.0 * ramp_fraction + end * post_fraction
            for start, end in zip(CURRICULUM_START_WEIGHTS, CURRICULUM_END_WEIGHTS)
        ]
    else:
        curr = get_curriculum_weights(step)
        return [
            (start + current) / 2.0
            for start, current in zip(CURRICULUM_START_WEIGHTS, curr)
        ]


# ── Reasoning-depth sampling ─────────────────────────────────────────────────
# The reasoning-loop depth is drawn uniformly per micro-step instead of following
# a fixed curriculum. Because the model never knows how many steps it gets, every
# step's slot state must be a viable answer — which is what makes extra steps
# improve the prediction rather than collapse into a copy of step 1. At inference
# the depth is INFERENCE_DEPTH (trm/config.py), the serving default.
# Only the looping arches draw it, through their `training_depth` hook; the trainer
# asks the model, and the plain stack has no depth (#316).

def sample_reasoning_depth(micro_step):
    """Uniform depth in [1, MAX_STEPS_LIMIT], derived deterministically from the
    micro-step so resumed runs replay the exact same depth sequence."""
    rng = np.random.default_rng(DATA_SEED * 1_000_003 + micro_step)
    return int(rng.integers(1, MAX_STEPS_LIMIT + 1))
