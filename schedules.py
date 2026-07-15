import numpy as np
import optax

from config import MAX_STEPS_LIMIT, DATA_SEED, TOKENS_PER_OPT_STEP, TRAIN_TOKEN_BUDGET

# Warmup is absolute: it stabilizes the optimizer's first moments, a fixed-cost
# phase that does not grow with the run.
WARMUP_STEPS = 1000

# The LR anneal's horizon must match the run length (#83). DECAY_STEPS derives
# from the planned token budget (config.TRAIN_TOKEN_BUDGET); with no budget set
# it stays at the historical 15000 opt steps, so existing configs and the golden
# run resolve unchanged.
_DEFAULT_DECAY_STEPS = 15000


def resolve_decay_steps(token_budget, tokens_per_opt_step=TOKENS_PER_OPT_STEP):
    """Opt-step horizon for the run: token budget / tokens per opt step
    (None → the historical default). A budget that doesn't clear warmup is a
    config error, not a run worth starting — fail loud."""
    if token_budget is None:
        return _DEFAULT_DECAY_STEPS
    steps = round(token_budget / tokens_per_opt_step)
    if steps <= WARMUP_STEPS:
        raise ValueError(
            f"TRAIN_TOKEN_BUDGET={token_budget} resolves to {steps} opt steps, "
            f"inside the {WARMUP_STEPS}-step warmup — the cosine would never decay."
        )
    return steps


def build_learning_schedule(decay_steps):
    """The run's LR schedule at an explicit horizon; module-level
    learning_schedule is this at the resolved DECAY_STEPS."""
    return optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-4,
        warmup_steps=WARMUP_STEPS,
        decay_steps=decay_steps,
        end_value=1e-6
    )


DECAY_STEPS = resolve_decay_steps(TRAIN_TOKEN_BUDGET)
learning_schedule = build_learning_schedule(DECAY_STEPS)

# The λ anneals deliberately do NOT follow DECAY_STEPS (#83): they relax
# regularization pressure over early training — absolute-step optimizer
# dynamics, like warmup — not a function of the run's energy budget. On a
# longer run they sit at their end values from 15k on, which is today's
# behavior made explicit rather than silently stretched. (For the refiner
# arch both terms are exactly zero anyway.)
LAMBDA_DECAY_STEPS = 15000

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.05,
    warmup_steps=WARMUP_STEPS,
    decay_steps=LAMBDA_DECAY_STEPS,
    end_value=0.001
)

diversity_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1.0,
    warmup_steps=WARMUP_STEPS,
    decay_steps=LAMBDA_DECAY_STEPS,
    end_value=0.1
)

weight_decay_schedule = optax.constant_schedule(1e-2)


# ── Data curriculum ──────────────────────────────────────────────────────────
# Mixture weights ramp linearly from web-heavy toward a code/math-heavy blend
# over the first CURRICULUM_STEPS optimizer steps, then hold steady.

CURRICULUM_STEPS = 10000.0
# Endpoints over the (web, code, math) sources, in DataMixer source order.
CURRICULUM_START_WEIGHTS = [0.85, 0.10, 0.05]
CURRICULUM_END_WEIGHTS = [0.35, 0.40, 0.25]
# SFT-phase mixture over (chat, web, code, math) — chat-led with pretrain replay.
# Single source of truth: the trainer builds its mixer AND prints from this list.
SFT_MIX_WEIGHTS = [0.70, 0.15, 0.10, 0.05]

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
# the depth is always MAX_STEPS_LIMIT.

def sample_reasoning_depth(micro_step):
    """Uniform depth in [1, MAX_STEPS_LIMIT], derived deterministically from the
    micro-step so resumed runs replay the exact same depth sequence."""
    rng = np.random.default_rng(DATA_SEED * 1_000_003 + micro_step)
    return int(rng.integers(1, MAX_STEPS_LIMIT + 1))
