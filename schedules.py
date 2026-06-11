import numpy as np
import optax

from config import MAX_STEPS_LIMIT, DATA_SEED

WARMUP_STEPS = 1000
DECAY_STEPS = 15000

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=1e-4,
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=1e-6
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.05, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=0.001
)

diversity_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=1.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
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
