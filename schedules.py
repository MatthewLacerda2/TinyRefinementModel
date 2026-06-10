import optax

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

ponder_lambda_schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.0),
        optax.linear_schedule(init_value=0.0, end_value=0.02, transition_steps=3000),
        optax.constant_schedule(0.02)
    ],
    boundaries=[2000, 5000] # Zero penalty for first 2k steps, ramps up over next 3k steps
)

weight_decay_schedule = optax.constant_schedule(1e-2)


# ── Data curriculum ──────────────────────────────────────────────────────────
# Mixture weights ramp from web-heavy toward a code/math-heavy blend over the
# first CURRICULUM_STEPS optimizer steps, then hold steady.

CURRICULUM_STEPS = 10000.0

def get_curriculum_weights(loader_step):
    step = float(loader_step)
    if step >= CURRICULUM_STEPS:
        return [0.35, 0.40, 0.25]
    fraction = step / CURRICULUM_STEPS
    w_web = 0.85 - 0.50 * fraction
    w_code = 0.10 + 0.30 * fraction
    w_math = 0.05 + 0.20 * fraction
    return [w_web, w_code, w_math]

def get_average_curriculum_weights(loader_step):
    """Average mixture weights over steps [0, loader_step] — used on resume to
    estimate how many samples each source has already served."""
    step = float(loader_step)
    if step == 0:
        return [0.85, 0.10, 0.05]
    if step >= CURRICULUM_STEPS:
        curriculum_fraction = CURRICULUM_STEPS / step
        post_fraction = 1.0 - curriculum_fraction
        avg_web = 0.60 * curriculum_fraction + 0.35 * post_fraction
        avg_code = 0.25 * curriculum_fraction + 0.40 * post_fraction
        avg_math = 0.15 * curriculum_fraction + 0.25 * post_fraction
        return [avg_web, avg_code, avg_math]
    else:
        curr = get_curriculum_weights(step)
        return [
            (0.85 + curr[0]) / 2.0,
            (0.10 + curr[1]) / 2.0,
            (0.05 + curr[2]) / 2.0
        ]

def get_curriculum_steps(train_opt_step):
    """Reasoning-loop depth curriculum: grow max steps as training progresses."""
    if train_opt_step < 1000:
        return 1
    elif train_opt_step < 4000:
        return 2
    elif train_opt_step < 8000:
        return 4
    else:
        return 8
