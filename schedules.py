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


# ── Fixed auxiliary loss weights (used in grad_step.loss_fn) ─────────────────

# Penalty on the second segment's CE exceeding the first's: pushes the carried
# hunch state to actually help (refine) rather than hurt.
REFINEMENT_LOSS_WEIGHT = 0.08

# Small direct CE weight on the first (anchor) segment beyond its base CE term.
ANCHOR_CE_WEIGHT = 0.03


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


# ── Reasoning-depth curriculum ───────────────────────────────────────────────
# (opt-step boundary, reasoning steps used below that boundary); past the last
# boundary the depth is DEPTH_FINAL.

DEPTH_CURRICULUM = [(1000, 1), (4000, 2), (8000, 4)]
DEPTH_FINAL = 8

def get_curriculum_steps(train_opt_step):
    """Reasoning-loop depth curriculum: grow max steps as training progresses."""
    for boundary, depth in DEPTH_CURRICULUM:
        if train_opt_step < boundary:
            return depth
    return DEPTH_FINAL
