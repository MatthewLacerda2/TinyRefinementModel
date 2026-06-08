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
