import optax

WARMUP_STEPS = 1000
DECAY_STEPS = 15000

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=2e-4, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=1e-6
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.3, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=0.01
)


diversity_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.05, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=0.01
)

weight_decay_schedule = optax.constant_schedule(2e-2)
