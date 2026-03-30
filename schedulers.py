import optax

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=2e-4,
    warmup_steps=1000, 
    decay_steps=3000, 
    end_value=1e-5
)

ponder_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=1e-4,
    transition_steps=1000,
    transition_begin=3000,
)

forget_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=4e-3,
    transition_steps=1000,
    transition_begin=3000,
)

diversity_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=0.12,
    transition_steps=1000,
    transition_begin=3000,
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=learning_schedule),
)