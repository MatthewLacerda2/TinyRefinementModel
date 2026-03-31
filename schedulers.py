import optax

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=2e-4,
    warmup_steps=500, 
    decay_steps=1000, 
    end_value=1e-5
)

ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=500, 
    decay_steps=1000, 
    end_value=2e-4
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=500, 
    decay_steps=1000, 
    end_value=4e-3
)

#TODO: use a raw lambda or rather make the model figure it out
diversity_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.5, 
    peak_value=0.5, 
    warmup_steps=500, 
    decay_steps=1000, 
    end_value=0.5
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=learning_schedule),
    optax.apply_every(128),
)