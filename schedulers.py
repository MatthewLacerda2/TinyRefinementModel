import optax
from train_local import ACCUMULATION_STEPS

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=3e-4,
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=1e-5
)

ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=2e-4
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=4e-3
)

#TODO: use a raw lambda or rather make the model figure it out
diversity_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.5, 
    peak_value=0.5, 
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=0.5
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale(1.0 / ACCUMULATION_STEPS),   # normalize before accumulation so adamw sees the average, not the sum
    optax.apply_every(ACCUMULATION_STEPS),
    optax.adamw(learning_rate=learning_schedule),
)