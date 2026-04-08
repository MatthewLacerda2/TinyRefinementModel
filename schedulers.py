import jax
import optax

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4, 
    peak_value=3e-4,
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=1e-4
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

def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)

weight_decay_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=500, 
    decay_steps=2000, 
    end_value=1e-2
)

optimizer_chain = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_schedule,
            weight_decay=weight_decay_schedule,
            mask=weight_decay_mask,
        ),
    ),
    every_k_schedule=128,
    use_grad_mean=True
)