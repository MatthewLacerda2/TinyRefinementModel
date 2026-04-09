import jax
import optax

# Target: ~1440 steps (12-hour overnight run at 30s/step)
WARMUP_STEPS = 100
DECAY_STEPS = 500

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=5e-5,    # Slightly lower start since warmup is shorter
    peak_value=3e-4,    # Fast enough to escape local minima
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=5e-5      # Lower floor to help it settle deeply at the end
)

# Ponder and Forget start at 0 during warmup (free exploration)
# Then they "reverse decay" (ramp up) over the next 850 steps
ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=2e-4
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
    end_value=4e-3
)

# Flat 0.5 margin for the diversity loss
diversity_lambda_schedule = optax.constant_schedule(0.5)

def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)

weight_decay_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP_STEPS, 
    decay_steps=DECAY_STEPS, 
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