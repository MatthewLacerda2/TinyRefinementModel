import optax
from train_local import BATCH_SIZE, MAX_SEQ_LEN 

# def calc_sched():
#     tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
#     TARGET_TOKENS = 60 * 1000 * 1000
#     total_steps = TARGET_TOKENS // tokens_per_step
#     return total_steps

WARMUP = 500
DECAY = 2000

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=2e-4,
    warmup_steps=WARMUP, 
    decay_steps=DECAY, 
    end_value=5e-5
)

# Penalties and alphas stabilize at 1000 steps
ponder_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=1e-4,
    transition_steps=500,
)

forget_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=4e-3,
    transition_steps=500,
)

diversity_lambda_schedule = optax.linear_schedule(
    init_value=0.5,
    end_value=1.0,
    transition_steps=500,
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=learning_schedule),
)