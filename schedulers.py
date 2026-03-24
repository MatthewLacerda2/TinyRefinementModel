import optax
from train_local import BATCH_SIZE, MAX_SEQ_LEN 

def calc_sched():
    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    TARGET_TOKENS = 6000000
    total_steps = TARGET_TOKENS // tokens_per_step
    return total_steps

WARMUP = calc_sched() // 3
DECAY = WARMUP * 2

learning_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, 
    peak_value=2e-4,
    warmup_steps=WARMUP, 
    decay_steps=DECAY, 
    end_value=1e-5
)

ponder_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP, 
    decay_steps=DECAY, 
    end_value=1e-4
)

forget_lambda_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, 
    peak_value=0.0, 
    warmup_steps=WARMUP, 
    decay_steps=DECAY, 
    end_value=4e-3
)

diversity_lambda_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=0.12,
    transition_steps=WARMUP,
    transition_begin=DECAY,
)

semantic_alpha_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=0.5,
    transition_steps=DECAY,
)

optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=learning_schedule),
)