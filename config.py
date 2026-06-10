# Single source of truth for all architecture and training constants.
# Keep (most) values powers of 2 if you know what's good for you.

# Architecture
LATENT_DIM = 512
NUM_BLOCKS = 8
SHARED_SLOTS = 32
MAX_SEQ_LEN = 512
VOCAB_SIZE = 100352
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4

# Training
MAX_STEPS_LIMIT = 8
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
PAD_TOKEN_ID = 100257
