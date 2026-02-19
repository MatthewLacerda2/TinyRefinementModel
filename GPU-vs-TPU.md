# GPU

NUM_DIM must be multiple of 8
BATCH_SIZE of 8 is the most the GPU can handle right now
MAX_SEQ_LEN 512, or OOM
GPT-2 vocabulary or OOM
float16 is needed

# TPU

NUM_DIM must be multiple of 128
BATCH_SIZE of 128 is the minimum you should use
MAX_SEQ_LEN 2048 and that's lowballing
GPT-3 set and forget
bfloat16, and forget

Use Orbax Checkpointer