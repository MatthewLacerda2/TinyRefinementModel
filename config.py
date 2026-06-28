# Single source of truth for all architecture and training constants.
# Keep (most) values powers of 2 if you know what's good for you.

import os
import jax.numpy as jnp

# Dtype policy: training runs on an RTX 2060 (Turing), which has no bfloat16
# support — float16 compute is the deliberate, permanent policy here.
# Parameters are stored float32 (NNX's default param_dtype); COMPUTE_DTYPE only
# sets the computation dtype of the matmul-heavy layers. If f16 gradient
# underflow ever becomes a problem, the fix is optax loss scaling, not a
# dtype change.
# FORCE_F32_COMPUTE is a test-only escape hatch: CPU XLA cannot lower the
# f16-with-f32-accumulation matmuls inside the rematerialized scan, so the
# test suite (which defaults to CPU while the GPU trains) sets it. Never set
# it for training.
COMPUTE_DTYPE = jnp.float32 if os.environ.get("FORCE_F32_COMPUTE") else jnp.float16
PARAM_DTYPE = jnp.float32

def resolve_root(path):
    """abspath for local paths; remote URLs (gs://, s3://, ...) pass through
    untouched — abspath would prepend the cwd and mangle them."""
    if "://" in path:
        return path
    return os.path.abspath(path)

# Architecture
# v1 base-run width: 960 is the widest that fits the 6GB card with bf16 optimizer
# moments (#18) — ~5.3GB peak at depth-8/batch-1, ~0.7GB margin; dim1024 OOMs. Not a
# power of 2, but a clean multiple of 64, and the big VRAM lines (the 50304×dim
# embedding/LM head, the FFN) don't care. What matters is head_dim — see NUM_HEADS.
LATENT_DIM = 960
NUM_BLOCKS = 8
SHARED_SLOTS = 32
MAX_SEQ_LEN = 512
# Padded to a multiple of 128 (tensor-core friendly) above the tokenizer's real
# n_vocab. With r50k_base (50257) that is 50304; this is the model's single biggest
# VRAM line (embedding + tied LM head), so the smaller vocab is the headline saving.
# Must be ≥ the tokenizer's n_vocab — update both together if TOKENIZER_NAME changes.
VOCAB_SIZE = 50304
# 15 heads → head_dim = 960/15 = 64, the tensor-core-clean size on Turing f16.
# (16 heads would give head_dim 60, not a multiple of 8 → XLA pads to 64: you pay
# near-1024 attention cost for 960 of width. Avoid.) Verified end-to-end: refiner
# asserts pass (dim%heads==0, head_dim even for RoPE); baseline GQA NUM_GROUPS=15//4=3,
# and 15%3==0 so the jnp.repeat that expands KV groups tiles cleanly (layers.py:82).
NUM_HEADS = 15
NUM_GROUPS = NUM_HEADS // 4

# Architecture selector (env-overridable so a run is chosen at launch, not by a
# code edit):
#   "reasoner" — UniversalReasoner, the cross-window-hunch baseline (default;
#                the hunch is proven inert, so this is effectively a vanilla
#                random-depth transformer and serves as the control).
#   "refiner"  — Plan A CausalRefiner: causal within-window depth recurrence
#                (docs/findings/2026-06-13-plan-a-depth-recurrence-works.md).
# A refiner run has a different param tree, so it must start fresh (--new-run);
# it cannot resume a reasoner checkpoint.
MODEL_ARCH = os.environ.get("MODEL_ARCH", "reasoner")
# Plan A: number of causal encoder layers beneath the single shared refine block
# (which is looped up to MAX_STEPS_LIMIT times). Tuned to land the param count
# near the reasoner baseline; init prints the actual count for both arches.
REFINER_ENCODER_LAYERS = int(os.environ.get("REFINER_ENCODER_LAYERS", "7"))

# Training
MAX_STEPS_LIMIT = 8
BATCH_SIZE = 1
ACCUMULATION_STEPS = 128
# Padding reuses the tokenizer's end-of-text id (sequences are eot-separated, so the
# pad token and the document separator are the same symbol). r50k_base eot = 50256.
PAD_TOKEN_ID = 50256

# Tokenizer — single source of truth. prefill, inference, and the transcript dump
# all import this name so the encoding can never drift between tokenizing the corpus
# and serving the model. Switched cl100k_base → r50k_base (#21): the GPT-2/GPT-3
# family 50k vocab halves VOCAB_SIZE (100352→50304), freeing the biggest VRAM line
# for width/data. Trade-off: r50k packs text less tightly than cl100k, so a fixed
# token budget covers less raw text. Changing this requires re-tokenizing the corpus
# and updating VOCAB_SIZE/PAD_TOKEN_ID to match the new encoding.
TOKENIZER_NAME = "r50k_base"

# Seed for data-pipeline randomness (start-offset augmentation, mixture draws).
# Recorded in run_metadata.json so runs are reproducible.
DATA_SEED = 42
