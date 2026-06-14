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
LATENT_DIM = 512
NUM_BLOCKS = 8
SHARED_SLOTS = 32
MAX_SEQ_LEN = 512
VOCAB_SIZE = 100352
NUM_HEADS = 16
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
PAD_TOKEN_ID = 100257

# Seed for data-pipeline randomness (start-offset augmentation, mixture draws).
# Recorded in run_metadata.json so runs are reproducible.
DATA_SEED = 42
