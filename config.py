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
# Padded to a multiple of 128 (tensor-core friendly) above the tokenizer's real
# n_vocab. With r50k_base (50257) that is 50304; this is the model's single biggest
# VRAM line (embedding + tied LM head), so the smaller vocab is the headline saving.
# Must be ≥ the tokenizer's n_vocab — update both together if TOKENIZER_NAME changes.
VOCAB_SIZE = 50304
NUM_HEADS = 16
NUM_GROUPS = NUM_HEADS // 4

# Architecture selector (env-overridable so a run is chosen at launch, not by a
# code edit):
#   "refiner"  — Plan A CausalRefiner: causal within-window depth recurrence.
#                The default: it is the live bet, proven on the toy gate and
#                through pretraining (findings 2026-06-13 / 06-16 / 06-18).
#   "reasoner" — UniversalReasoner, the cross-window-hunch baseline. The hunch
#                is proven inert (finding 2026-06-13), so this is effectively a
#                vanilla random-depth transformer, kept as the control —
#                select it explicitly (MODEL_ARCH=reasoner) for control runs.
# The two arches have different param trees, so a checkpoint from one cannot be
# resumed by the other — resuming an old reasoner run now requires the env var.
MODEL_ARCH = os.environ.get("MODEL_ARCH", "refiner")
# Blockwise memory-lean attention for the refiner (#66): removes the O(seq²)
# score/probability transients that dominate the grad-step peak (mem_profile).
# Opt-in until the dim960 GPU fit-test + wall-clock bench pass on the box;
# same math as stock attention up to float summation order.
CHUNKED_ATTENTION = os.environ.get("CHUNKED_ATTENTION", "0") == "1"
# Plan A: number of causal encoder layers beneath the single shared refine block
# (which is looped up to MAX_STEPS_LIMIT times). Tuned to land the param count
# near the reasoner baseline; init prints the actual count for both arches.
REFINER_ENCODER_LAYERS = int(os.environ.get("REFINER_ENCODER_LAYERS", "7"))

# Refiner serving depth. The dense 1→8 sweep
# (docs/findings/2026-06-19-plan-a-depth-dense-sweep.md) put the accuracy
# plateau at ~d6 (peak d7; d6–d8 inside seed noise), and pretraining shifts the
# curve LEFT — the same ceiling in fewer loops. Loops past the knee buy nothing
# measurable and cost a full pass of the shared block each, so inference/eval
# tooling defaults here. Training is a separate decision and still samples up to
# MAX_STEPS_LIMIT (pre-registered with the sweep: "MAX_STEPS_LIMIT=8 stays").
INFERENCE_DEPTH = int(os.environ.get("INFERENCE_DEPTH", "6"))

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

# Seeds — env-overridable per run (#17: the seed-variance noise floor needs
# same-config runs differing ONLY in seed). Both are recorded in
# run_metadata.json so every run stays reproducible.
#   DATA_SEED  — data-pipeline randomness (start-offset augmentation, mixture
#                draws, per-step depth sampling).
#   MODEL_SEED — parameter initialization (the nnx.Rngs the trainer builds
#                the model with).
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
MODEL_SEED = int(os.environ.get("MODEL_SEED", "42"))
