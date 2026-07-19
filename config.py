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
_KNOWN_ARCHES = ("refiner", "reasoner")
if MODEL_ARCH not in _KNOWN_ARCHES:
    # Fail closed at import (#104): the selector otherwise falls through to a
    # default, so a typo would silently train the wrong architecture for the
    # whole run — the one failure mode a launch banner does not reliably catch.
    raise SystemExit(
        f"MODEL_ARCH={MODEL_ARCH!r} is not a known architecture; "
        f"use one of {', '.join(_KNOWN_ARCHES)} (unset defaults to 'refiner')")
# Refiner time signal (#86): how each refinement pass is told which step it is.
#   "sinusoidal" — continuous diffusion-style step encoding, defined at ANY step,
#                  so inference depth is an open dial (finding
#                  2026-07-18-sinusoidal-time-signal-depth-extrapolates.md:
#                  parity with the table at trained depths, +0.11 from
#                  extrapolated loops under length shift). The default — what
#                  the base run trains.
#   "table"      — the learned per-step embedding; rows end at MAX_STEPS_LIMIT
#                  and the signal clamps past them (chance + NaN, same finding).
#                  Required to RESUME refiner checkpoints from before this flip:
#                  the two modes have different param trees.
TIME_SIGNAL = os.environ.get("TIME_SIGNAL", "sinusoidal")
if TIME_SIGNAL not in ("table", "sinusoidal"):
    # Same fail-closed contract as MODEL_ARCH (#104): a typo must not silently
    # train a different model for a whole run.
    raise SystemExit(
        f"TIME_SIGNAL={TIME_SIGNAL!r} is not a known time signal; "
        f"use one of table, sinusoidal (unset defaults to 'sinusoidal')")
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
# Target tokens consumed per optimizer step: each micro-step scores two
# MAX_SEQ_LEN prediction windows, ACCUMULATION_STEPS micro-steps make one opt step.
TOKENS_PER_OPT_STEP = ACCUMULATION_STEPS * BATCH_SIZE * 2 * MAX_SEQ_LEN

# Planned token budget for the run (#83) — env-overridable per run like the seeds,
# recorded in run_metadata.json. Drives schedules.DECAY_STEPS so the LR cosine
# bottoms out when training ends instead of at a constant chosen for past short
# runs (15000 opt steps ≈ 2.0B tokens — an anneal that would sit frozen at the
# 1e-6 floor for most of a longer run). Accepts plain ints or scientific notation
# ("2e9"). Unset → the historical 15000-step horizon, so existing configs and the
# golden run resolve unchanged.
_TOKEN_BUDGET_ENV = os.environ.get("TRAIN_TOKEN_BUDGET")
TRAIN_TOKEN_BUDGET = int(float(_TOKEN_BUDGET_ENV)) if _TOKEN_BUDGET_ENV else None
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
