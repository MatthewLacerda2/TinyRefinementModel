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

# Persistent compilation cache (#204). Every process used to compile from scratch:
# each supervisor relaunch, test run, smoke and instrument. It makes nothing
# faster once compiled — a cold-start saving only, and we pay cold starts
# constantly. Measured on the RTX 2060: the plain stack's first grad step compiles
# in 15.5s cold and loads in 3.5s warm, with a bit-identical loss. Set here rather
# than in each entry point because every entry point imports this module, and the
# flag works after jax is imported as long as nothing has compiled yet.
#   * On the SSD beside the runs (hot tier), never the HDD.
#   * Bounded: the key includes the JAX/XLA version, so an upgrade orphans the
#     whole previous set, and / runs close to full.
#   * JAX_COMPILATION_CACHE_DIR in the environment wins, as for any JAX flag.
COMPILATION_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "runs", ".jax_cache")
COMPILATION_CACHE_MAX_BYTES = 2 * 1024**3
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    import jax
    jax.config.update("jax_compilation_cache_dir", COMPILATION_CACHE_DIR)
    jax.config.update("jax_compilation_cache_max_size", COMPILATION_CACHE_MAX_BYTES)

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
# Env-overridable only so tests/apparatus/test_trainer_end_to_end.py can run the real
# trainer at toy size on CPU; both shape the param tree, so a resume checks them (#317).
LATENT_DIM = int(os.environ.get("LATENT_DIM", "960"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
# Padded to a multiple of 128 (tensor-core friendly) above the tokenizer's real
# n_vocab. With r50k_base (50257) that is 50304; this is the model's single biggest
# VRAM line (embedding + tied LM head), so the smaller vocab is the headline saving.
# Must be ≥ the tokenizer's n_vocab — update both together if TOKENIZER_NAME changes.
VOCAB_SIZE = 50304
# 15 heads → head_dim = 960/15 = 64, the tensor-core-clean size on Turing f16.
# (16 heads would give head_dim 60, not a multiple of 8 → XLA pads to 64: you pay
# near-1024 attention cost for 960 of width. Avoid.) Verified end-to-end: refiner
# asserts pass (dim%heads==0, head_dim even for RoPE).
NUM_HEADS = int(os.environ.get("NUM_HEADS", "15"))

# Architecture selector (env-overridable so a run is chosen at launch, not by a
# code edit):
#   "refiner"  — Plan A CausalRefiner: causal within-window depth recurrence.
#                RETIRED as the default on 2026-09-12. The mechanism works on
#                sequential composition (findings 2026-06-13 / 06-16 / 06-18,
#                unretracted) and is actively SUPPRESSED on language: the trained
#                gate routes to 6 of 960 channels on prose, the second refine pass
#                costs 5.7 nats at 0.66B and nothing at 3.99B, and bounding the
#                activation scale does not recover it. See
#                docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md.
#                Kept selectable: it is the architecture of the 4B champion, and
#                MODEL_ARCH=refiner is required to load or resume it.
#   "reasoner" — UniversalReasoner, the cross-window-hunch baseline. The hunch
#                is proven inert (finding 2026-06-13), so this is effectively a
#                vanilla random-depth transformer, kept as the control —
#                select it explicitly (MODEL_ARCH=reasoner) for control runs.
#   "plain"    — PlainTransformer: N distinct causal blocks, no loop, no gate, no
#                time signal, no depth dial. The default since depth recurrence
#                was retired.
# The arches have different param trees, so a checkpoint from one cannot be resumed
# by another — resuming the 4B champion now requires MODEL_ARCH=refiner.
MODEL_ARCH = os.environ.get("MODEL_ARCH", "plain")
_KNOWN_ARCHES = ("plain", "refiner", "reasoner")
if MODEL_ARCH not in _KNOWN_ARCHES:
    # Fail closed at import (#104): the selector otherwise falls through to a
    # default, so a typo would silently train the wrong architecture for the
    # whole run — the one failure mode a launch banner does not reliably catch.
    raise SystemExit(
        f"MODEL_ARCH={MODEL_ARCH!r} is not a known architecture; "
        f"use one of {', '.join(_KNOWN_ARCHES)} (unset defaults to 'plain')")

# Optimizer selector (#26). Same fail-closed contract as MODEL_ARCH: a typo must
# not silently train a whole run on the wrong optimizer.
#   muon  — THE DEFAULT since 2026-09-19 (#388): orthogonalized momentum for the 2-D
#           weight matrices, AdamW for the token embedding, norms and biases. It
#           reaches a fixed held-out CE in 1.32x fewer tokens than AdamW at AdamW's
#           own best peak (docs/findings/2026-09-18-muon-holds-1-32x-over-adamw-at-its-best-lr.md).
#   adamw — the recipe every run before that used (trm/train/optimizers.py).
# The embedding is 2-D but is a lookup table, not a linear map, so Muon must NOT
# orthogonalize it; the partition lives in optimizers.muon_partition and
# tests/core/test_muon_partition.py holds it, because a wrong partition trains
# happily and silently.
TRM_OPTIMIZER = os.environ.get("TRM_OPTIMIZER", "muon")
if TRM_OPTIMIZER not in ("adamw", "muon"):
    raise SystemExit(
        f"TRM_OPTIMIZER={TRM_OPTIMIZER!r} is not one of adamw, muon — refusing to start "
        f"rather than fall through to a default (#26)."
    )
# Muon's update has RMS ~1 per element after Newton-Schulz and the sqrt(rows/cols)
# factor, so it wants a much larger LR than Adam's. The matrix partition runs the
# shared schedule times this multiplier; the embedding/norm partition keeps the
# schedule as is. 16.667 on the 6e-4 peak puts the matrices at 1e-2, the value the
# #26 sweep chose (x25/x50/x100/x200 on 1e-4) and #382 confirmed at the new peak.
MUON_LR_MULT = float(os.environ.get("MUON_LR_MULT", "16.666667"))

# The optimizer's remaining knobs, named so that none is a library default nobody can
# read (#358): an optax upgrade that moved one would have changed the recipe silently.
# Each is what every run so far trained with, so every existing config resolves the same.
#   ADAM_B2 is the memory of Adam's variance estimate, 1/(1-b2) opt steps: ~1000 at
#   0.999, ~20 at 0.95 (#359 asks which).
ADAM_B1 = float(os.environ.get("ADAM_B1", "0.9"))
ADAM_B2 = float(os.environ.get("ADAM_B2", "0.999"))
ADAM_EPS = float(os.environ.get("ADAM_EPS", "1e-8"))
# Decoupled weight decay on the >=2-D params, multiplied by the LR (so coupled to it, #360).
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-2"))
# The global-norm clip on the accumulation window's MEAN gradient. The trainer logs that
# norm (applied_grad_norm, #180) and whether the clip bit (clip_active).
CLIP_NORM = float(os.environ.get("CLIP_NORM", "1.0"))
# Muon's own (optax.contrib.scale_by_muon): momentum, Newton-Schulz iterations, the
# normalization epsilon, Nesterov. The Newton-Schulz coefficients are #375's subject.
MUON_BETA = float(os.environ.get("MUON_BETA", "0.95"))
MUON_NS_STEPS = int(os.environ.get("MUON_NS_STEPS", "5"))
MUON_EPS = float(os.environ.get("MUON_EPS", "1e-8"))
MUON_NESTEROV = os.environ.get("MUON_NESTEROV", "1") == "1"
# Clean micro-steps before the f16 loss scaler tries a larger S (#199). Each probe that
# overflows discards a micro-step: ~200 of 65,536 in a 512-step pair. PyTorch's default
# is 2,000; #368 asks whether ours should move. Named here, value unchanged.
LOSS_SCALE_GROWTH_INTERVAL = int(os.environ.get("LOSS_SCALE_GROWTH_INTERVAL", "256"))

# Normalize each residual BRANCH's output before it is added back ("sandwich" /
# post-norm, as in Gemma 2). The pre-norms bound what goes INTO attention and the
# MLP; nothing bounds what comes out, and on the 4B champion that is measurably a
# problem: encoder block 7's SwiGLU product reaches 39,104 on code from a norm2
# input of 6 (SwiGLU multiplies two projections, each ~5x larger on code-shaped
# input), and down_proj carries it to 64,896 -- 99.1% of f16's 65,504 ceiling.
# That is the root cause of #229's whole-window NaN. See #235.
#
# Default OFF: it changes what the model is, so no stored checkpoint survives it
# and it must earn its place through a matched ablation before a run adopts it.
POST_NORM = os.environ.get("POST_NORM", "0") == "1"

# PlainTransformer depth. 8 matched what the refiner ran at depth 1 (7 encoder
# blocks + one refine pass). It was 9 between 2026-09-13 and 2026-09-20, on the
# allocator numbers for batch 1 (8 layers left 771 MiB of arena headroom, 9 left
# 446, 10 left 121).
#
# Back to 8 for the base run (#385, owner 2026-09-20): the depth pays for itself
# only if the card cannot spend the memory better, and it can. Measured on the real
# trainer, 40 opt steps each with checkpoints every 16 and the validation probe
# running (instruments/bench_train_step never ran either, which is why #24's batch-2
# win was real and irrelevant):
#   9 layers, batch 1   4,424 tok/s   arena peak 4058 MiB, 825 MiB headroom
#   8 layers, batch 2   6,149 tok/s   arena peak 4502 MiB, 381 MiB headroom  <- this
# +39% throughput, which is 531M tokens a day against 382M. The ninth layer costs
# more than a day of every four. The fallback if a long run OOMs is 8 layers at
# batch 1, which is both cheaper in memory than what shipped before and faster.
PLAIN_LAYERS = int(os.environ.get("PLAIN_LAYERS", "8"))

# ── Retired architectures ───────────────────────────────────────────────────────
# Knobs only the refiner and the reasoner read, kept so their checkpoints still load
# (MODEL_ARCH=refiner / reasoner). None of them shapes the plain model; #292 deletes
# this block once a plain champion exists.
NUM_BLOCKS = 8     # reasoner
SHARED_SLOTS = 32  # reasoner
# The deepest sampled training depth, for the arches that have one: since #316 the
# trainer asks the model for its depth, and plain answers None.
MAX_STEPS_LIMIT = 8
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
# ── end of retired architectures ────────────────────────────────────────────────

# Training
# BATCH_SIZE and ACCUMULATION_STEPS move together, always keeping their product
# fixed (#24): the optimizer + global-norm clip over 138.7M params costs a flat
# ~69ms per micro-step regardless of batch — 25% of a batch-1 step — so fewer,
# fatter micro-steps amortize it. #24 measured that on an idle card with
# instruments.bench_train_step: +43% at depth 4 (5.0k -> 7.2k tok/s) and +40% at
# depth 8 (4.2k -> 5.9k), and flipped this pair to 2/64.
#
# BATCH_SIZE IS 2 since 2026-09-20, at 8 layers. It was 1 for as long as the stack
# was 9 layers deep, and for good reason: batch 2 OOMed the real trainer on its first
# optimizer step at dim960/depth8 in 2026-08-13, and the 2026-09-13 re-measurement
# found 45 MiB of headroom at 8 layers and -371 at 9 — the lever was dead at dim 960.
# Muon's 380 MiB (#388) and #316's single compiled program changed that arithmetic.
#
# What flipped it is a real trainer launch, which is what the old note here demanded
# and what a bench number could never give: 40 opt steps at 8 layers and batch 2,
# checkpoints every 16 steps, validation probe running, 6,149 tok/s against 4,424 for
# 9 layers at batch 1, arena peak 4502 of 4883 MiB (381 MiB headroom, matching the
# #385 smoke's 382). Note that instruments.vram_headroom_smoke still will NOT catch a
# regression here: it defaults to --batch 1 and samples nvidia-smi under the
# `platform` allocator rather than the trainer's preallocated arena.
#
# 381 MiB is thinner than the 834 MiB the only completed 10-day run had, so this is
# the knob the supervisor's fit gate (#168) is for; if a long run OOMs, drop to
# batch 1 at 8 layers rather than adding the layer back.
#
# The product with ACCUMULATION_STEPS stays 128, so tokens per optimizer step, the LR
# schedule, the token budget and the learning dynamics are identical either way: same
# model, same run, fewer and fatter micro-steps amortizing the flat ~69ms optimizer +
# global-norm clip over 138.7M params (#24).
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
if 128 % BATCH_SIZE:
    raise SystemExit(f"BATCH_SIZE={BATCH_SIZE} must divide 128, so tokens per opt step stay fixed")
ACCUMULATION_STEPS = 128 // BATCH_SIZE
# Target tokens consumed per optimizer step: each micro-step scores two
# MAX_SEQ_LEN prediction windows, ACCUMULATION_STEPS micro-steps make one opt step.
TOKENS_PER_OPT_STEP = ACCUMULATION_STEPS * BATCH_SIZE * 2 * MAX_SEQ_LEN

# Held-out evaluation reads a fixed number of *rows* (prediction-window pairs)
# and scores them ONE ROW AT A TIME, deliberately ignoring BATCH_SIZE. Sizing or
# chunking the eval slice by a training throughput knob would silently redefine
# what "val CE" means and break comparability with every number already recorded
# (the champion's 4.7092, the #17 noise floor of sigma~0.03, the time machine's
# +/-0.06 reproduction check). Eval is a handful of rows, so there is nothing to
# gain by batching it — and batch-1 scoring is also the shape every stored
# checkpoint of both arches was written at.
#
# CHANGEOVER 2026-09-15 (#184): 4 -> 64 rows. Four rows gave val CE a per-probe
# noise of ~0.011 nats per interval, more than twice the plateau detector's
# 0.005 bar, so a detector reading it read noise. The cost is comparability:
# a plain run's val CE is NOT comparable to any number recorded before this
# line (the refiner champion's 3.6474, the July 4.7092, the #17 sigma~0.03
# floor) — those were measured on the first 4 rows of the same slice. The new
# probe's own sigma is measured, not assumed: instruments/probe_sigma.py.
EVAL_ROWS = int(os.environ.get("EVAL_ROWS", "64"))

# The plateau detector reads HELD-OUT CE (#184): train CE is moved by the
# curriculum underneath it (the #157 run's train CE rose 3.20 -> 3.36 while the
# model improved). A plateau is "the windowed val CE has not improved by
# PLATEAU_MIN_DELTA for PLATEAU_PATIENCE opt steps". The bar must sit above the
# probe's own noise or the detector is a coin flip. Measured 2026-09-15
# (instruments/probe_sigma.py, one plain checkpoint, 6 disjoint slices):
#   4 rows  mean 4.74  sigma 0.195   (readings 4.49 .. 5.06 — which rows you got)
#   64 rows mean 4.78  sigma 0.097
# That is the level error between slices; documents are heavy-tailed, so 16x
# the rows only halves it. The detector watches ONE fixed slice, whose
# probe-to-probe jitter during training measured 0.011 nats at 4 rows (#184);
# 0.01 is set at that jitter, above what 64 rows should show, and 2x the old
# 0.005 that sat below it.
PLATEAU_MIN_DELTA = float(os.environ.get("PLATEAU_MIN_DELTA", "0.01"))
PLATEAU_PATIENCE = int(os.environ.get("PLATEAU_PATIENCE", "400"))

# Planned token budget for the run (#83) — env-overridable per run like the seeds,
# recorded in run_metadata.json. Drives schedules.DECAY_STEPS so the LR cosine
# bottoms out when training ends instead of at a constant chosen for past short
# runs (15000 opt steps ≈ 2.0B tokens — an anneal that would sit frozen at the
# 1e-6 floor for most of a longer run). Accepts plain ints or scientific notation
# ("2e9"). Unset → the historical 15000-step horizon, so existing configs and the
# golden run resolve unchanged.
_TOKEN_BUDGET_ENV = os.environ.get("TRAIN_TOKEN_BUDGET")
TRAIN_TOKEN_BUDGET = int(float(_TOKEN_BUDGET_ENV)) if _TOKEN_BUDGET_ENV else None
# The document separator prefill writes between documents: r50k_base's end-of-text.
EOT_TOKEN_ID = 50256
# The id masked out of attention keys and loss targets. It used to be EOT_TOKEN_ID,
# and that plumbed the document separator into the pad sink (#373): EOT was never a
# target (no learned "the document ends here" — a model that cannot stop talking) and
# never a key (tokens after a boundary attended into the previous document with no
# marker that one passed). 50257 is the first id past r50k's real vocabulary, inside
# the padded table and never written by prefill, so as the pad it leaves EOT an
# ordinary token. Adopted as the default for the base run (owner, 2026-09-20) without
# its matched pair: #373 is a bug, not a hypothesis — a base run that never learns to
# end a document is broken whatever the pair would have said. Set PAD_TOKEN_ID=50256
# to reproduce a run recorded under the old convention.
PAD_TOKEN_ID = int(os.environ.get("PAD_TOKEN_ID", "50257"))
if PAD_TOKEN_ID not in (EOT_TOKEN_ID, 50257):
    raise SystemExit(f"PAD_TOKEN_ID={PAD_TOKEN_ID}: use {EOT_TOKEN_ID} (historical) or 50257 (#373)")

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
#   DATA_SEED  — data-pipeline randomness: the start offset into each source
#                (under 1,025 tokens), the mixture draws, per-step depth sampling.
#                NOT the document order. Every source is read front to back, so two
#                seeds see nearly the same documents in the same order, and a pair's
#                seed spread is init variance, not data variance (#378).
#   MODEL_SEED — parameter initialization (the nnx.Rngs the trainer builds
#                the model with).
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
MODEL_SEED = int(os.environ.get("MODEL_SEED", "42"))
