# The compute policy and the constants that are not knobs. The knobs (every value a
# launch can set from the environment) are the fields of trm.settings.Config, read once,
# recorded whole and passed down (#475); nothing here re-exports one.
# Keep (most) values powers of 2 if you know what's good for you.

import os

import jax.numpy as jnp

from trm.settings import CONFIG

# Dtype policy: training runs on an RTX 2060 (Turing), which has no bfloat16
# support — float16 compute is the deliberate, permanent policy here.
# Parameters are stored float32 (NNX's default param_dtype); COMPUTE_DTYPE only
# sets the computation dtype of the matmul-heavy layers. If f16 gradient
# underflow ever becomes a problem, the fix is optax loss scaling, not a
# dtype change. FORCE_F32_COMPUTE (trm/settings.py) is the test suite's way out.
COMPUTE_DTYPE = jnp.float32 if CONFIG.FORCE_F32_COMPUTE else jnp.float16
# What accumulates stays f32 (CLAUDE.md's dtype rule): the residual stream of the plain
# stack (Config.RESIDUAL_DTYPE, #357; tests/core/test_adopted_recipe.py holds it), the
# gradient accumulator and Adam's variance. Not f32, and why: Adam's and Muon's first
# moments are stored bf16 (trm/train/optimizers.py, #37).

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
#   * JAX_COMPILATION_CACHE_DIR in the environment wins, as for any JAX flag: it is
#     JAX's own variable, not one of our knobs.
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

# Padded to a multiple of 128 (tensor-core friendly) above the tokenizer's real
# n_vocab. With r50k_base (50257) that is 50304; this is the model's single biggest
# VRAM line (embedding + tied LM head), so the smaller vocab is the headline saving.
# Must be ≥ the tokenizer's n_vocab — update both together if TOKENIZER_NAME changes.
VOCAB_SIZE = 50304

# Tokenizer — single source of truth. prefill, inference, and the transcript dump
# all import this name so the encoding can never drift between tokenizing the corpus
# and serving the model. Switched cl100k_base → r50k_base (#21): the GPT-2/GPT-3
# family 50k vocab halves VOCAB_SIZE (100352→50304), freeing the biggest VRAM line
# for width/data. Trade-off: r50k packs text less tightly than cl100k, so a fixed
# token budget covers less raw text. Changing this requires re-tokenizing the corpus
# and updating VOCAB_SIZE/PAD_TOKEN_ID to match the new encoding.
TOKENIZER_NAME = "r50k_base"
# The document separator prefill writes between documents: r50k_base's end-of-text.
# Not the pad any more (PAD_TOKEN_ID, #373).
EOT_TOKEN_ID = 50256
