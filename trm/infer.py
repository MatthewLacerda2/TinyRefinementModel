import argparse
import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import orbax.checkpoint as ocp
import time
from functools import partial

from trm.config import (
    INFERENCE_DEPTH,
    LATENT_DIM,
    MAX_SEQ_LEN,
    MODEL_ARCH,
    EOT_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKENIZER_NAME,
    resolve_root,
)
from trm.model import build_model
from trm.model.contract import LanguageModel
from trm.runtime.layout import CHECKPOINT_ITEMS

from dotenv import load_dotenv
load_dotenv()

# `new_document` is flipped on every Nth generated token. Only the reasoner reads it
# (it resets its cross-window hunch cache, proven inert — docs/findings/
# 2026-06-13-cross-window-hunch-inert.md); plain and refiner ignore it. Kept so the
# reasoner's generation stays what it was.
REASONER_REFRESH_EVERY = 4

# Sampling default. 0.5 was too cold to see the model: it divides the logits, so it
# *doubles* every gap, and with max|logit| already ~21 partway through the base run
# that makes sampling effectively greedy — which is the classic way to turn an
# undertrained LM into a repetition loop ("the ratio of the ratio of the ratio").
# What you read then is the decoder's failure mode, not the weights'.
DEFAULT_TEMPERATURE = 0.7


def build_serving_model(arch=MODEL_ARCH):
    """The architecture MODEL_ARCH selects — the same choice, and the same factory,
    the trainer uses. The seed is irrelevant: the checkpoint overwrites every weight.
    Serving once hardcoded one architecture (#185); tests/core/test_infer_arch.py
    guards the choice.
    """
    return build_model(arch, LATENT_DIM, nnx.Rngs(0))

def run_model_inference(
    model: LanguageModel,
    tokens: jnp.ndarray,
    depth: int = INFERENCE_DEPTH,
    new_document: bool = True,
    logits_at=None,
) -> jnp.ndarray:
    out = model(
        tokens, depth=depth, training=False, new_document=new_document,
        logits_at=logits_at,
    )
    return out.logits

def _temperature_truncate(logits, temperature, top_k, top_p):
    """Scale by temperature first, then truncate — top-p's cutoff must be
    computed on the distribution actually being sampled (HF/nanoGPT/llama.cpp
    convention), not on the untempered one."""
    logits = logits / temperature

    if top_k > 0:
        top_vals, _ = jax.lax.top_k(logits, top_k)
        k_threshold = top_vals[-1]
        logits = jnp.where(logits < k_threshold, -jnp.inf, logits)

    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = jax.nn.softmax(sorted_logits)
        cum_probs = jnp.cumsum(probs)

        cum_probs_shifted = jnp.roll(cum_probs, 1).at[0].set(0.0)
        cutoff_mask = cum_probs_shifted < top_p

        p_threshold = jnp.min(jnp.where(cutoff_mask, sorted_logits, jnp.inf))
        logits = jnp.where(logits < p_threshold, -jnp.inf, logits)

    return logits

class NonFiniteLogits(RuntimeError):
    """The model produced logits that cannot be sampled from.

    Raised instead of returning a token, because every sampler in this file
    degrades *silently* on non-finite input: `jnp.argmax` over an all-NaN row
    returns index 0, and `jax.random.categorical` returns some index as well.
    Both look exactly like a real prediction, so a checkpoint that overflows
    emits plausible token ids forever and nothing downstream can tell (#229).
    """


def reject_unsampleable(logits, *, where):
    """Raise `NonFiniteLogits` unless `logits` is something we can sample from.

    `jnp.isfinite` is the wrong test here: top-k and top-p set rejected entries
    to -inf on purpose, so -inf is the normal state of nearly the whole row.
    What is never legitimate:

    - **NaN** — an overflow upstream. It passes through truncation untouched,
      because every comparison against NaN is False, so it is still NaN by the
      time a sampler sees it.
    - **+inf** — an f16 overflow in the encoder or the head, which softmax then
      turns into NaN anyway.
    - **every entry -inf** — truncation ate the whole row, leaving no
      distribution at all. Not the #229 failure, but the same silent kind.

    Costs one host transfer, batched into a single array, on a path that already
    synchronizes once per token to read the sampled id.
    """
    n_nan, n_posinf, n_neginf = jnp.stack([
        jnp.isnan(logits).sum(),
        jnp.isposinf(logits).sum(),
        jnp.isneginf(logits).sum(),
    ]).tolist()
    size = int(logits.size)

    if n_nan or n_posinf:
        raise NonFiniteLogits(
            f"{where}: {n_nan} NaN and {n_posinf} +inf among {size} logits. "
            f"The forward pass overflowed. Under COMPUTE_DTYPE=float16 this has "
            f"been checkpoint- and corpus-specific rather than a broken weight "
            f"(#229) — re-run the same input with FORCE_F32_COMPUTE=1 to confirm "
            f"it is the dtype and not the checkpoint."
        )
    if n_neginf == size:
        raise NonFiniteLogits(
            f"{where}: all {size} logits are -inf, so truncation left nothing to "
            f"sample from. Check top_k and top_p."
        )


# `refresh` is deliberately NOT static (#207). It flips every REASONER_REFRESH_EVERY
# tokens, so as a jit cache key it built two executables for one computation — two
# resident programs and two sets of CUDA graphs, which is the driver-memory pressure
# trm/config.py already blames for squeezing batch-2 from outside the BFC arena.
#
# Traced is correct for every architecture, not only the plain one. Plain and the
# refiner ignore the flag outright (they carry no state between windows, so
# each window is a standalone causal prediction), and the reasoner reads it only
# through jax.lax.cond — at reasoner.py:207 and end_step — which takes a traced
# predicate natively. Nothing in the tree branches on it in Python.
#
# The other three must stay static: top_k and top_p gate Python-level branches in
# _temperature_truncate (and lax.top_k needs a static k), and depth is the trip
# count of the refinement loop.
@partial(nnx.jit, static_argnames=['top_k', 'top_p', 'depth'])
def get_logits_for_token(model, padded_tks, token_idx, refresh, top_k, top_p, temperature, depth):
    # Ask for the one row we are about to sample from, rather than projecting the
    # tied head over all MAX_SEQ_LEN positions and slicing (#206). token_idx is
    # traced, so the old `all_logits[0, token_idx, :]` was a dynamic slice that XLA
    # could not dead-code — every emitted token paid ~24.7 GMAC and a ~103MB f32
    # transient for 511 rows nobody reads. The model returns [b, 1, vocab] now.
    row = run_model_inference(model, padded_tks, depth=depth, new_document=refresh,
                              logits_at=token_idx)
    logits = row[0, 0, :]
    return _temperature_truncate(logits, temperature, top_k, top_p)

def generate_text(model, enc, prompt, max_new_tokens=256, temperature=DEFAULT_TEMPERATURE,
                  top_k=50, top_p=0.9, depth=INFERENCE_DEPTH, seed=None, quiet=False):
    # The interactive CLI wants a fresh roll every time, so the wall clock stays the
    # default. An explicit seed makes the same checkpoint reproduce the same text,
    # which is what the transcript logbook needs to compare two checkpoints at all
    # (#203) — and, held constant across depths, is what makes depth the only
    # variable moving between two completions.
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    tokens_list = enc.encode(prompt)
    valid_len = len(tokens_list)

    if valid_len >= MAX_SEQ_LEN:
        tokens_list = tokens_list[:MAX_SEQ_LEN]
        valid_len = MAX_SEQ_LEN

    padded_array = tokens_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - valid_len)
    # Initialize tensor ONCE
    input_ids = jnp.array([padded_array], dtype=jnp.int32)

    if not quiet:
        print("🤖 Assistant: ", end="", flush=True)

    for i in range(max_new_tokens):
        if valid_len >= MAX_SEQ_LEN:
            break

        new_document = (i % REASONER_REFRESH_EVERY == 0)

        # temperature=0 means greedy argmax below; pass 1.0 so the jitted
        # truncation step is a no-op scale rather than a division by zero.
        effective_temperature = temperature if temperature > 0.0 else 1.0
        logits = get_logits_for_token(
            model, input_ids, valid_len - 1,
            refresh=new_document, top_k=top_k, top_p=top_p,
            temperature=effective_temperature, depth=depth,
        )
        # Before sampling, not after: both samplers below turn an unsampleable
        # row into an ordinary-looking token id (#229).
        reject_unsampleable(logits, where=f"token {i}, position {valid_len - 1}, depth {depth}")

        rng, subkey = jax.random.split(rng)

        if temperature > 0.0:
            next_token = int(jax.random.categorical(subkey, logits))
        else:
            next_token = int(jnp.argmax(logits))

        # The end of a document (#373): with EOT a real token the model can predict it,
        # and a pad was never a target, so either one ends the generation.
        if next_token in (EOT_TOKEN_ID, PAD_TOKEN_ID):
            break

        tokens_list.append(next_token)
        if not quiet:
            print(enc.decode([next_token]), end="", flush=True)

        input_ids = input_ids.at[0, valid_len].set(next_token)
        valid_len += 1

    if not quiet:
        print()
    return tokens_list

def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Interactive generation from the latest checkpoint.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="softmax temperature; 0 means greedy argmax "
                         f"(default {DEFAULT_TEMPERATURE})")
    ap.add_argument("--top-k", type=int, default=50,
                    help="keep only the k highest-scoring tokens; 0 disables (default 50)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="nucleus cutoff, applied after temperature (default 0.9)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generation length cap (default 256)")
    ap.add_argument("--depth", type=int, default=INFERENCE_DEPTH,
                    help="refinement loops per forward pass, for the depth-recurrent "
                         "arches (refiner, reasoner); the plain model ignores it. The "
                         f"dense sweep put the refiner's plateau at ~6 (default "
                         f"{INFERENCE_DEPTH}). The refiner's sinusoidal time signal is "
                         "defined at any step, so it extrapolates past the trained range; "
                         "the reasoner's learned time table stops at MAX_STEPS_LIMIT")
    return ap


def run_inference(argv=None):
    args = build_arg_parser().parse_args(argv)

    print(f"🔮 Initializing '{MODEL_ARCH}' (Dim={LATENT_DIM}, serving depth {args.depth})...")
    print(f"   sampling: temperature {args.temperature}, top_k {args.top_k}, "
          f"top_p {args.top_p}, max {args.max_new_tokens} tokens")

    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    model = build_serving_model()

    # CHECKPOINT_ROOT names a checkpoint dir; without it, the latest checkpointed run
    # under runs/ is served. There is no third, default path: the old one
    # (`orbax_checkpoints`) pointed at a directory nothing writes.
    if os.environ.get("CHECKPOINT_ROOT") is not None:
        active_checkpoint_dir = resolve_root(os.environ["CHECKPOINT_ROOT"])
    else:
        from trm.runtime.checkpoints import discover_latest_checkpoint_run
        discovered_path, discovered_run_id = discover_latest_checkpoint_run()
        if discovered_path is None:
            print("❌ Error: no checkpointed run under runs/, and CHECKPOINT_ROOT is not set.")
            print("Train one with `python -m trm.train.start`, or set CHECKPOINT_ROOT to a checkpoint dir.")
            return
        active_checkpoint_dir = os.path.abspath(discovered_path)
        print(f"🔎 Auto-discovered latest checkpointed run for inference: {discovered_run_id}")

    mngr = ocp.CheckpointManager(
        active_checkpoint_dir,
        item_names=CHECKPOINT_ITEMS,
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        print(f"❌ Error: No available weights here (no checkpoints found in {active_checkpoint_dir}).")
        print("Please train the model first using: python -m trm.train.start")
        return

    print(f"🔄 Loading weights from step {latest_step}...")

    # Tolerate buffers a past version of this model saved and the current one no
    # longer defines (#105's vestigial hunch buffer, which every base-run
    # checkpoint before it still carries). Same helper the trainer's resume uses,
    # so serving and resuming accept exactly the same set of checkpoints.
    from trm.runtime.checkpoints import restore_tolerating_legacy
    restored = restore_tolerating_legacy(
        lambda model_target: mngr.restore(latest_step, args=ocp.args.Composite(
            model=ocp.args.StandardRestore(model_target),
        )),
        model,
    )
    nnx.update(model, restored['model'])

    print("✅ Model loaded and ready!")

    print("\n" + "="*50)
    print("TinyRefinementModel CLI (Orbax-Linked)")
    print("Type your prompt and press Enter (/exit to quit)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("👤 User: ").strip()

            if user_input.lower() == "/exit":
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            generate_text(model, enc, user_input,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p, depth=args.depth)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()