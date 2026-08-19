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
    PAD_TOKEN_ID,
    TOKENIZER_NAME,
    resolve_root,
)
from trm.model.contract import LanguageModel

from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_DIR = resolve_root(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))
HUNCH_REFRESH_EVERY = 4

# Sampling default. 0.5 was too cold to see the model: it divides the logits, so it
# *doubles* every gap, and with max|logit| already ~21 partway through the base run
# that makes sampling effectively greedy — which is the classic way to turn an
# undertrained LM into a repetition loop ("the ratio of the ratio of the ratio").
# What you read then is the decoder's failure mode, not the weights'.
DEFAULT_TEMPERATURE = 0.7


def build_model(arch=MODEL_ARCH):
    """The architecture MODEL_ARCH selects — the same choice the trainer makes.

    This used to construct UniversalReasoner unconditionally, while MODEL_ARCH has
    defaulted to 'refiner' since Plan A became the live bet. The two have different
    param trees, so serving a refiner checkpoint failed on a structure mismatch and
    inference was simply unavailable for the architecture we actually train — the
    same defect the plotter carried (#181), from the same cause: a tool naming one
    architecture while the run selects another.
    """
    if arch == "refiner":
        # Imported lazily so the baseline path never touches Plan A code, matching
        # trm/train/trainer.py's init.
        from trm.model.refiner_lm import RefinerForTraining
        return RefinerForTraining(LATENT_DIM, nnx.Rngs(0))
    from trm.model.reasoner import UniversalReasoner
    return UniversalReasoner(LATENT_DIM, nnx.Rngs(0))

def run_model_inference(
    model: LanguageModel,
    tokens: jnp.ndarray,
    depth: int = INFERENCE_DEPTH,
    new_document: bool = True,
) -> jnp.ndarray:
    out = model(
        tokens, depth=depth, training=False, new_document=new_document
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

@partial(nnx.jit, static_argnames=['refresh', 'top_k', 'top_p', 'depth'])
def get_logits_for_token(model, padded_tks, token_idx, refresh, top_k, top_p, temperature, depth):
    all_logits = run_model_inference(model, padded_tks, depth=depth, new_document=refresh)
    logits = all_logits[0, token_idx, :]
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

        new_document = (i % HUNCH_REFRESH_EVERY == 0)

        # temperature=0 means greedy argmax below; pass 1.0 so the jitted
        # truncation step is a no-op scale rather than a division by zero.
        effective_temperature = temperature if temperature > 0.0 else 1.0
        logits = get_logits_for_token(
            model, input_ids, valid_len - 1,
            refresh=new_document, top_k=top_k, top_p=top_p,
            temperature=effective_temperature, depth=depth,
        )

        rng, subkey = jax.random.split(rng)

        if temperature > 0.0:
            next_token = int(jax.random.categorical(subkey, logits))
        else:
            next_token = int(jnp.argmax(logits))

        if next_token == PAD_TOKEN_ID:
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
                    help="refinement loops per forward pass. The dense sweep put the "
                         f"accuracy plateau at ~6 (default {INFERENCE_DEPTH}); the "
                         "sinusoidal time signal is defined at any step, so this "
                         "extrapolates past the trained range")
    return ap


def run_inference(argv=None):
    args = build_arg_parser().parse_args(argv)

    print(f"🔮 Initializing '{MODEL_ARCH}' (Dim={LATENT_DIM}, serving depth {args.depth})...")
    print(f"   sampling: temperature {args.temperature}, top_k {args.top_k}, "
          f"top_p {args.top_p}, max {args.max_new_tokens} tokens")

    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    model = build_model()

    active_checkpoint_dir = CHECKPOINT_DIR
    if os.environ.get("CHECKPOINT_ROOT") is None:
        from trm.runtime.checkpoints import discover_latest_checkpoint_run
        discovered_path, discovered_run_id = discover_latest_checkpoint_run()
        if discovered_path is not None:
            active_checkpoint_dir = os.path.abspath(discovered_path)
            print(f"🔎 Auto-discovered latest checkpointed run for inference: {discovered_run_id}")
        else:
            print("❌ Error: No available weights here.")
            print("Please train the model first using: python -m trm.train.start")
            return

    mngr = ocp.CheckpointManager(
        active_checkpoint_dir,
        item_names=('model', 'optimizer', 'monitor_state', 'step'),
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