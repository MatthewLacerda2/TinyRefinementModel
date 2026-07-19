"""Depth playground: one prompt, one checkpoint, the refinement dial swept.

Generates the same completion at several inference depths (default 1,2,4,8,16)
so the depth-recurrence bet is inspectable by eye: does more "thinking" per
token change what the model says? The sinusoidal time signal (#86) is defined
at every step, so depths past the trained max (8) are legal extrapolation.

The sampling RNG is seeded identically for every depth — the depth is the only
variable; completions diverge only where the logits themselves diverge. Use
--temperature 0 for fully deterministic (greedy) comparisons.

Usage:
    PYTHONPATH=. python tools/depth_playground.py --prompt "The meaning of life is"
    PYTHONPATH=. python tools/depth_playground.py            # interactive loop
    ... [--depths 1,2,4,8,16] [--max-new-tokens 128] [--temperature 0.5]
        [--seed 42] [--checkpoint-path runs/run_X/checkpoints]

Needs the GPU free (a training run owns the card; run this after it stops).
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import tiktoken
from dotenv import load_dotenv

load_dotenv()

from config import MAX_SEQ_LEN, PAD_TOKEN_ID, TOKENIZER_NAME
from infer_local import _temperature_truncate
from tools.common import restore_refiner


def generate_at_depth(model, enc, prompt, depth, *, max_new_tokens, temperature,
                      top_k, top_p, seed):
    """infer_local.generate_text with the depth dial exposed and a fixed seed."""
    rng = jax.random.PRNGKey(seed)

    tokens_list = enc.encode(prompt)[:MAX_SEQ_LEN]
    valid_len = len(tokens_list)
    padded = tokens_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - valid_len)
    input_ids = jnp.array([padded], dtype=jnp.int32)

    t0 = time.time()
    generated = []
    for _ in range(max_new_tokens):
        if valid_len >= MAX_SEQ_LEN:
            break
        out = model(input_ids, max_steps=depth, training=False)
        logits = out.logits[0, valid_len - 1, :]
        logits = _temperature_truncate(
            logits, temperature if temperature > 0.0 else 1.0, top_k, top_p)

        rng, subkey = jax.random.split(rng)
        next_token = (int(jax.random.categorical(subkey, logits))
                      if temperature > 0.0 else int(jnp.argmax(logits)))
        if next_token == PAD_TOKEN_ID:
            break
        generated.append(next_token)
        input_ids = input_ids.at[0, valid_len].set(next_token)
        valid_len += 1

    dt = time.time() - t0
    return enc.decode(generated), (len(generated) / dt if dt > 0 else 0.0)


def sweep(model, enc, prompt, depths, args):
    print(f"\n{'=' * 70}\n👤 {prompt}\n{'=' * 70}")
    for d in depths:
        text, tps = generate_at_depth(
            model, enc, prompt, d,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, seed=args.seed)
        print(f"\n--- depth {d}  ({tps:.1f} tok/s) ---")
        print(text.strip() or "(empty)")
    print()


def main():
    p = argparse.ArgumentParser(description="sweep inference depth on one prompt")
    p.add_argument("--prompt", action="append", default=None,
                   help="prompt to sweep (repeatable); omit for an interactive loop")
    p.add_argument("--depths", type=str, default="1,2,4,8,16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="0 = greedy (deterministic across runs)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42,
                   help="sampling seed, shared across depths so depth is the only variable")
    p.add_argument("--checkpoint-path", type=str, default=None)
    args = p.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    model, step = restore_refiner(args.checkpoint_path)
    print(f"✅ refiner checkpoint at step {step} | depths {depths} | "
          f"temperature {args.temperature} | seed {args.seed}")
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    if args.prompt:
        for prompt in args.prompt:
            sweep(model, enc, prompt, depths, args)
        return

    print("Interactive — empty line to quit.")
    while True:
        try:
            prompt = input("\n👤 prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        sweep(model, enc, prompt, depths, args)


if __name__ == "__main__":
    main()
