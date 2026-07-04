"""The one-command GPT-2-small yardstick run (#48).

Point it at a checkpoint and it answers the base-model-bar question in one
glance: LAMBADA last-word accuracy + LAMBADA perplexity next to the GPT-2-small
reference, plus held-out perplexity on our own corpus — and a JSON row shaped
for the model card (docs/registry/MODEL_CARD_TEMPLATE.md).

    DATA_ROOT=runs/data PYTHONPATH=. python tools/eval_yardstick.py \
        [--arch refiner] [--checkpoint-path runs/run_x/checkpoints] \
        [--depth 4] [--limit 500] [--json-out path.json]

Reading the result honestly (the caveat lives in issue #48): our corpus is
fineweb-edu / code / math, not general web/narrative like WebText — LAMBADA is
narrative last-word prediction, so it may run harder for our model than for
GPT-2 at equal capability. The held-out ppl on our own distribution is printed
alongside so the two readings can be separated: missing the LAMBADA bar with a
healthy own-distribution ppl says "distribution gap", missing both says "the
model is undertrained (or the pipeline is broken)".

On CPU prepend FORCE_F32_COMPUTE=1 (CPU XLA cannot lower the f16 matmuls); on a
GPU shared with a training run the default 0.5 mem fraction leaves room.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import json
import subprocess

import jax.numpy as jnp
import numpy as np
import tiktoken
from flax import nnx
from dotenv import load_dotenv

from config import BATCH_SIZE, MAX_SEQ_LEN, PAD_TOKEN_ID, TOKENIZER_NAME, MODEL_ARCH
from tools.common import restore_model, restore_refiner
from tools.yardstick import (
    GPT2_SMALL_REFERENCE,
    LAMBADA_SHA256,
    encode_example,
    fetch_lambada,
    load_examples,
    score_examples,
    summarize,
)

# .env supplies DATA_ROOT (read at runtime by the held-out probe); config's own
# env knobs are process-level and must be set in the shell, as everywhere else.
load_dotenv()

# Matches the validation probe's fixed depth (validation.py), so the yardstick
# and the training-time val curve read the model at the same setting. Sweep
# --depth explicitly when the question is depth-dependent.
DEFAULT_DEPTH = 4


@nnx.jit(static_argnames=["depth"])
def _forward_logits(model, tokens, depth):
    return model(tokens, max_steps=depth, training=False, should_refresh=True).logits


def make_logits_fn(model, depth):
    """Adapt a restored model to the yardstick's numpy-causal interface.

    Every batch starts from fresh cross-window state (the hunch cache is
    vestigial-zero for the refiner and refreshed by should_refresh=True for the
    reasoner), so LAMBADA examples stay independent."""
    def logits_fn(tokens):
        return np.asarray(_forward_logits(model, jnp.asarray(tokens), depth=depth))
    return logits_fn


def heldout_perplexity(model):
    """exp(held-out CE) on our own distribution, via the same ValidationProbe the
    trainer reports — one number, directly comparable to the training val curve.
    Returns None (with a note) when the tokenized corpus isn't reachable."""
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        print("⚠️ Held-out ppl skipped: DATA_ROOT is not set (try DATA_ROOT=runs/data).")
        return None
    from config import resolve_root
    from validation import ValidationProbe

    ce = ValidationProbe(resolve_root(data_root)).run(model)
    if ce is None:
        return None
    return {"val_ce": ce, "ppl": float(np.exp(ce))}


def main():
    ap = argparse.ArgumentParser(description="GPT-2-small yardstick: LAMBADA acc/ppl + held-out ppl")
    ap.add_argument("--checkpoint-path", default=None, help="Orbax dir (default: latest run)")
    ap.add_argument("--arch", default=MODEL_ARCH, choices=["reasoner", "refiner"],
                    help="which param tree the checkpoint holds (default: MODEL_ARCH)")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                    help=f"refinement/reasoning depth at eval (default {DEFAULT_DEPTH}, as validation.py)")
    ap.add_argument("--batch", type=int, default=4, help="examples per forward")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N examples (smoke); the bar needs the full set")
    ap.add_argument("--data-path", default=None, help="local lambada_test.jsonl (default: fetch+cache)")
    ap.add_argument("--json-out", default=None,
                    help="where to write the model-card row (default runs/yardstick/<step>.json)")
    ap.add_argument("--no-heldout", action="store_true", help="skip the own-corpus ppl probe")
    args = ap.parse_args()

    if args.arch == "reasoner" and args.batch != BATCH_SIZE:
        # The reasoner's slot/hunch caches are built (and checkpointed) at
        # BATCH_SIZE; its forward asserts on any other leading dim.
        print(f"⚠️ reasoner arch: clamping --batch {args.batch} -> {BATCH_SIZE}.")
        args.batch = BATCH_SIZE
    restore = {"reasoner": restore_model, "refiner": restore_refiner}[args.arch]
    model, step = restore(args.checkpoint_path)

    path = args.data_path or fetch_lambada()
    texts = load_examples(path)
    if args.limit:
        texts = texts[:args.limit]
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    encoded, skipped = [], 0
    for text in texts:
        pair = encode_example(enc, text, MAX_SEQ_LEN)
        if pair is None:
            skipped += 1
        else:
            encoded.append(pair)
    if skipped:
        print(f"⚠️ Skipped {skipped} degenerate examples (no context/target after encoding).")

    print(f"📏 LAMBADA: {len(encoded)} examples | arch {args.arch} | depth {args.depth} | batch {args.batch}")
    scores = score_examples(
        make_logits_fn(model, args.depth), encoded, PAD_TOKEN_ID, batch_size=args.batch,
        progress=lambda done, total: print(f"  … {done}/{total}", flush=True) if done % 512 < args.batch else None,
    )
    result = summarize(scores)
    heldout = None if args.no_heldout else heldout_perplexity(model)

    ref_acc, ref_ppl = GPT2_SMALL_REFERENCE["lambada_acc"], GPT2_SMALL_REFERENCE["lambada_ppl"]
    print()
    print(f"{'metric':<28} {'ours':>10} {'GPT-2-small':>12}   verdict")
    print(f"{'LAMBADA last-word acc':<28} {result['lambada_acc']:>10.4f} {ref_acc:>12.4f}   "
          f"{'meets the bar ✅' if result['lambada_acc'] >= ref_acc else 'below the bar'}")
    print(f"{'LAMBADA ppl':<28} {result['lambada_ppl']:>10.2f} {ref_ppl:>12.2f}   "
          f"{'meets the bar ✅' if result['lambada_ppl'] <= ref_ppl else 'below the bar'}")
    if heldout:
        print(f"{'held-out ppl (our corpus)':<28} {heldout['ppl']:>10.2f} {'—':>12}   "
              f"(val CE {heldout['val_ce']:.4f}; internal track, no external reference)")
    if args.limit:
        print(f"⚠️ --limit {args.limit}: a smoke reading, not the bar.")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {
        "commit": commit,
        "arch": args.arch,
        "checkpoint": {"path": args.checkpoint_path or "latest", "step": int(step)},
        "eval_depth": args.depth,
        "tokenizer": TOKENIZER_NAME,
        "lambada": {**result, "data_sha256": LAMBADA_SHA256, "limit": args.limit},
        "heldout": heldout,
        "gpt2_small_reference": GPT2_SMALL_REFERENCE,
    }
    out = args.json_out or f"runs/yardstick/step{step}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(f"🧾 Model-card row -> {out}")


if __name__ == "__main__":
    main()
