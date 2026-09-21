"""The one-command GPT-2-small yardstick run (#48).

Point it at a checkpoint and it answers the base-model-bar question in one
glance: LAMBADA last-word accuracy + LAMBADA perplexity next to the GPT-2-small
reference, plus held-out perplexity on our own corpus — and a JSON row shaped
for the model card (docs/registry/MODEL_CARD_TEMPLATE.md).

    DATA_ROOT=runs/data PYTHONPATH=. python -m instruments.yardstick.eval_yardstick \
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

import jax.numpy as jnp
import numpy as np
import tiktoken
from flax import nnx

from instruments._common import add_checkpoint_argument, git_head, load_env

from trm.config import MAX_SEQ_LEN, PAD_TOKEN_ID, TOKENIZER_NAME
from trm.runtime.restore import EVAL_BATCH_SIZE, restore_arch
from instruments.arch import add_arch_argument
from instruments.yardstick.yardstick import (
    GPT2_SMALL_REFERENCE,
    LAMBADA_SHA256,
    encode_example,
    fetch_lambada,
    load_examples,
    score_examples,
    summarize,
)
from instruments.yardstick import fineweb_val

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "LAMBADA acc, ppl": ("measured", "the full LAMBADA test set; with --limit it is a subsample and not the bar"),
    "held-out ppl": ("sampled", "a fixed slice of held-out rows from our corpus"),
    "FineWeb val CE": ("sampled", "the first --fineweb-tokens of the speedrun's FineWeb val shard at our window; "
                                  "the reference reads 10,485,760 of them at 1,024"),
}

# Off-config defaults, on purpose (tests/apparatus/test_instrument_defaults.py).
CONFIG_DIVERGENCES = {"--batch": "examples per eval forward, not the training micro-batch"}

# 2^20 targets by default: the full 10.5M shard is ~10x the CPU time of LAMBADA at a
# milestone, and 1M targets already reads the mean to about a hundredth of a nat.
# The model card's number takes --fineweb-tokens 10485760.
FINEWEB_DEFAULT_TOKENS = 1 << 20

# Where this differs from production's environment, and why (#166).
ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "an eval that may share the card with a training run"}

# .env supplies DATA_ROOT (read at runtime by the held-out probe); config's own
# env knobs are process-level and must be set in the shell, as everywhere else.
load_env()

# Matches the validation probe's fixed depth (validation.py), so the yardstick
# and the training-time val curve read the model at the same setting. Sweep
# --depth explicitly when the question is depth-dependent. Only the retired
# refiner/reasoner read it: `plain` has no depth dial and ignores the argument.
DEFAULT_DEPTH = 4


@nnx.jit(static_argnames=["depth"])
def _forward_logits(model, tokens, depth):
    return model(tokens, depth=depth, training=False, new_document=True).logits


def make_logits_fn(model, depth):
    """Adapt a restored model to the yardstick's numpy-causal interface.

    Every batch is scored as a fresh document (new_document=True), so a model
    that carries cross-window state starts clean and LAMBADA examples stay
    independent. A stateless model carries nothing to begin with."""
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
    from trm.config import resolve_root
    from trm.train.validation import ValidationProbe

    ce = ValidationProbe(resolve_root(data_root)).run(model)
    if ce is None:
        return None
    return {"val_ce": ce, "ppl": float(np.exp(ce))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="GPT-2-small yardstick: LAMBADA acc/ppl + held-out ppl")
    add_checkpoint_argument(ap)
    ap.add_argument("--step", type=int, default=None,
                    help="which step of that dir to score (default: its newest). A milestones dir "
                         "holds many, and the one to score is the milestone that was asked for")
    # The param tree the checkpoint holds. A run records its own in run_metadata.json
    # and instruments.base_run passes that; MODEL_ARCH is only the fallback.
    add_arch_argument(ap)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                    help=f"refinement/reasoning depth at eval (default {DEFAULT_DEPTH}, as validation.py)")
    ap.add_argument("--batch", type=int, default=4, help="examples per forward")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N examples (smoke); the bar needs the full set")
    ap.add_argument("--data-path", default=None, help="local lambada_test.jsonl (default: fetch+cache)")
    ap.add_argument("--json-out", default=None,
                    help="where to write the model-card row (default runs/yardstick/<step>.json)")
    ap.add_argument("--no-heldout", action="store_true", help="skip the own-corpus ppl probe")
    ap.add_argument("--fineweb-tokens", type=int, default=FINEWEB_DEFAULT_TOKENS,
                    help=f"tokens of the speedrun's FineWeb val shard to score (0 skips; the full reference "
                         f"set is {fineweb_val.SPEEDRUN_VAL_TOKENS})")
    ap.add_argument("--fineweb-window", type=int, default=MAX_SEQ_LEN,
                    help="context window the FineWeb shard is cut into (default: the trained MAX_SEQ_LEN)")
    args = ap.parse_args(argv)

    if args.arch == "reasoner" and args.batch != EVAL_BATCH_SIZE:
        # The reasoner's slot/hunch caches are built (and checkpointed) at the
        # eval batch size; its forward asserts on any other leading dim. This is
        # EVAL_BATCH_SIZE, not the training BATCH_SIZE (#24) — the yardstick must
        # keep restoring checkpoints written before batching changed.
        print(f"⚠️ reasoner arch: clamping --batch {args.batch} -> {EVAL_BATCH_SIZE}.")
        args.batch = EVAL_BATCH_SIZE
    model, step = restore_arch(args.arch, args.checkpoint_path, step=args.step)

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
    fineweb = None
    if args.fineweb_tokens:
        tokens = fineweb_val.read_tokens(fineweb_val.fetch_fineweb_val(), args.fineweb_tokens + 1)
        print(f"📏 FineWeb val: {args.fineweb_tokens} targets | window {args.fineweb_window}")
        fineweb = fineweb_val.score(make_logits_fn(model, args.depth), tokens, args.fineweb_window, batch=args.batch)
        fineweb["sha256"] = fineweb_val.FINEWEB_VAL_SHA256

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
    if fineweb:
        print(f"{'FineWeb val CE':<28} {fineweb['val_ce']:>10.4f} {fineweb_val.REFERENCE['val_ce']:>12.2f}   "
              f"(speedrun target; ours at a {fineweb['window']}-token window, theirs 1,024)")
    if args.limit:
        print(f"⚠️ --limit {args.limit}: a smoke reading, not the bar.")

    row = {
        "commit": git_head(short=False),
        "arch": args.arch,
        "checkpoint": {"path": args.checkpoint_path or "latest", "step": int(step)},
        "eval_depth": args.depth,
        "tokenizer": TOKENIZER_NAME,
        "lambada": {**result, "data_sha256": LAMBADA_SHA256, "limit": args.limit},
        "heldout": heldout,
        "fineweb_val": fineweb and {**fineweb, "reference": fineweb_val.REFERENCE},
        "gpt2_small_reference": GPT2_SMALL_REFERENCE,
    }
    out = args.json_out or f"runs/yardstick/step{step}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(f"🧾 Model-card row -> {out}")


if __name__ == "__main__":
    main()
