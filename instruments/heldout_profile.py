"""What the mean held-out CE averages away, per checkpoint (#460).

The mean is the right measure of learning, but it is dominated by tokens any model
gets right, it has the same shape at every scale, and it cannot be compared across
tokenizers. This scores a checkpoint on the trainer's own held-out rows (the same
slice and #373 target mask `trm.train.validation` uses) and reports four things:

  1. bits per byte — summed CE in bits over the UTF-8 bytes of the targets; the one
     number comparable across tokenizers;
  2. CE by context position, in log-spaced buckets of context length — whether the
     model still gains from context late in the window;
  3. the hard tail — mean CE of the worst 10% and 1% of targets, and the per-token
     loss quantiles, where models differ most;
  4. calibration — expected calibration error of the top-1 probability.

Everything comes from the weights, so any saved checkpoint can be scored after the
fact without touching the run that wrote it. One JSON row per call is appended to
the run's `heldout_profile.jsonl`, keyed by step, so a run's checkpoints make a curve.

    FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu DATA_ROOT=runs/data PYTHONPATH=. \\
        python -m instruments.heldout_profile \\
        --checkpoint-path runs/<run>/checkpoints/milestones --step <n>
"""

from __future__ import annotations

import os

# CPU and f32 by default, before anything imports jax or trm.config: this runs beside
# a training run, and on the card it would take memory the trainer's arena counts on.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("FORCE_F32_COMPUTE", "1")

import argparse
import datetime
import json
import math
import pathlib

import numpy as np

from instruments._common import add_checkpoint_argument, git_head, load_env
from instruments.arch import add_arch_argument
from instruments.results import emit

TAILS = (0.10, 0.01)
QUANTILES = (0.5, 0.9, 0.99)
ECE_BINS = 15
# The session running this is the OOM victim beside a trainer (the kernel scores it
# higher), so it refuses to start rather than squeeze one. A 138M model in f32 plus
# one window's logits and their softmax needs ~2 GiB; the rest is margin.
MIN_AVAILABLE_GIB = 3.0

REPORTS = {
    "CE, bpb, CE by position, tail, ECE": ("sampled", "every target of the trainer's fixed held-out rows "
                                           "(EVAL_ROWS per source), scored in f32 on CPU"),
}


def position_bucket(context_len):
    """Bucket k holds context lengths [2^k, 2^(k+1)): 1, 2-3, 4-7, ..."""
    return np.floor(np.log2(np.asarray(context_len))).astype(int)


def profile(nll, top_prob, correct, context_len, nbytes, window=None):
    """The four readings, from per-target arrays: loss in nats, top-1 probability,
    whether top-1 was the target, context length it was predicted from, and the
    UTF-8 byte count of the target token. Pure numpy, so it is tested exactly.

    Bucket labels are the lengths a bucket can hold in a `window`-long window (the
    longest context seen, by default), so every row of one run keys the same way."""
    nll, top_prob = np.asarray(nll, np.float64), np.asarray(top_prob, np.float64)
    correct, nbytes = np.asarray(correct, bool), np.asarray(nbytes, np.int64)
    out = {"targets": int(nll.size), "ce": float(nll.mean()),
           "bpb": float(nll.sum() / math.log(2) / nbytes.sum())}

    context_len = np.asarray(context_len)
    window = int(context_len.max()) if window is None else window
    buckets = position_bucket(context_len)
    out["ce_by_position"] = {}
    for k in np.unique(buckets):
        lo, hi = 2 ** int(k), min(2 ** (int(k) + 1) - 1, window)
        out["ce_by_position"][str(lo) if lo == hi else f"{lo}-{hi}"] = float(nll[buckets == k].mean())

    worst = np.sort(nll)[::-1]
    for share in TAILS:
        out[f"worst_{share:.0%}_ce"] = float(worst[:max(1, round(share * worst.size))].mean())
    for q in QUANTILES:
        out[f"p{round(q * 100)}_ce"] = float(np.quantile(nll, q))

    bins = np.minimum((top_prob * ECE_BINS).astype(int), ECE_BINS - 1)
    out["ece"] = float(sum(abs(correct[bins == b].mean() - top_prob[bins == b].mean()) * (bins == b).mean()
                           for b in np.unique(bins)))
    out["top1_accuracy"] = float(correct.mean())
    return out


def token_bytes(vocab_size):
    """UTF-8 byte length of every token id; 0 for ids past the tokenizer (padding)."""
    import tiktoken

    from trm.config import TOKENIZER_NAME
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    return np.array([len(enc.decode_single_token_bytes(t)) if t < enc.n_vocab else 0
                     for t in range(vocab_size)], dtype=np.int64)


def score_rows(model, rows):
    """Per-target arrays over held-out rows. Each row is two windows, window 1 opening
    a document and window 2 continuing it, the way `trm.train.validation` scores them;
    a target's context length is its position in its window plus one."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from trm.config import MAX_SEQ_LEN, PAD_TOKEN_ID
    from trm.train.validation import VAL_FIXED_DEPTH, heldout_targets

    @nnx.jit(static_argnames=["new_document"])
    def window(model, tokens, targets, new_document):
        logits = model(tokens, depth=VAL_FIXED_DEPTH, training=False, new_document=new_document).logits
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return nll, jnp.exp(logp.max(-1)), logp.argmax(-1) == targets

    parts = {k: [] for k in ("nll", "top_prob", "correct", "context_len", "target")}
    with model.isolated_state():
        for row in rows:
            model.reset_state()
            row = jnp.asarray(row)
            for w, new_document in ((0, True), (1, False)):
                tokens = row[:, w * MAX_SEQ_LEN:(w + 1) * MAX_SEQ_LEN]
                targets = heldout_targets(row[:, w * MAX_SEQ_LEN + 1:(w + 1) * MAX_SEQ_LEN + 1], PAD_TOKEN_ID)
                nll, top_prob, correct = window(model, tokens, targets, new_document)
                keep = np.asarray(targets != PAD_TOKEN_ID)
                for key, value in (("nll", nll), ("top_prob", top_prob), ("correct", correct),
                                   ("context_len", np.broadcast_to(np.arange(1, tokens.shape[1] + 1), keep.shape)),
                                   ("target", targets)):
                    parts[key].append(np.asarray(value)[keep])
    return {k: np.concatenate(v) for k, v in parts.items()}


def memory_available_gib():
    """MemAvailable from /proc/meminfo, or None where there is no such file."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        return None
    return None


def run_dir(checkpoint_path):
    """The run a checkpoint manager dir belongs to: runs/<run>/checkpoints, or one of
    its subdirs (milestones, best_val_ce)."""
    from trm.runtime.layout import BEST_SUBDIR, MILESTONE_SUBDIR
    path = pathlib.Path(os.path.abspath(checkpoint_path))
    return (path.parent if path.name in (MILESTONE_SUBDIR, BEST_SUBDIR) else path).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_checkpoint_argument(ap, required=True, aliases=("--ckpt",))
    ap.add_argument("--step", type=int, default=None, help="step to score (default: the dir's newest)")
    add_arch_argument(ap)
    ap.add_argument("--rows", type=int, default=None, help="held-out rows per source (default: the trainer's)")
    ap.add_argument("--sources", default="fineweb-edu,codeparrot,finemath",
                    help="fineweb-edu reads the trainer's fixed skip; the others their tails (#363)")
    ap.add_argument("--jsonl", type=pathlib.Path, default=None,
                    help="where to append the row (default: <run>/heldout_profile.jsonl)")
    args = ap.parse_args(argv)

    available = memory_available_gib()
    if available is not None and available < MIN_AVAILABLE_GIB:
        raise SystemExit(f"only {available:.1f} GiB available (< {MIN_AVAILABLE_GIB}); not starting beside a trainer")

    load_env()
    from trm.config import MAX_SEQ_LEN, VOCAB_SIZE, resolve_root
    from trm.runtime.restore import restore_arch
    from trm.train.validation import VAL_ROWS, VAL_SKIP_SAMPLES, ValidationProbe

    model, step = restore_arch(args.arch, args.checkpoint_path, step=args.step)
    nbytes = token_bytes(VOCAB_SIZE)
    data_root = resolve_root(os.environ.get("DATA_ROOT", "runs/data"))
    row = {"step": int(step), "arch": args.arch, "checkpoint": str(args.checkpoint_path),
           "commit": git_head(short=False), "when": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "sources": {}}
    for source in args.sources.split(","):
        probe = ValidationProbe(data_root, rows=args.rows or VAL_ROWS,
                                skip=VAL_SKIP_SAMPLES if source == "fineweb-edu" else None, source=source)
        rows = probe.load_rows()
        if not rows:
            continue
        scored = score_rows(model, rows)
        result = profile(scored["nll"], scored["top_prob"], scored["correct"],
                         scored["context_len"], nbytes[scored["target"]], window=MAX_SEQ_LEN)
        row["sources"][source] = result
        print(f"\n{source}: {result['targets']} targets  CE {result['ce']:.4f}  bpb {result['bpb']:.4f}  "
              f"top-1 {result['top1_accuracy']:.3f}  ECE {result['ece']:.4f}")
        print("  CE by context length: " + "  ".join(f"{k}:{v:.3f}" for k, v in result["ce_by_position"].items()))
        print(f"  tail: worst 10% {result['worst_10%_ce']:.3f}  worst 1% {result['worst_1%_ce']:.3f}  "
              f"p50 {result['p50_ce']:.3f}  p90 {result['p90_ce']:.3f}  p99 {result['p99_ce']:.3f}")
        emit(f"{source}@{step}", **{k: v for k, v in result.items() if not isinstance(v, dict)},
             **{f"ce_ctx_{k}": v for k, v in result["ce_by_position"].items()})

    out = args.jsonl or run_dir(args.checkpoint_path) / "heldout_profile.jsonl"
    with open(out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"\n🧾 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
