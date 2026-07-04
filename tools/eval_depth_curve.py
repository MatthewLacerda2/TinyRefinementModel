"""Does reasoning about the previous window help predict the next one?

Since the 2026-06-11 causality fix, a window's reasoning loop only influences
the NEXT window (through the hunch cache) — so this measures the architecture's
honest claim. For each held-out sample: run window 1 at depth d (priming the
hunch), then score window 2's tokens against that hunch. The fresh-slots
baseline (no hunch at all) is the control. If the curve sits below the baseline
and falls with depth, reasoning works; flat at the baseline means the hunch is
dead weight.

Run offline against the latest (or a given) checkpoint:
    PYTHONPATH=. python tools/eval_depth_curve.py [--batches 16] [--skip 3000000]
"""

import os

# Eval needs no optimizer/gradient memory; a modest arena is plenty and leaves
# room in case a training process is still holding the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from dotenv import load_dotenv

from config import MAX_SEQ_LEN, MAX_STEPS_LIMIT, MODEL_ARCH, PAD_TOKEN_ID

load_dotenv()

from tools.common import restore_model, load_eval_batches


@nnx.jit(static_argnames=["max_steps"])
def prime_hunch(model, window1, max_steps):
    """Run the reasoning loop over window 1; its output lands in the hunch cache."""
    model(window1, max_steps=max_steps, training=False, should_refresh=True)


@nnx.jit(static_argnames=["use_hunch"])
def score_window2(model, tokens, use_hunch):
    """Per-token CE on window 2. Its own loop depth is irrelevant to its logits
    (the decoder reads the slots the window started with), so run depth 1."""
    seq_in = tokens[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN]
    seq_out = tokens[:, MAX_SEQ_LEN + 1:2 * MAX_SEQ_LEN + 1]
    out = model(seq_in, max_steps=1, training=False, should_refresh=not use_hunch)
    mask = seq_out != PAD_TOKEN_ID
    token_ce = optax.softmax_cross_entropy_with_integer_labels(logits=out.logits, labels=seq_out)
    return token_ce, mask


def main():
    parser = argparse.ArgumentParser(description="window-2 CE vs window-1 reasoning depth")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Orbax checkpoint dir (defaults to the latest run's)")
    parser.add_argument("--batches", type=int, default=16, help="number of held-out batches to average over")
    parser.add_argument("--skip", type=int, default=3_000_000, help="samples to skip so eval data is past the trained range")
    parser.add_argument("--source", type=str, default="pretrain/fineweb-edu", help="data subdirectory under DATA_ROOT")
    args = parser.parse_args()

    if MODEL_ARCH == "refiner":
        raise SystemExit(
            "eval_depth_curve probes the reasoner's cross-window hunch; the refiner has none "
            "(its hunch_cache is a vestigial zero buffer). Use tools/eval_refiner_depth_transfer.py instead."
        )

    model, ckpt_step = restore_model(args.checkpoint_path)
    batches = load_eval_batches(args.source, args.batches, args.skip)
    print(f"📚 Evaluating {len(batches)} batches: window-2 CE | fresh baseline vs window-1 depths 1..{MAX_STEPS_LIMIT}")

    # "fresh" = no hunch at all (control); depth d = hunch primed on window 1.
    # Same window-2 tokens in every condition, so the curve isolates the hunch.
    # Hard tokens are the worst quartile ranked under the FRESH condition: the
    # slice is fixed before the hunch enters, otherwise re-ranking would bias it.
    conditions = ["fresh"] + list(range(1, MAX_STEPS_LIMIT + 1))
    ce_sums = {c: 0.0 for c in conditions}
    ce_counts = {c: 0 for c in conditions}
    hard_sums = {c: 0.0 for c in conditions}
    hard_counts = {c: 0 for c in conditions}

    for batch in batches:
        window1 = batch[:, :MAX_SEQ_LEN]
        hard_mask = None
        for cond in conditions:
            model.hunch_cache[...] = jnp.zeros_like(model.hunch_cache[...])
            if cond == "fresh":
                token_ce, mask = score_window2(model, batch, use_hunch=False)
            else:
                prime_hunch(model, window1, cond)
                token_ce, mask = score_window2(model, batch, use_hunch=True)
            token_ce, mask = np.asarray(token_ce), np.asarray(mask)
            if cond == "fresh":
                threshold = np.quantile(token_ce[mask], 0.75)
                hard_mask = mask & (token_ce >= threshold)
            ce_sums[cond] += token_ce[mask].sum()
            ce_counts[cond] += mask.sum()
            hard_sums[cond] += token_ce[hard_mask].sum()
            hard_counts[cond] += hard_mask.sum()

    mean_ces = {c: ce_sums[c] / ce_counts[c] for c in conditions}
    hard_ces = {c: hard_sums[c] / hard_counts[c] for c in conditions}
    for c in conditions:
        label = "fresh (no hunch)" if c == "fresh" else f"depth {c}"
        print(f"  {label:>16}: CE {mean_ces[c]:.4f} | hard-quartile CE {hard_ces[c]:.4f}")

    depths = conditions[1:]
    print("-" * 40)
    print(f"All tokens   : fresh {mean_ces['fresh']:.4f} -> depth {depths[-1]} {mean_ces[depths[-1]]:.4f} | gain {mean_ces['fresh'] - mean_ces[depths[-1]]:+.4f}")
    print(f"Hard quartile: fresh {hard_ces['fresh']:.4f} -> depth {depths[-1]} {hard_ces[depths[-1]]:.4f} | gain {hard_ces['fresh'] - hard_ces[depths[-1]]:+.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ces, title, color in (
        (ax1, mean_ces, "All tokens", "#ff007b"),
        (ax2, hard_ces, "Hard quartile (ranked with no hunch)", "#ffaa00"),
    ):
        ax.plot(depths, [ces[d] for d in depths], marker="o", color=color, label="hunch from window 1")
        ax.axhline(ces["fresh"], linestyle="--", color="#888888", label="fresh slots (no hunch)")
        ax.set_title(title)
        ax.set_xlabel("Window-1 reasoning depth (steps)")
        ax.set_ylabel("Window-2 cross entropy (held-out)")
        ax.grid(True, alpha=0.2)
        ax.legend()
    fig.suptitle(f"Window-2 CE vs window-1 reasoning depth — checkpoint step {ckpt_step}")
    out_path = "depth_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✨ Saved {out_path}")


if __name__ == "__main__":
    main()
