"""Decode the model at every reasoning depth and plot CE against depth.

This is the chess-engine curve: if extra reasoning steps genuinely refine the
prediction, CE should fall monotonically with depth (with diminishing returns).
A flat curve means the loop is dead weight.

Run offline against the latest (or a given) checkpoint:
    PYTHONPATH=. python tools/eval_depth_curve.py [--batches 16] [--skip 200000]
"""

import os

# Eval needs no optimizer/gradient memory; a modest arena is plenty and leaves
# room in case a training process is still holding the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse

import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
from dotenv import load_dotenv

from config import LATENT_DIM, MAX_SEQ_LEN, MAX_STEPS_LIMIT, BATCH_SIZE, PAD_TOKEN_ID, resolve_root
from model import UniversalReasoner
from data_loaders import TextDataGenerator
from checkpoint_utils import discover_latest_checkpoint_run

load_dotenv()


@nnx.jit(static_argnames=["max_steps"])
def eval_token_ces(model, tokens, max_steps):
    """Per-token CE and validity mask, so the caller can slice by difficulty."""
    seq_in, seq_out = tokens[:, :MAX_SEQ_LEN], tokens[:, 1:MAX_SEQ_LEN + 1]
    out = model(seq_in, max_steps=max_steps, training=False, should_refresh=True)
    mask = seq_out != PAD_TOKEN_ID
    token_ce = optax.softmax_cross_entropy_with_integer_labels(logits=out.logits, labels=seq_out)
    return token_ce, mask


def restore_model(checkpoint_path):
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "optimizer", "monitor_state", "step"),
    )
    latest = mngr.latest_step()
    if latest is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}")
    print(f"📖 Restoring model weights from step {latest} ({checkpoint_path})")
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
    )
    nnx.update(model, restored["model"])
    return model, latest


def load_eval_batches(source_dir, num_batches, skip):
    gen = TextDataGenerator(source_dir)
    # Jump past the data the training run has consumed so the curve is measured
    # on sequences the model has never seen.
    gen.skip_count = skip
    batches = []
    while len(batches) < num_batches:
        batch, _ = gen.get_batch(BATCH_SIZE)
        if batch is None:
            break
        batches.append(batch)
    if not batches:
        raise SystemExit(f"No eval data available in {source_dir} after skipping {skip} samples.")
    return batches


def main():
    parser = argparse.ArgumentParser(description="CE-vs-depth diagnostic")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Orbax checkpoint dir (defaults to the latest run's)")
    parser.add_argument("--batches", type=int, default=16, help="number of held-out batches to average over")
    parser.add_argument("--skip", type=int, default=200_000, help="samples to skip so eval data is past the trained range")
    parser.add_argument("--source", type=str, default="pretrain/fineweb-edu", help="data subdirectory under DATA_ROOT")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")

    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise SystemExit("DATA_ROOT is not set.")
    source_dir = f"{resolve_root(data_root)}/{args.source}"

    model, ckpt_step = restore_model(os.path.abspath(checkpoint_path))
    batches = load_eval_batches(source_dir, args.batches, args.skip)
    print(f"📚 Evaluating {len(batches)} batches (segment 1, fresh slots) at depths 1..{MAX_STEPS_LIMIT}")

    # Same batches at every depth, so the curve isolates depth and nothing else.
    # Hard tokens are the worst quartile *ranked at depth 1*: the set is fixed
    # before depth varies, otherwise re-ranking per depth would bias the slice.
    depths = list(range(1, MAX_STEPS_LIMIT + 1))
    ce_sums = {d: 0.0 for d in depths}
    ce_counts = {d: 0 for d in depths}
    hard_sums = {d: 0.0 for d in depths}
    hard_counts = {d: 0 for d in depths}

    import numpy as np
    for batch in batches:
        hard_mask = None
        for depth in depths:
            token_ce, mask = eval_token_ces(model, batch, depth)
            token_ce, mask = np.asarray(token_ce), np.asarray(mask)
            if depth == 1:
                valid_ces = token_ce[mask]
                threshold = np.quantile(valid_ces, 0.75)
                hard_mask = mask & (token_ce >= threshold)
            ce_sums[depth] += token_ce[mask].sum()
            ce_counts[depth] += mask.sum()
            hard_sums[depth] += token_ce[hard_mask].sum()
            hard_counts[depth] += hard_mask.sum()

    mean_ces = [ce_sums[d] / ce_counts[d] for d in depths]
    hard_ces = [hard_sums[d] / hard_counts[d] for d in depths]
    for d, ce, hard in zip(depths, mean_ces, hard_ces):
        print(f"  depth {d}: CE {ce:.4f} | hard-quartile CE {hard:.4f}")

    print("-" * 40)
    print(f"All tokens   : depth 1 CE {mean_ces[0]:.4f} -> depth 8 CE {mean_ces[-1]:.4f} | gain {mean_ces[0] - mean_ces[-1]:+.4f}")
    print(f"Hard quartile: depth 1 CE {hard_ces[0]:.4f} -> depth 8 CE {hard_ces[-1]:.4f} | gain {hard_ces[0] - hard_ces[-1]:+.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(depths, mean_ces, marker="o", color="#ff007b")
    ax1.set_title("All tokens")
    ax2.plot(depths, hard_ces, marker="o", color="#ffaa00")
    ax2.set_title("Hard quartile (ranked at depth 1)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Reasoning depth (steps)")
        ax.set_ylabel("Cross entropy (held-out)")
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"CE vs reasoning depth — checkpoint step {ckpt_step}")
    out_path = "depth_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✨ Saved {out_path}")


if __name__ == "__main__":
    main()
