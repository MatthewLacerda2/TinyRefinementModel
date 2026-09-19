"""Pictures of the residual stream inside a trained model: where each token's
state goes, block by block (#391).

    MODEL_ARCH=plain python -m instruments.trajectory_figures \\
        --checkpoint runs/<run>/checkpoints [--step N] [--rows 4] [--out DIR]

Three images, one chart each (the `instruments/plots.py` convention):

  trajectory_pca.png         each token's path through the blocks in 2D. The PCA is
                             fitted on the displacements z_k - z_0, never on z itself:
                             the raw states are dominated by a mean every position
                             shares, which would eat both components.
  trajectory_steps.png       how far the stream moves in each block, as a fraction of
                             how big it already is.
  trajectory_difficulty.png  the same, split by how sure the model ends up: tokens whose
                             final top-2 probabilities are nearly tied against tokens it
                             is confident about.

Offline, on a checkpoint, reproducible from the rows it reads. A picture here is a
hypothesis, not evidence: anything it suggests becomes a number through a
pre-registered pair before it reaches a findings file (#391, inherited from #228).
`--checkpoint` is the MANAGER ROOT, the directory holding numeric step dirs (see
`instruments/latents.py`).
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from instruments._common import add_checkpoint_argument, load_env

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "trajectory figures": ("sampled", "the residual stream over the held-out rows read, not the whole corpus"),
}

# How many token paths the PCA chart draws: enough to show the spread, few enough to read.
PATHS_DRAWN = 160
# Quartiles of the final top-2 probability gap: below the first is "near a tie",
# above the third is "confident".
TIE_QUANTILE, SURE_QUANTILE = 0.25, 0.75


def collect(model, rows, pad_token_id):
    """(states, gaps): every real token's trajectory, and how sure the model ends up.

    `states` is [K+1, P, dim] over the P non-pad positions of all rows; `gaps` is
    the final top-2 probability gap per position, [P]. Rows whose trajectory holds
    a non-finite value are skipped and counted, never averaged over (#229).
    """
    import jax
    import jax.numpy as jnp

    from instruments.latents import capture

    kept_states, kept_gaps, skipped = [], [], 0
    for row in rows:
        tokens = jnp.asarray(row)
        traj = capture(model, tokens, depth=None)
        if not traj.ok:
            skipped += 1
            continue
        probs = np.asarray(jax.nn.softmax(model(tokens).logits.astype(jnp.float32), axis=-1))
        top2 = np.sort(probs, axis=-1)[..., -2:]
        real = np.asarray(tokens != pad_token_id).reshape(-1)
        kept_states.append(traj.states.reshape(traj.states.shape[0], -1, traj.states.shape[-1])[:, real])
        kept_gaps.append((top2[..., 1] - top2[..., 0]).reshape(-1)[real])
    if skipped:
        print(f"trajectory_figures: skipped {skipped} row(s) with non-finite states (#229/#233)")
    if not kept_states:
        raise SystemExit("trajectory_figures: no row produced a finite trajectory")
    return np.concatenate(kept_states, axis=1), np.concatenate(kept_gaps)


def relative_steps(states):
    """‖z_k − z_{k−1}‖ / ‖z_{k−1}‖ per block and position, [K, P]: how much of the
    stream each block rewrites. Scale-free, so a stream that grows through the stack
    does not make the deep blocks look busy by size alone."""
    steps = np.linalg.norm(np.diff(states, axis=0), axis=-1)
    before = np.linalg.norm(states[:-1], axis=-1)
    return steps / np.maximum(before, 1e-12)


def displacement_pca(states):
    """(coords [K+1, P, 2], explained variance ratio [2]) of z_k − z_0.

    Fitted on the displacements of every block and position together, so all paths
    share one pair of axes. z_0 − z_0 is the origin: every path starts there."""
    disp = states - states[0]
    flat = disp[1:].reshape(-1, disp.shape[-1])
    mean = flat.mean(axis=0)
    _, singular, vt = np.linalg.svd(flat - mean, full_matrices=False)
    variance = singular ** 2
    coords = (disp - mean) @ vt[:2].T
    return coords - coords[0:1], variance[:2] / variance.sum()


def draw(states, gaps, outdir, label, seed=0):
    """Write the three figures; return their paths. Pure numpy and matplotlib, so a
    test can drive it without a model or a checkpoint."""
    import matplotlib.pyplot as plt

    from instruments.plots import AQUA, BLUE, INK_DIM, ORANGE, STYLE, _note

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    blocks = np.arange(1, states.shape[0])
    rel = relative_steps(states)
    written = []

    def save(fig, ax, name):
        ax.set_title(label, loc="right", fontsize=7.5, fontweight="normal", color=INK_DIM)
        path = outdir / name
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    def band(ax, values, colour, text):
        median = np.median(values, axis=1)
        low, high = np.percentile(values, [25, 75], axis=1)
        ax.fill_between(blocks, low, high, color=colour, alpha=0.18, linewidth=0)
        ax.plot(blocks, median, color=colour, linewidth=2.0, marker="o", markersize=4, label=text)

    with plt.rc_context(STYLE):
        # ── each token's path, in the plane its displacements vary most in
        coords, explained = displacement_pca(states)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(coords.shape[1], size=min(PATHS_DRAWN, coords.shape[1]), replace=False)
        fig, ax = plt.subplots(figsize=(11.0, 5.2))
        for p in chosen:
            ax.plot(coords[:, p, 0], coords[:, p, 1], color=INK_DIM, alpha=0.18, linewidth=0.7)
        cmap = plt.get_cmap("viridis")
        for k in blocks:
            ax.scatter(coords[k, chosen, 0], coords[k, chosen, 1], s=9, color=cmap(k / blocks[-1]),
                       label=f"after block {k}", zorder=3)
        ax.scatter([0], [0], s=40, color=ORANGE, zorder=4, label="the embedding (z₀)")
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of the displacement variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%})")
        ax.set_title("Each token's path through the blocks", loc="left")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        _note(ax, f"{len(chosen)} of {coords.shape[1]:,} token positions. The axes are fitted on "
                  "z_k − z₀ for every block and position together, so all paths share one plane; "
                  "a path is one token's state as the stack rewrites it.")
        save(fig, ax, "trajectory_pca.png")

        # ── how much of the stream each block rewrites
        fig, ax = plt.subplots(figsize=(11.0, 5.2))
        band(ax, rel, BLUE, "median, with the middle 50% of positions")
        ax.set_xticks(blocks)
        ax.set_xlabel("block")
        ax.set_ylabel("‖z_k − z_{k−1}‖ / ‖z_{k−1}‖")
        # Log: block 1 writes a stream tens of times the embedding it starts from,
        # which on a linear axis flattens every later block onto the floor.
        ax.set_yscale("log")
        ax.set_title("How much of the stream each block rewrites", loc="left")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        norms = np.median(np.linalg.norm(states, axis=-1), axis=1)
        _note(ax, "scale-free: each step over the size of the stream it was added to, on a log "
                  "axis because block 1 rewrites an embedding far smaller than its output. Median "
                  "‖z_k‖ by block: " + ", ".join(f"{n:.1f}" for n in norms) + ".")
        save(fig, ax, "trajectory_steps.png")

        # ── the same, for the tokens the model ends up unsure of vs sure of
        tie, sure = np.quantile(gaps, [TIE_QUANTILE, SURE_QUANTILE])
        fig, ax = plt.subplots(figsize=(11.0, 5.2))
        band(ax, rel[:, gaps <= tie], ORANGE, f"near a tie (top-2 gap ≤ {tie:.3f})")
        band(ax, rel[:, gaps >= sure], AQUA, f"confident (top-2 gap ≥ {sure:.3f})")
        ax.set_xticks(blocks)
        ax.set_xlabel("block")
        ax.set_ylabel("‖z_k − z_{k−1}‖ / ‖z_{k−1}‖")
        ax.set_yscale("log")
        ax.set_title("Unsure tokens against confident ones", loc="left")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        _note(ax, "tokens split by the gap between the final top-2 next-token probabilities: the "
                  "bottom quarter against the top quarter. A picture to form a question with, "
                  "not an answer.")
        save(fig, ax, "trajectory_difficulty.png")
    return written


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_checkpoint_argument(ap, required=True, aliases=("--checkpoint",))
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to restore (default: the newest the manager holds)")
    ap.add_argument("--source", default="pretrain/fineweb-edu")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", default=None, help="directory for the PNGs (default: beside the checkpoint)")
    args = ap.parse_args(argv)
    load_env()

    from trm.config import MAX_SEQ_LEN, MODEL_ARCH, PAD_TOKEN_ID
    from trm.runtime.restore import load_eval_batches, restore_arch

    model, _ = restore_arch(MODEL_ARCH, args.checkpoint_path, step=args.step)
    rows = [row[:, :MAX_SEQ_LEN] for row in load_eval_batches(args.source, num_rows=args.rows)]
    states, gaps = collect(model, rows, PAD_TOKEN_ID)

    checkpoint = pathlib.Path(args.checkpoint_path).resolve()
    step = args.step if args.step is not None else "latest"
    out = pathlib.Path(args.out) if args.out else checkpoint.parent / "trajectory" / str(step)
    label = f"{checkpoint.parent.name} · step {step}"
    for path in draw(states, gaps, out, label):
        print(f"  wrote {path}")
    rel = relative_steps(states)
    print(f"{states.shape[1]:,} positions from {len(rows)} row(s); median relative step by block: "
          + ", ".join(f"{v:.3f}" for v in np.median(rel, axis=1)))


if __name__ == "__main__":
    _main()
