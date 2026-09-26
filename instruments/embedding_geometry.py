"""How the token embedding organizes itself over a run, as a picture and as numbers (#464).

The vivid "latent space crystallizing" animations come from grokking on a closed
algorithmic vocabulary, where one correct geometry exists to snap into. A 50,304-token
language model on web text has no such shape, so the honest version of that picture is
this: the embedding of the most frequent tokens, at every milestone, under **one
projection**, so that what moves is the model and not the projection.

It reads checkpoints that already exist. No new telemetry, no card, nothing added to a
run — so it can be pointed at the live run's milestones while it trains.

    FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m instruments.embedding_geometry \\
        --checkpoint-path runs/<run>/checkpoints/milestones

A picture here is a hypothesis, so three numbers are printed beside it (and emitted as
RESULT lines) for every milestone:

  separation   mean cosine between tokens of the same kind, minus between kinds, over the
               NAMED kinds only. `other` is "none of the above" and has no reason to
               cohere, so counting it as a kind would put ~23% of the signal on a bucket
               that means nothing. Reported per kind as well, because the named kinds are
               wildly unequal in size: words are ~76% of the same-kind pairs, so the
               headline is mostly about them and the small kinds need their own number.
  anisotropy   the share of variance on the first principal direction. LM embeddings are
               famously anisotropic (Ethayarajh 2019); watching it move says whether the
               spread is going into one direction or spreading out.
  rms          the embedding's RMS norm, which grows through training and is the reason
               the frames are drawn on a shared scale.
"""

from __future__ import annotations

import os

# CPU and f32 before anything imports jax or trm.config: this runs beside a trainer.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("FORCE_F32_COMPUTE", "1")

import argparse
import collections
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from instruments._common import add_checkpoint_argument, git_head, load_env
from instruments.arch import add_arch_argument
from instruments.plots import AQUA, BLUE, GRID, INK, INK_DIM, ORANGE, STYLE, _note
from instruments.results import emit
from instruments.runlog import checkpoint_steps, recorded_tokens_per_opt_step

REPORTS = {
    "separation, anisotropy, rms": ("measured", "the checkpoint's own embedding rows for the "
                                    "--tokens most frequent ids, counted in the corpus slice read"),
}

# Four kinds drawn in colour (the palette validates three saturated slots plus ink) and
# everything else as dim background: a token's kind comes from what it decodes to.
KINDS = ("punctuation", "digits", "code", "word", "other")
COLOURS = {"punctuation": ORANGE, "digits": AQUA, "code": BLUE, "word": INK, "other": GRID}
_CODE = re.compile(r"^[\s]*[(){}\[\]<>=+\-*/%|&^~!;:@#$\\`_]+$")
_WORD = re.compile(r"^\s[A-Za-z][A-Za-z']*$")
_PUNCT = re.compile(r"^[\s]*[.,;:!?'\"“”‘’…\-–—]+[\s]*$")


def kind_of(text):
    """Which kind a decoded token belongs to. Order matters: a bare '=' is code, a
    bare '.' is punctuation, and both are punctuation characters."""
    if text.strip() == "" or _PUNCT.match(text):
        return "punctuation"
    if any(ch.isdigit() for ch in text) and not any(ch.isalpha() for ch in text):
        return "digits"
    if _CODE.match(text):
        return "code"
    if _WORD.match(text):
        return "word"
    return "other"


def frequent_tokens(data_root, sources, count, budget):
    """The `count` most frequent token ids over `budget` tokens read from each source's
    shards. Frequency comes from our own corpus, so the picture is about the tokens this
    model actually sees, not about a guessed vocabulary."""
    counts = collections.Counter()
    for source in sources:
        for shard in sorted(pathlib.Path(f"{data_root}/pretrain/{source}").glob("chunk_*.npy"))[:1]:
            ids, tally = np.unique(np.asarray(np.load(shard, mmap_mode="r")[:budget]), return_counts=True)
            counts.update(dict(zip(ids.tolist(), tally.tolist())))
    return np.array([tid for tid, _ in counts.most_common(count)], dtype=np.int64)


def token_kinds(ids):
    import tiktoken

    from trm.config import TOKENIZER_NAME
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    kinds = []
    for tid in ids:
        if int(tid) >= enc.n_vocab:  # a padding slot the tokenizer has no token for
            kinds.append("other")
            continue
        kinds.append(kind_of(enc.decode_single_token_bytes(int(tid)).decode("utf-8", errors="replace")))
    return np.array(kinds)


def projection(rows, components=2):
    """The projection this instrument draws in: the top principal directions of `rows`
    (the LAST milestone), returned with the centre they are taken around so every earlier
    milestone can be projected through the same one."""
    centre = rows.mean(0)
    _, _, vt = np.linalg.svd(rows - centre, full_matrices=False)
    return centre, vt[:components]


NAMED_KINDS = tuple(k for k in KINDS if k != "other")


def _gap(gram, same, off):
    """Mean cosine inside `same` minus inside `off`, or nan when either side is empty:
    undefined says "not measured here", 0 would say "measured, no grouping"."""
    return float(gram[same].mean() - gram[off].mean()) if same.any() and off.any() else float("nan")


def readings(rows, kinds):
    """separation, anisotropy and rms of one milestone's embedding rows, plus a
    separation per named kind (that kind against everything else)."""
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    gram = unit @ unit.T
    named = np.isin(kinds, NAMED_KINDS)
    same = np.zeros_like(gram, dtype=bool)
    for kind in NAMED_KINDS:
        picked = kinds == kind
        same |= np.outer(picked, picked)
    np.fill_diagonal(same, False)
    off = np.outer(named, named) & ~same
    np.fill_diagonal(off, False)
    spectrum = np.linalg.svd(rows - rows.mean(0), compute_uv=False) ** 2
    out = {"separation": _gap(gram, same, off),
           # Undefined, not 1: identical rows have no direction to put variance on.
           "anisotropy": float(spectrum[0] / spectrum.sum()) if spectrum.sum() > 0 else float("nan"),
           "rms": float(np.sqrt((rows ** 2).mean()))}
    for kind in NAMED_KINDS:
        picked = kinds == kind
        inside = np.outer(picked, picked)
        np.fill_diagonal(inside, False)
        out[f"separation_{kind}"] = _gap(gram, inside, np.outer(picked, ~picked))
    return out


def density(points, limit, bins=150, smooth=4.0):
    """A smoothed 2D histogram of `points` on the shared [-limit, limit] square, peak
    normalised to 1. Gaussian-blurred by hand (a separable box blur applied three times
    approximates one) so the instrument keeps its dependency list to numpy."""
    grid, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=bins,
                                range=[[-limit, limit], [-limit, limit]])
    # Odd width, always: mode="same" centres an even kernel half a sample off, and three
    # passes per axis would draw every field 1.5 bins up and to the right of its tokens.
    width = max(int(smooth) | 1, 1)
    kernel = np.ones(width) / width
    for _ in range(3):
        grid = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, grid)
        grid = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, grid)
    return grid / max(grid.max(), 1e-12)


def panel(ax, points, kinds, title, limit):
    """Each kind laid down as a soft field of its own colour, so where two kinds share
    ground the colours mix into the shade in between and the regions read as regions.
    The dots stay on top, faint: the field says where the mass is, the dots say that it
    is made of tokens."""
    for kind in KINDS:
        picked = kinds == kind
        if not picked.any():
            continue
        field = density(points[picked], limit)
        layer = np.zeros(field.shape + (4,))
        layer[..., :3] = matplotlib.colors.to_rgb(COLOURS[kind])
        layer[..., 3] = np.clip(field ** 0.55, 0, 1) * (0.45 if kind == "other" else 0.8)
        ax.imshow(layer, extent=(-limit, limit, -limit, limit), origin="lower",
                  interpolation="bilinear", zorder=1)
        ax.scatter(points[picked, 0], points[picked, 1], s=1.6, linewidths=0, zorder=2,
                   color=COLOURS[kind], alpha=0.30 if kind == "other" else 0.55, label=kind)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def figure(frames, kinds, out, run_label):
    """One panel per milestone, on one shared scale. A deliberate deviation from
    instruments/plots.py's one-chart-per-image rule: the comparison across milestones IS
    the chart, and separate images would be compared by memory."""
    # The 99.5th percentile, not the maximum: one far-flung token would shrink every
    # cloud to a dot, and the point of a shared scale is to show the growth.
    limit = float(np.percentile(np.abs(frames[-1][1]), 99.5)) * 1.25
    with plt.rc_context(STYLE):
        columns = min(len(frames), 3)
        rows = (len(frames) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.9 * columns, 4.1 * rows), squeeze=False)
        for ax in axes.ravel()[len(frames):]:
            ax.axis("off")
        for ax, (label, points, reading) in zip(axes.ravel(), frames):
            panel(ax, points, kinds, label, limit)
            ax.set_xlabel(f"separation {reading['separation']:+.3f}   rms {reading['rms']:.2f}",
                          fontsize=7.5, color=INK_DIM)
        axes[0][0].legend(loc="upper left", fontsize=7, markerscale=2.5, handletextpad=0.2)
        fig.suptitle("token embedding, every milestone in one projection", fontsize=12.5,
                     fontweight="bold", color=INK)
        fig.text(0.99, 0.005, f"{run_label} · projection fitted on the last milestone, "
                 "applied to all · frequent tokens only", ha="right", fontsize=7, color=INK_DIM)
        fig.savefig(out)
        plt.close(fig)
    return out


def write_point_cloud(points, kinds, path, arm=0.012):
    """The cloud as an OBJ the board can turn: one named object per kind, and every token
    a three-segment cross, because the mesh widget draws EDGES — a bare vertex is
    invisible there. `arm` is the cross's half-length as a share of the cloud's extent."""
    span = float(np.abs(points).max())
    step = arm * span
    lines, vertex = [f"# token embedding point cloud, {len(points)} tokens"], 1
    for kind in KINDS:
        picked = np.flatnonzero(kinds == kind)
        if not picked.size:
            continue
        lines.append(f"o {kind}")
        body = []
        for point in points[picked]:
            for axis in range(3):
                for sign in (-1, 1):
                    offset = np.zeros(3)
                    offset[axis] = sign * step
                    tip = point + offset
                    lines.append(f"v {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}")
                body.append(f"l {vertex} {vertex + 1}")
                vertex += 2
        lines.extend(body)
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
    return path


def run_dir(checkpoint_path):
    """The run a checkpoint dir belongs to: runs/<run>/checkpoints, or a subdir of it.
    A dir with no run_metadata.json beside it is not a run, and saying so beats writing
    a run's figure and journal into whatever directory happened to be two levels up."""
    path = pathlib.Path(os.path.abspath(checkpoint_path))
    for candidate in (path.parent, path.parent.parent):
        if (candidate / "run_metadata.json").exists():
            return candidate
    raise SystemExit(f"{checkpoint_path}: no run_metadata.json above it — point --checkpoint-path "
                     "at a run's checkpoints dir (or its milestones subdir)")


def tokens_per_micro_step(run_dir):
    """Tokens in one checkpointed step, from the run's OWN recipe (#305). A checkpoint
    step is a micro-step, so the accumulation factor comes back out of the opt-step
    figure. None when the run did not record the three knobs, and the caller says so."""
    path = pathlib.Path(run_dir) / "run_metadata.json"
    if not path.exists():
        return None
    params = json.loads(path.read_text()).get("parameters", {})
    per_opt = recorded_tokens_per_opt_step(params)
    try:
        return per_opt and per_opt // int(params["ACCUMULATION_STEPS"])
    except (KeyError, TypeError, ValueError):
        return None


def trend_figure(rows, out, run_label, tokens_per_step):
    """Separation per kind against tokens — one chart, the plots.py convention, in the
    same colours the clouds use so the two figures read as one. The overall line is the
    named kinds together; anisotropy and rms are in the JSON beside it, not here, because
    they are a different quantity and this chart answers one question."""
    # Without the run's recipe there is no honest x axis in tokens, so the steps stay steps.
    tokens = [r["step"] * (tokens_per_step or 1) for r in rows]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for kind in NAMED_KINDS:
            ax.plot(tokens, [r[f"separation_{kind}"] for r in rows], color=COLOURS[kind],
                    marker="o", markersize=3.5, label=kind)
        ax.plot(tokens, [r["separation"] for r in rows], color=INK_DIM, linewidth=1.1,
                linestyle="--", label="all named kinds")
        ax.set_xscale("log")
        ax.set_xlabel("tokens trained" if tokens_per_step else
                      "checkpoint step (the run recorded no recipe to read tokens from)")
        ax.set_ylabel("separation")
        ax.legend(loc="upper left")
        ax.set_title("each kind finding its place")
        ax.set_title(run_label, loc="right", fontsize=7.5, fontweight="normal", color=INK_DIM)
        _note(ax, "separation = mean cosine inside a kind minus to everything else; "
                  "the small closed kinds group hardest")
        fig.savefig(out)
        plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_checkpoint_argument(ap, required=True, aliases=("--ckpt",))
    add_arch_argument(ap)
    ap.add_argument("--tokens", type=int, default=3000, help="how many of the most frequent ids to draw")
    ap.add_argument("--sources", default="fineweb-edu,codeparrot,finemath")
    ap.add_argument("--count-budget", type=int, default=2_000_000,
                    help="tokens read per source when counting frequency")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="where the figure goes")
    ap.add_argument("--obj-out", type=pathlib.Path, default=None,
                    help="also write the last milestone's cloud in 3D as an OBJ (for the board)")
    ap.add_argument("--obj-tokens", type=int, default=600,
                    help="how many of the most frequent tokens the 3D cloud holds; a wireframe "
                         "of thousands of crosses reads as fog and renders slowly")
    ap.add_argument("--obj-stride", type=int, default=2,
                    help="keep every Nth token of that range (2 halves it), so the cloud thins "
                         "out without becoming only the very commonest tokens")
    ap.add_argument("--tokens-per-step", type=int, default=None,
                    help="tokens per checkpointed step, for the trend's x axis; the default "
                         "comes from the run's own recipe (a checkpoint step is a micro-step: "
                         "batch x 2 windows x MAX_SEQ_LEN)")
    args = ap.parse_args(argv)

    load_env()
    from trm.config import resolve_root
    from trm.runtime.restore import restore_arch

    data_root = resolve_root(os.environ.get("DATA_ROOT", "runs/data"))
    ids = frequent_tokens(data_root, args.sources.split(","), args.tokens, args.count_budget)
    kinds = token_kinds(ids)
    print(f"🔤 {len(ids)} frequent tokens: "
          + "  ".join(f"{k} {int((kinds == k).sum())}" for k in KINDS if (kinds == k).any()))

    steps = checkpoint_steps(args.checkpoint_path)
    if not steps:
        raise SystemExit(f"no checkpoints under {args.checkpoint_path}")
    embeddings = []
    for step in steps:
        model, _ = restore_arch(args.arch, args.checkpoint_path, step=step)
        # Index first, cast second: the whole table in f64 is 386 MB, and this runs
        # beside a trainer on a box where this session is what the OOM killer picks.
        embeddings.append(np.asarray(model.embed.embedding[...])[ids].astype(np.float64))
        del model

    centre, axes = projection(embeddings[-1])
    frames, rows = [], []
    for step, rows_at_step in zip(steps, embeddings):
        reading = readings(rows_at_step, kinds)
        frames.append((f"step {step:,}", (rows_at_step - centre) @ axes.T, reading))
        rows.append({"step": int(step), **reading})
        print(f"  step {step:>9,}  separation {reading['separation']:+.4f}  "
              f"anisotropy {reading['anisotropy']:.4f}  rms {reading['rms']:.3f}")
        emit(f"embedding@{step}", **reading)

    run = run_dir(args.checkpoint_path)
    per_step = args.tokens_per_step or tokens_per_micro_step(run)
    out = args.out or run / "embedding_geometry.png"
    print(f"\n🖼  {figure(frames, kinds, out, run.name)}")
    print(f"🖼  {trend_figure(rows, out.with_name(out.stem + '_trend.png'), run.name, per_step)}")
    if args.obj_out:
        keep = slice(0, args.obj_tokens * args.obj_stride, args.obj_stride)
        centre3, axes3 = projection(embeddings[-1], components=3)
        cloud = (embeddings[-1][keep] - centre3) @ axes3.T
        print(f"🧊 {write_point_cloud(cloud, kinds[keep], args.obj_out)}")
    with open(run / "embedding_geometry.jsonl", "a") as handle:
        handle.write(json.dumps({"commit": git_head(short=False), "tokens": len(ids),
                                 "milestones": rows}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
