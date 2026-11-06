"""The same text painted by per-token CE at every milestone (#465).

Aggregate CE says how much a model learned; a fixed passage coloured by surprise says
what. Read across milestones it shows the order things are learned in: punctuation and
common words first, then grammar, then names and facts. This scores one short passage
per bucket (prose, code, math) at each milestone of a run and draws one figure per
milestone, every panel on one colour scale shared by all milestones, since a per-panel
scale would hide the very improvement the figure is drawn to show.

    FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu PYTHONPATH=. \\
        python -m instruments.surprise_text --checkpoint-path runs/<run>/checkpoints/milestones

**The passages** are cut from each bucket's held-out tail (the rows the trainer's
probe reads, #596): the first document that starts in the first held-out row, up to
`--tokens` tokens. They are written to `<out>/passages.json` with their sha256 the
first time, and every later call reads that file and refuses a passage whose sha no
longer matches, so a run's milestones are always scored on the same text, even after
the corpus changes.

**Scoring** is one window per passage, in f32 on CPU, with the targets masked as the
trainer's held-out CE masks them (the document separator is never a target, #373).
Printed per milestone and passage: the mean CE, and the targets whose CE fell most
since the previous milestone, the concrete "what it just learned".
"""

from __future__ import annotations

import os

# CPU and f32 by default, before anything imports jax: this may run beside a trainer.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("FORCE_F32_COMPUTE", "1")

import argparse
import hashlib
import json
import pathlib
import resource
import time

import numpy as np

from instruments._common import add_checkpoint_argument, git_head, load_env
from instruments.results import emit

REPORTS = {
    "passage CE": ("measured", "mean CE over every target of a fixed held-out passage, f32 on CPU"),
    "largest drops": ("measured", "the targets whose CE fell most since the previous milestone"),
}

SOURCES = ("fineweb-edu", "stack-edu", "finemath")
DEFAULT_TOKENS = 160
DROPS_SHOWN = 8
# The shared colour scale's top: this quantile of every CE scored, over all milestones
# and passages, so a few near-impossible targets don't wash the rest out.
SCALE_QUANTILE = 0.99
# Layout of a panel: a monospace grid this many characters wide.
LINE_CHARS = 96


# A passage shorter than this many tokens says too little; the cut looks further on.
MIN_PASSAGE = 32


def passage_from_row(row, length, eot, shortest=MIN_PASSAGE):
    """The first document that starts inside `row` (a held-out row of token ids) with
    at least `shortest` tokens before it ends, up to `length` tokens, never crossing a
    separator; the row's head when no document in it is that long."""
    row = np.asarray(row).reshape(-1)
    want = min(length, shortest)
    for start in (np.flatnonzero(row[:-1] == eot) + 1).tolist() + [0]:
        tokens = row[start:start + length]
        end = np.flatnonzero(tokens == eot)
        tokens = tokens[:end[0]] if end.size else tokens
        if len(tokens) >= want or start == 0:
            return tokens
    return row[:length]


def sha(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def token_text(enc, token):
    """What one token shows on the page. A byte-level token that is not valid UTF-8 on
    its own (part of a multi-byte character) shows as its bytes in hex, so it is still
    drawn, once, where it sits."""
    raw = enc.decode_single_token_bytes(int(token))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{b:02x}" for b in raw)


def layout(texts, width=LINE_CHARS):
    """(line, column, text) for each target's text, in order, on a grid `width`
    characters wide: a newline inside a token ends its line, and a token that would
    run past the edge starts the next one. Every target gets exactly one cell; an
    empty or whitespace-only token keeps its cell, drawn as a visible marker. A token
    that spans lines is drawn where it starts, labelled with its first visible piece
    (↵ when it starts with the newline); what it puts after its last newline still
    moves the cursor, so a newline-and-indent token keeps the code's indentation."""
    cells, line, col = [], 0, 0
    for text in texts:
        pieces = text.replace("\t", "    ").split("\n")
        visible = next((piece.strip() for piece in pieces if piece.strip()), "")
        head = pieces[0] if pieces[0].strip() else ("↵" if len(pieces) > 1 else "·")
        if len(pieces) > 1 and not pieces[0].strip() and visible:
            head = "↵" + visible
        if col + len(head) > width and col > 0:
            line, col = line + 1, 0
        cells.append((line, col, head))
        col += len(head)
        if len(pieces) > 1:
            line, col = line + len(pieces) - 1, len(pieces[-1])
    return cells


def colour_scale(all_ce, quantile=SCALE_QUANTILE):
    """The one (vmin, vmax) every panel of every milestone shares."""
    flat = np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c in all_ce])
    return 0.0, max(float(np.quantile(flat, quantile)), 1e-6)


def largest_drops(before, after, texts, shown=DROPS_SHOWN):
    """[(text, ce before, ce after)] for the targets whose CE fell most."""
    fall = np.asarray(before) - np.asarray(after)
    order = np.argsort(-fall)[:shown]
    return [(texts[i], float(before[i]), float(after[i])) for i in order if fall[i] > 0]


def score_passage(model, tokens, pad_id):
    """Per-target CE of `tokens[1:]` given what precedes each, in f32."""
    import jax
    import jax.numpy as jnp

    from trm.train.validation import heldout_targets

    inputs = jnp.asarray(tokens[:-1], dtype=jnp.int32)[None, :]
    targets = heldout_targets(jnp.asarray(tokens[1:], dtype=jnp.int32)[None, :], pad_id)
    logits = model(inputs, training=False).logits
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return np.asarray(nll[0]), np.asarray(targets[0] != pad_id)


def milestone_steps(checkpoint_path):
    path = pathlib.Path(checkpoint_path)
    return sorted(int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit())


def load_passages(path, data_root, config, length):
    """The pinned passages: read from `path`, or cut from the corpus tails and written
    there. A passage whose tokens no longer hash to its recorded sha is refused; a
    pinned one keeps its own length, whatever `length` asks."""
    from trm.config import EOT_TOKEN_ID
    from trm.train.validation import ValidationProbe

    if path.exists():
        saved = json.loads(path.read_text())
        for name, entry in saved.items():
            if sha(entry["tokens"]) != entry["sha256"]:
                raise SystemExit(f"{path}: passage {name!r} no longer matches its sha256")
        print(f"📌 passages pinned in {path}; --tokens applies only to a new pin")
        return {name: np.asarray(entry["tokens"]) for name, entry in saved.items()}
    passages = {}
    for source in SOURCES:
        rows = ValidationProbe.of(config, data_root, source=source, rows=1).load_rows()
        if rows:
            passages[source] = passage_from_row(rows[0], length, EOT_TOKEN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({name: {"tokens": [int(t) for t in tokens], "sha256": sha(tokens)}
                                for name, tokens in passages.items()}, indent=1))
    return passages


def draw(path, step, panels, scale):
    """One figure: a panel per passage, each target's text on a cell shaded by its CE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap, (vmin, vmax) = plt.get_cmap("magma_r"), scale
    lines = [max((c[0] for c in cells), default=0) + 1 for _, cells, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 0.26 * sum(lines) + 1.0 * len(panels)),
                             gridspec_kw={"height_ratios": [n + 2 for n in lines]}, squeeze=False)
    for ax, (name, cells, ce), n in zip(axes[:, 0], panels, lines, strict=True):
        ax.set_xlim(0, LINE_CHARS)
        ax.set_ylim(n + 0.5, -1.2)
        ax.axis("off")
        ax.text(0, -0.9, f"{name}: mean CE {float(np.mean(ce)):.3f} nats", fontsize=9, weight="bold")
        for (line, col, text), value in zip(cells, ce, strict=True):
            shade = cmap((min(value, vmax) - vmin) / (vmax - vmin))
            ax.add_patch(plt.Rectangle((col, line - 0.45), len(text), 0.9, color=shade, linewidth=0))
            ink = "white" if value > (vmin + vmax) / 2 else "black"
            ax.text(col, line, text, family="monospace", fontsize=7.4, va="center", color=ink)
    fig.suptitle(f"surprise per token at step {step} (shared scale 0–{vmax:.2f} nats)", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def default_out(checkpoint_path):
    """`runs/<run>/surprise_text` in this checkout, <run> being the checkpoint path's
    last `run_*` folder: a live run's `runs/<run>/checkpoints/milestones` and a cold
    run's `<cold>/<run>/milestones` both name their own run, and the figures stay on
    the SSD. None when the path names no run."""
    from trm.settings import REPO_ROOT
    names = [part for part in pathlib.Path(os.path.abspath(checkpoint_path)).parts if part.startswith("run_")]
    return REPO_ROOT / "runs" / names[-1] / "surprise_text" if names else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_checkpoint_argument(ap, required=True, aliases=("--ckpt",))
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="passage length, in tokens")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where the passages and figures go (default: <run>/surprise_text)")
    args = ap.parse_args(argv)

    from instruments.heldout_profile import MIN_AVAILABLE_GIB, memory_available_gib
    available = memory_available_gib()
    if available is not None and available < MIN_AVAILABLE_GIB:
        raise SystemExit(f"only {available:.1f} GiB available (< {MIN_AVAILABLE_GIB}); not starting beside a trainer")

    load_env()
    import tiktoken

    from trm.config import TOKENIZER_NAME, resolve_root
    from trm.runtime.restore import restore_model
    from trm.settings import CONFIG

    if args.tokens > CONFIG.MAX_SEQ_LEN + 1:
        raise SystemExit(f"--tokens {args.tokens}: one window holds at most {CONFIG.MAX_SEQ_LEN + 1} tokens")
    t0 = time.time()
    out = args.out or default_out(args.checkpoint_path)
    if out is None:
        raise SystemExit(f"{args.checkpoint_path} names no run_* folder: say where the figures go with --out")
    passages = load_passages(out / "passages.json", resolve_root(os.environ.get("DATA_ROOT", "runs/data")),
                             CONFIG, args.tokens)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    texts = {name: [token_text(enc, t) for t in tokens[1:]] for name, tokens in passages.items()}
    steps = milestone_steps(args.checkpoint_path)
    print(f"{len(passages)} passages x {len(steps)} milestones; commit {git_head()}")
    for name, tokens in passages.items():
        print(f"  {name}: {len(tokens) - 1} targets, sha256 {sha(tokens)[:16]}")

    scored = {}
    for step in steps:
        model, _ = restore_model(CONFIG, args.checkpoint_path, step=step)
        scored[step] = {}
        for name, tokens in passages.items():
            nll, keep = score_passage(model, tokens, CONFIG.PAD_TOKEN_ID)
            scored[step][name] = (nll, keep)
    scale = colour_scale([nll[keep] for by in scored.values() for nll, keep in by.values()])

    previous = None
    for step in steps:
        panels = []
        for name in passages:
            nll, keep = scored[step][name]
            shown = [t for t, k in zip(texts[name], keep, strict=True) if k]
            ce = nll[keep]
            panels.append((name, layout(shown), ce))
            emit(f"{name}@{step}", mean_ce=float(ce.mean()))
            line = f"step {step:>8} {name:<12} mean CE {ce.mean():.3f}"
            if previous is not None:
                drops = largest_drops(scored[previous][name][0][keep], ce, shown)
                line += "  fell most: " + ", ".join(f"{t!r} {b:.1f}→{a:.1f}" for t, b, a in drops)
            print(line)
        draw(out / f"step_{step}.png", step, panels, scale)
        previous = step

    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"\n🖼  {len(steps)} figures in {out} | shared scale 0–{scale[1]:.2f} nats | "
          f"{time.time() - t0:.0f}s on CPU, peak RSS {peak_gib:.1f} GiB, no GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
