"""A forward pass shown: the logit lens per block, and the last token's path through the cloud (#472).

The output head is tied to the embedding, so the model's prediction is literally which
point of the token cloud the final state points at. For a few fixed prompts this takes
the last token's residual stream after the embedding and after every block, and writes:

  * **the lens** (`lens.json`, `lens.png`): each state through the final RMSNorm and the
    tied head, top-k tokens and probabilities per block. The ground truth of this tool;
  * **the path** (`path.glb`): the cloud of the most frequent tokens' embedding rows,
    and the states as line segments `layer_0` ... `layer_{N-1}`, one per block, with
    scale keyframes so the path grows one block at a time and loops. An illustration.

    FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu PYTHONPATH=. \\
        python -m instruments.logit_lens --checkpoint-path runs/<run>/checkpoints/milestones --step N

**What the path is, and is not.** Each state is drawn post-final-norm, the vector the
head reads, then rescaled to the cloud's mean radius: the head scores a token by its dot
product with the state, so the direction is what predicts, and the raw scale grows with
depth while the cloud's does not. It is projected on the cloud's own top-3 principal
directions (`embedding_geometry.projection`), whose share of the cloud's variance is
printed with it. Three of 960 dimensions: a path that ends near a token here is a shadow
of the prediction, and the lens is what says the prediction.

**The gate.** The lens at the last block must reproduce the model's own next-token
argmax exactly; the tool refuses to write anything when it doesn't, since a lens that
disagrees with the model is miswired. Each prompt is preceded by the document separator,
as every document in the corpus is.
"""

from __future__ import annotations

import os

# CPU and f32 by default, before anything imports jax: this may run beside a trainer.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("FORCE_F32_COMPUTE", "1")

import argparse
import json
import pathlib
import resource
import struct
import time

import numpy as np

from instruments._common import add_checkpoint_argument, git_head, load_env
from instruments.results import emit

REPORTS = {
    "lens top-k": ("measured", "each block's last-token state through the final norm and the tied head"),
    "explained variance": ("measured", "share of the cloud's variance on the 3 directions the path is drawn in"),
}

DEFAULT_PROMPTS = ("The capital of France is", "def add(a, b):\n    return", "1, 2, 3, 4,")
TOP_K = 5
CLOUD_TOKENS = 3000
# Seconds each block's segment takes to grow in the animation; the loop holds the whole
# path for one more segment's time before it restarts.
SEGMENT_SECONDS = 0.6
# A cell this probable is shaded dark enough that its label reads in white.
WHITE_INK_ABOVE = 0.5


def lens(model, tokens):
    """[blocks + 1, vocab] probabilities of the next token after `tokens`, read from the
    stream after the embedding (row 0) and after every block, through the final norm and
    the tied head; and the model's own next-token logits for the gate."""
    import jax
    import jax.numpy as jnp

    batch = jnp.asarray(np.asarray(tokens, dtype=np.int32))[None, :]
    states = model.capture_trajectory(batch)[:, 0, -1, :]  # [blocks + 1, dim]
    normed = model.out_norm(states.astype(model.dtype))
    head = model.embed.embedding[...].astype(model.dtype).T
    logits = jnp.matmul(normed, head, preferred_element_type=jnp.float32)
    own = model(batch, training=False).logits[0, -1]
    return (np.asarray(jax.nn.softmax(logits, axis=-1)), np.asarray(normed, dtype=np.float32),
            np.asarray(own))


def top_k(probs, k=TOP_K):
    """[(token id, probability)] of each row's k most likely tokens, most likely first."""
    order = np.argsort(-probs, axis=-1)[:, :k]
    return [[(int(t), float(row[t])) for t in ids] for row, ids in zip(probs, order, strict=True)]


def gate(probs, own_logits):
    """The lens's last row must pick the model's own next token."""
    return int(np.argmax(probs[-1])) == int(np.argmax(own_logits))


def path_points(normed_states, centre, directions, radius):
    """States as points in the cloud's projection: each rescaled to `radius` (direction
    only, see the module doc), centred like the cloud, projected on its directions."""
    unit = normed_states / np.maximum(np.linalg.norm(normed_states, axis=1, keepdims=True), 1e-9)
    return (unit * radius - centre) @ directions.T


def explained(rows, directions):
    """Share of `rows`' variance on `directions` (orthonormal)."""
    centred = rows - rows.mean(0)
    return float(np.square(centred @ directions.T).sum() / np.square(centred).sum())


# --- glTF 2.0, binary (.glb): written by hand, the format is small and no library is pinned
_FLOAT, _VEC3, _SCALAR = 5126, "VEC3", "SCALAR"
_POINTS, _LINES = 0, 1


class _Glb:
    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors, self.meshes, self.nodes = [], [], [], []
        self.samplers, self.channels = [], []

    def accessor(self, values, kind):
        data = np.ascontiguousarray(values, dtype=np.float32)
        while len(self.blob) % 4:
            self.blob.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": data.nbytes})
        self.blob += data.tobytes()
        flat = data.reshape(len(data), -1)
        self.accessors.append({"bufferView": len(self.views) - 1, "componentType": _FLOAT, "count": len(data),
                               "type": kind, "min": flat.min(0).tolist(), "max": flat.max(0).tolist()})
        return len(self.accessors) - 1

    def node(self, name, positions, mode, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        self.meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": self.accessor(positions, _VEC3)},
                                                          "mode": mode}]})
        self.nodes.append({"name": name, "mesh": len(self.meshes) - 1,
                           "translation": list(map(float, translation)), "scale": list(map(float, scale))})
        return len(self.nodes) - 1

    def animate_scale(self, node, times, scales):
        self.samplers.append({"input": self.accessor(np.asarray(times)[:, None], _SCALAR),
                              "output": self.accessor(scales, _VEC3), "interpolation": "LINEAR"})
        self.channels.append({"sampler": len(self.samplers) - 1, "target": {"node": node, "path": "scale"}})

    def bytes(self):
        doc = {"asset": {"version": "2.0", "generator": "instruments.logit_lens"},
               "scene": 0, "scenes": [{"nodes": list(range(len(self.nodes)))}],
               "nodes": self.nodes, "meshes": self.meshes, "accessors": self.accessors,
               "bufferViews": self.views, "buffers": [{"byteLength": len(self.blob)}]}
        if self.channels:
            doc["animations"] = [{"name": "path", "samplers": self.samplers, "channels": self.channels}]
        text = json.dumps(doc).encode()
        text += b" " * (-len(text) % 4)
        blob = bytes(self.blob) + b"\0" * (-len(self.blob) % 4)
        total = 12 + 8 + len(text) + 8 + len(blob)
        return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(text), b"JSON") + text
                + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


def path_glb(cloud, points):
    """The cloud as points, and the path as one segment per block (`layer_k`, from the
    state before block k+1 to the state after it), each growing from its start in turn."""
    glb = _Glb()
    glb.node("cloud", cloud, _POINTS)
    steps = len(points) - 1
    loop = (steps + 1) * SEGMENT_SECONDS
    for k in range(steps):
        start, end = points[k], points[k + 1]
        node = glb.node(f"layer_{k}", np.stack([np.zeros(3), end - start]), _LINES,
                        translation=start, scale=(0.0, 0.0, 0.0))
        grow, done = k * SEGMENT_SECONDS, (k + 1) * SEGMENT_SECONDS
        times = sorted({0.0, grow, done, loop})
        scales = [[0.0] * 3 if t <= grow else [1.0] * 3 for t in times]
        glb.animate_scale(node, times, scales)
    return glb.bytes()


def draw(path, prompts, tables, labels):
    """One panel per prompt: rows are the states (embedding at the bottom, the last
    block at the top, the flow diagram's reading direction), the top-k cells shaded by
    probability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(prompts), figsize=(5.2 * len(prompts), 4.2), squeeze=False)
    for ax, prompt, table in zip(axes[0], prompts, tables, strict=True):
        rows = len(table)
        probs = np.array([[p for _, p in row] for row in table])
        ax.imshow(probs[::-1], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r, row in enumerate(table[::-1]):
            for c, (tid, p) in enumerate(row):
                ax.text(c, r, f"{labels(tid)!r}\n{p:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if p > WHITE_INK_ABOVE else "black")
        ax.set_yticks(range(rows), [f"block {rows - 1 - r}" if r < rows - 1 else "embedding" for r in range(rows)],
                      fontsize=7)
        ax.set_xticks([])
        ax.set_title(f"{prompt!r} →", fontsize=8)
    fig.suptitle("logit lens: the next-token guess after each block (top 5)", fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_checkpoint_argument(ap, required=True, aliases=("--ckpt",))
    ap.add_argument("--step", type=int, default=None, help="step to read (default: the dir's newest)")
    ap.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    ap.add_argument("--cloud-tokens", type=int, default=CLOUD_TOKENS,
                    help="how many of the corpus's most frequent tokens make the cloud")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="where lens.json, lens.png and path.glb go")
    args = ap.parse_args(argv)

    load_env()
    import tiktoken

    from instruments.embedding_geometry import frequent_tokens, projection
    from trm.config import EOT_TOKEN_ID, TOKENIZER_NAME, resolve_root
    from trm.runtime.restore import restore_model
    from trm.settings import CONFIG

    t0 = time.time()
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    model, step = restore_model(CONFIG, args.checkpoint_path, step=args.step)
    embedding = np.asarray(model.embed.embedding[...], dtype=np.float32)
    ids = frequent_tokens(resolve_root(os.environ.get("DATA_ROOT", "runs/data")),
                          ["fineweb-edu", "stack-edu", "finemath"], args.cloud_tokens, 2_000_000)
    cloud_rows = embedding[ids]
    centre, directions = projection(cloud_rows, components=3)
    share = explained(cloud_rows, directions)
    radius = float(np.linalg.norm(cloud_rows, axis=1).mean())
    cloud = (cloud_rows - centre) @ directions.T

    def label(tid):
        return enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace") if tid < enc.n_vocab else "<pad>"

    tables, record = [], {"step": int(step), "checkpoint": str(args.checkpoint_path), "commit": git_head(short=False),
                          "path_projection": {"components": 3, "explained_variance": share,
                                              "drawn": "post-final-norm, rescaled to the cloud's mean radius"},
                          "prompts": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(args.prompts):
        tokens = [EOT_TOKEN_ID, *enc.encode(prompt)]
        probs, normed, own = lens(model, tokens)
        if not gate(probs, own):
            raise SystemExit(f"{prompt!r}: the lens's last row picks {int(np.argmax(probs[-1]))}, the model "
                             f"{int(np.argmax(own))}: the lens is miswired, nothing written")
        table = top_k(probs)
        tables.append(table)
        record["prompts"][prompt] = [[{"token": label(t), "id": t, "p": p} for t, p in row] for row in table]
        print(f"\n{prompt!r}")
        for depth, row in reversed(list(enumerate(table))):
            name = "embedding" if depth == 0 else f"block {depth}"
            print(f"  {name:>9}: " + "  ".join(f"{label(t)!r} {p:.2f}" for t, p in row))
        emit(f"prompt{i}@{step}", final_top1=table[-1][0][1])
        if i == 0:
            (args.out / "path.glb").write_bytes(path_glb(cloud, path_points(normed, centre, directions, radius)))

    (args.out / "lens.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    draw(args.out / "lens.png", args.prompts, tables, label)
    emit(f"projection@{step}", explained_variance=share)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"\nlens gate: the last block reproduces the model's argmax on all {len(args.prompts)} prompts")
    print(f"path: prompt 1, {len(normed) - 1} segments, drawn post-final-norm at the cloud's radius, on 3 "
          f"directions holding {share:.1%} of the cloud's variance ({len(ids)} tokens)")
    print(f"🧾 {args.out} | {time.time() - t0:.0f}s on CPU, peak RSS {peak:.1f} GiB, no GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
