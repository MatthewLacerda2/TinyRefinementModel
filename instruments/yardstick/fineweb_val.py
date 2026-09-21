"""FineWeb validation CE: the public yardstick for token efficiency (#462).

The modded-nanogpt speedrun trains GPT-2-small-sized models on FineWeb and races to
a validation loss of 3.28 on a fixed shard, the value llm.c's GPT-2 124M reached
after 10B tokens. It uses the GPT-2 BPE, which is our r50k_base, so CE per token on
that same text compares with no conversion. Our own val CE (FineWeb-Edu, #373 mask)
compares only with ourselves.

Protocol, as the speedrun reads its shard:
  - the first SPEEDRUN_VAL_TOKENS tokens of `fineweb_val_000000.bin` (a 256-int32
    header, then uint16 ids; every document opens with the end-of-text token);
  - cut into consecutive non-overlapping windows, each target scored once;
  - every target counts, the end-of-text ones included. Ours is trained to predict
    it since #373, and the reference scores it, so excluding it would score easier
    text than the number beside it.

What it cannot match: the window. The speedrun scores at >= 1,024 tokens of
context; we score at our MAX_SEQ_LEN (512). Shorter context reads higher CE, so the
gap runs against us, and the window is printed beside every number.

Like yardstick.py this module is model-agnostic and imports no jax: the model is
`logits_fn(tokens) -> logits`, numpy in and out.
"""

import os
import urllib.request

import numpy as np

from instruments.yardstick.yardstick import verify_sha256

FINEWEB_VAL_URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin"
# The file's Git LFS object id on the Hub, which is its sha256.
FINEWEB_VAL_SHA256 = "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640"
FINEWEB_VAL_CACHE = "runs/data/eval/fineweb_val_000000.bin"
HEADER_INT32S, MAGIC = 256, 20240520
# The speedrun's val_tokens: 10 * 2^20.
SPEEDRUN_VAL_TOKENS = 10_485_760
EOT = 50256

REFERENCE = {
    "val_ce": 3.28,
    "source": "modded-nanogpt speedrun target: llm.c GPT-2 124M after 10B FineWeb tokens, "
              "on this shard's first 10,485,760 tokens at a 1,024-token window",
}
# OpenAI's GPT-2 124M (HF `gpt2`, trained on WebText, not FineWeb) measured by this
# module through calibrate_gpt2 on the shard's first 2^18 targets, 2026-09-21 (#462).
# The pair of windows is the window's own share of any gap: 0.086 nats.
GPT2_MEASURED = {512: 3.4272, 1024: 3.3416}


def fetch_fineweb_val(path=FINEWEB_VAL_CACHE):
    """Download-and-cache the pinned shard; verify its sha256 either way."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"⬇️  Downloading the FineWeb val shard -> {path}")
        urllib.request.urlretrieve(FINEWEB_VAL_URL, path)
    verify_sha256(path, FINEWEB_VAL_SHA256)
    return path


def read_tokens(path, count=SPEEDRUN_VAL_TOKENS):
    """The first `count` token ids of a speedrun-format shard, as int32."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INT32S)
    if header[0] != MAGIC:
        raise ValueError(f"{path}: not a speedrun token shard (magic {header[0]})")
    if count > header[2]:
        raise ValueError(f"{path} holds {header[2]} tokens, fewer than the {count} asked for")
    ids = np.memmap(path, dtype=np.uint16, mode="r", offset=4 * HEADER_INT32S, shape=(int(header[2]),))
    return np.asarray(ids[:count], dtype=np.int32)


def windows(tokens, seq):
    """(inputs, targets), each [n, seq]: consecutive windows over `tokens`, so every
    token after the first is a target exactly once. A tail too short for a window
    is dropped, and its length reported by the caller from the target count."""
    n = (len(tokens) - 1) // seq
    if n == 0:
        raise ValueError(f"{len(tokens)} tokens make no {seq}-token window")
    inputs = tokens[:n * seq].reshape(n, seq)
    targets = tokens[1:n * seq + 1].reshape(n, seq)
    return inputs, targets


def score(logits_fn, tokens, seq, batch=4, progress=None):
    """Mean CE (nats) over every target of `tokens` in `seq`-long windows, and the
    target count, with the share of targets that are end-of-text."""
    inputs, targets = windows(tokens, seq)
    total = 0.0
    for start in range(0, len(inputs), batch):
        logits = np.asarray(logits_fn(inputs[start:start + batch]), dtype=np.float64)
        top = logits.max(-1, keepdims=True)
        logz = np.log(np.exp(logits - top).sum(-1)) + top[..., 0]
        picked = np.take_along_axis(logits, targets[start:start + batch, :, None], -1)[..., 0]
        total += float((logz - picked).sum())
        if progress:
            progress(min(start + batch, len(inputs)), len(inputs))
    return {"val_ce": total / targets.size, "targets": int(targets.size), "window": seq,
            "eot_share": float((targets == EOT).mean())}


def document_keys(tokens, width=24):
    """A 64-bit key per document start in `tokens`: the `width` tokens after each
    end-of-text, hashed. Used to ask whether this shard's documents sit inside a
    training corpus (FineWeb-Edu is a filtered subset of FineWeb)."""
    starts = np.flatnonzero(tokens[:-width] == EOT) + 1
    spans = tokens[starts[:, None] + np.arange(width)].astype(np.uint64)
    keys = np.zeros(len(starts), dtype=np.uint64)
    for column in spans.T:  # FNV-style mix; wraps mod 2^64 on purpose
        keys = (keys ^ column) * np.uint64(1099511628211)
    return keys


def overlap(val_tokens, corpus_files, width=24):
    """How many of the shard's documents open with the same `width` tokens as some
    document in `corpus_files` (int32 .npy token streams, documents EOT-separated)."""
    wanted = np.unique(document_keys(val_tokens, width))
    found = np.zeros(0, dtype=np.uint64)
    for path in corpus_files:
        found = np.union1d(found, np.intersect1d(wanted, document_keys(np.load(path, mmap_mode="r"), width)))
    return {"documents": int(len(wanted)), "found_in_corpus": int(len(found)), "width": width}
