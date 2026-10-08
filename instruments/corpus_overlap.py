"""Where do a token stretch's 32-token windows reappear in a bucket? (#521, #485)

    python -m instruments.corpus_overlap heads runs/data/pretrain/fineweb-edu
    python -m instruments.corpus_overlap probe runs/data/pretrain/codeparrot --heldout

`heads`: does any chunk start by re-reading what an earlier chunk already holds? A
prefill resumed from a document count that its stream read as a record count
re-wrote what it had already written (#521). The first `--head-tokens` of every chunk
are hashed, and one pass over the bucket counts how many of each head's windows
appear in the chunks before it. A fresh chunk shares boilerplate only (licence
headers, navigation text); a re-read one shares most of its head.

`probe`: how much of a token stretch (the by-source held-out slice with `--heldout`)
reappears in the stream, per chunk and per `--bin` tokens (#485). The held-out slice
is a few files whose forks sit in the training data, and the bins where they recur
are where the probe's CE stepped down.

Windows are hashed with a polynomial over 64-bit integers; two different windows
that collide are counted as a match, which at 2^64 is a rounding error.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

REPORTS = {
    "head windows seen earlier": ("measured", "distinct 32-token windows of each chunk's head found in any earlier "
                                              "chunk, over the head's distinct windows"),
    "probe windows per bin": ("measured", "distinct probe windows found in each --bin tokens of the stream"),
    "repeated share": ("sampled", "windows kept by hash, one in --keep-one-in, that repeat an earlier window"),
}

WINDOW = 32
BASE = np.uint64(1_000_003)
# Hash this many tokens at a time: windows straddling two blocks are covered by a
# WINDOW - 1 token overlap.
BLOCK_TOKENS = 4_000_000
DEFAULT_HEAD_TOKENS = 1_000_000
DEFAULT_BIN_TOKENS = 2_000_000
# `unique` keeps one window in this many, by hash: ~60M samples for a 4B-token bucket.
DEFAULT_KEEP_ONE_IN = 64
# A head whose seen fraction is above this much did not come from fresh records.
REREAD_FRACTION = 0.5


def window_hashes(tokens: np.ndarray) -> np.ndarray:
    """One uint64 per WINDOW-token window of `tokens` (wrapping arithmetic)."""
    t = np.asarray(tokens).astype(np.uint64)
    n = t.size - WINDOW + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(WINDOW):
            h = h * BASE + t[j:j + n]
    return h


def found_in(sorted_hashes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Which of `values` occur in `sorted_hashes` (ascending): a binary search each,
    so a block sorted once is checked against many heads cheaply."""
    if sorted_hashes.size == 0:
        return np.zeros(values.size, dtype=bool)
    at = np.minimum(np.searchsorted(sorted_hashes, values), sorted_hashes.size - 1)
    return sorted_hashes[at] == values


def chunks(bucket: pathlib.Path) -> list[pathlib.Path]:
    """The bucket's chunk files in the order the prefill wrote them."""
    found = list(bucket.glob("chunk_*.npy"))
    return sorted(found, key=lambda p: int(re.search(r"chunk_(\d+)", p.name).group(1)))


def blocks(chunk: pathlib.Path):
    """(start token, window hashes) over the chunk, BLOCK_TOKENS at a time."""
    data = np.load(chunk, mmap_mode="r").reshape(-1)
    for start in range(0, data.size, BLOCK_TOKENS):
        yield start, window_hashes(data[start:start + BLOCK_TOKENS + WINDOW - 1])


def heads(bucket: pathlib.Path, head_tokens: int) -> list[tuple[str, int, int, str]]:
    """Per chunk: (name, distinct head windows, how many appear in an earlier chunk,
    the earlier chunk holding the most of them)."""
    files = chunks(bucket)
    head = [np.unique(window_hashes(np.load(f, mmap_mode="r").reshape(-1)[:head_tokens])) for f in files]
    seen = [np.zeros(h.size, dtype=bool) for h in head]
    most = [dict() for _ in files]
    for j, chunk in enumerate(files):
        for _, hashes in blocks(chunk):
            hashes = np.sort(hashes)
            for k in range(j + 1, len(files)):  # only chunks written after this one
                hit = found_in(hashes, head[k])
                if hit.any():
                    new = hit & ~seen[k]
                    most[k][chunk.name] = most[k].get(chunk.name, 0) + int(new.sum())
                    seen[k] |= hit
    return [(f.name, h.size, int(s.sum()), max(m, key=m.get) if m else "-")
            for f, h, s, m in zip(files, head, seen, most, strict=True)]


def unique(bucket: pathlib.Path, keep_one_in: int) -> list[tuple[str, int, int]]:
    """Per chunk: (name, sampled windows, how many repeat a window from earlier in the
    bucket). Windows are sampled by their hash (one in `keep_one_in`), so every copy
    of a window is sampled alike and repeats are counted exactly within the sample.
    A window repeated inside one chunk counts as a repeat too."""
    seen = np.zeros(0, dtype=np.uint64)
    rows = []
    for chunk in chunks(bucket):
        sampled = []
        for _, hashes in blocks(chunk):
            sampled.append(hashes[hashes % np.uint64(keep_one_in) == 0])
        here = np.concatenate(sampled) if sampled else np.zeros(0, dtype=np.uint64)
        distinct = np.unique(here)
        repeats = (here.size - distinct.size) + int(found_in(seen, distinct).sum())
        rows.append((chunk.name, int(here.size), repeats))
        seen = np.union1d(seen, distinct)
    return rows


def heldout_tokens(bucket: pathlib.Path, max_seq_len: int, data_seed: int) -> np.ndarray:
    """The by-source probe's own rows, as the trainer reads them (trm/train/validation.py)."""
    from trm.settings import CONFIG
    from trm.train.validation import VAL_TAIL_ROWS, corpus_samples, read_heldout_rows
    skip = max(corpus_samples(str(bucket), max_seq_len) - VAL_TAIL_ROWS, 0)
    rows = read_heldout_rows(str(bucket), CONFIG.EVAL_ROWS, skip, max_seq_len=max_seq_len, data_seed=data_seed)
    return np.concatenate([np.asarray(r).reshape(-1) for r in rows])


def probe(bucket: pathlib.Path, tokens: np.ndarray, bin_tokens: int) -> tuple[int, list[tuple[str, list[int]]]]:
    """(distinct probe windows, [(chunk, distinct probe windows found per bin)])."""
    target = np.unique(window_hashes(tokens))
    out = []
    for chunk in chunks(bucket):
        per_bin: dict[int, set] = {}
        for start, hashes in blocks(chunk):
            for b0 in range(0, hashes.size, bin_tokens):
                found = hashes[b0:b0 + bin_tokens]
                found = np.unique(found[found_in(target, found)])
                if found.size:
                    per_bin.setdefault((start + b0) // bin_tokens, set()).update(found.tolist())
        n_bins = -(-np.load(chunk, mmap_mode="r").size // bin_tokens)
        out.append((chunk.name, [len(per_bin.get(b, ())) for b in range(n_bins)]))
    return target.size, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("heads", help="does a chunk's head re-read an earlier chunk? (#521)")
    h.add_argument("bucket", type=pathlib.Path)
    h.add_argument("--head-tokens", type=int, default=DEFAULT_HEAD_TOKENS)
    u = sub.add_parser("unique", help="what share of each chunk repeats earlier text? (#521)")
    u.add_argument("bucket", type=pathlib.Path)
    u.add_argument("--keep-one-in", type=int, default=DEFAULT_KEEP_ONE_IN)
    p = sub.add_parser("probe", help="where do the held-out rows reappear in the stream? (#485)")
    p.add_argument("bucket", type=pathlib.Path)
    p.add_argument("--heldout", action="store_true", help="probe with the by-source held-out slice")
    p.add_argument("--bin", type=int, default=DEFAULT_BIN_TOKENS)
    args = ap.parse_args(argv)

    if args.mode == "unique":
        rows = unique(args.bucket, args.keep_one_in)
        print(f"{args.bucket}: windows sampled one in {args.keep_one_in}, repeats of anything earlier in the bucket")
        for name, sampled, repeats in rows:
            print(f"  {name:14s} {repeats / max(sampled, 1):6.1%} repeated  ({sampled:,} sampled)")
        total, repeated = sum(r[1] for r in rows), sum(r[2] for r in rows)
        print(f"  whole bucket: {repeated / max(total, 1):.1%} of windows repeat earlier text; "
              f"unique share {1 - repeated / max(total, 1):.1%}")
        return 0

    if args.mode == "heads":
        print(f"{args.bucket}: first {args.head_tokens:,} tokens of each chunk against the chunks before it")
        for name, total, seen, where in heads(args.bucket, args.head_tokens):
            frac = seen / max(total, 1)
            flag = "  <- re-read" if frac > REREAD_FRACTION else ""
            print(f"  {name:14s} {seen:>9,} of {total:>9,} windows seen earlier ({frac:6.1%}), most in {where}{flag}")
        return 0

    from trm.settings import CONFIG
    tokens = heldout_tokens(args.bucket, CONFIG.MAX_SEQ_LEN, CONFIG.DATA_SEED)
    total, per_chunk = probe(args.bucket, tokens, args.bin)
    print(f"{args.bucket}: {total:,} distinct held-out windows; per {args.bin:,}-token bin, how many recur")
    for name, bins in per_chunk:
        peak = max(bins) if bins else 0
        print(f"  {name:14s} max {peak:>7,} ({peak / max(total, 1):5.1%})  " + " ".join(map(str, bins)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
