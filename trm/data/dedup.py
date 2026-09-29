"""Near-duplicate removal for the prefill: MinHash + LSH over each document's raw text (#486).

Adopted, not tested: near-dedup is settled in favour (Lee et al. 2022, "Deduplicating
Training Data Makes Language Models Better"), and the settings are StarCoder's (Li et
al. 2023, §3.5: MinHash + LSH, word 5-grams, Jaccard 0.7) with the 256 permutations of
the BigCode pipeline it follows (huggingface.co/blog/dedup). The knobs live on
trm.settings.Config (DEDUP_*); this module takes them as a `DedupParams`.

How a document is judged:
  1. Shingles: split on anything that is not [A-Za-z0-9_] (BigCode's NON_ALPHA), drop the
     empty pieces, and take every run of `ngram` consecutive words. Dropping the empty
     pieces is a small departure from BigCode: re-indenting a file does not make it new.
     A document shorter than `ngram` words is one shingle, so only its exact copy matches.
  2. Signature: each shingle is hashed (the first 4 bytes of its SHA-1, as BigCode), then
     through `num_perm` random affine maps modulo the Mersenne prime 2^61-1; the signature
     is the minimum of each map. Two documents agree on one entry with probability equal
     to the Jaccard similarity of their shingle sets.
  3. LSH: the signature is cut into `bands` bands of `rows` entries, and each band hashed
     to one 64-bit key. A document is a near-duplicate when any of its band keys was
     already kept: with 25 bands of 10 rows, a pair at Jaccard 0.5 matches 2.4% of the
     time, at 0.7 51%, at 0.8 94%, at 0.9 100%. There is no exact-Jaccard check after
     the match, as in BigCode's final pipeline: it would mean keeping every document.

Streaming, first seen wins: documents are judged in stream order and a duplicate is
dropped without entering the index, so the kept stream holds no near-duplicate pair
the index can see, and the last document kept has been checked against every document
before it. That is what makes a source's tail (the held-out slice of #363) clean
against the rows a run trains on, within LSH's recall.

Memory: the index is `bands` 64-bit keys per KEPT document, in one open-addressing table
kept between a quarter and half full: at most 32 x bands bytes per kept document (800 B
at 25 bands), and half as much again for the moment the table doubles. Nothing else
grows with the stream. A signature costs `num_perm` x 4 bytes, computed in blocks of
`_SHINGLE_BLOCK` shingles, so one huge document cannot allocate more than ~8 MB.
"""

import dataclasses
import hashlib
import json
import os
import re

import numpy as np

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
_SHINGLE_BLOCK = 4096


@dataclasses.dataclass(frozen=True)
class DedupParams:
    """What a corpus was deduplicated with: recorded beside it (status.json) whole."""
    threshold: float
    num_perm: int
    bands: int
    rows: int
    ngram: int
    seed: int

    @classmethod
    def of(cls, config):
        """The run's DEDUP_* knobs (trm/settings.py)."""
        return cls(threshold=config.DEDUP_THRESHOLD, num_perm=config.DEDUP_NUM_PERM,
                   bands=config.DEDUP_BANDS, rows=config.DEDUP_ROWS,
                   ngram=config.DEDUP_NGRAM, seed=config.DEDUP_SEED)

    def record(self):
        return {"method": "minhash-lsh", **dataclasses.asdict(self)}


def shingles(text, ngram):
    """The set of word `ngram`-grams a document is compared by (module docstring, step 1)."""
    words = [w for w in NON_ALPHA.split(text) if w]
    if len(words) < ngram:
        return {" ".join(words) if words else text}
    return {" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)}


class MinHasher:
    """text -> its MinHash signature (uint32[num_perm]). Picklable, so a worker pool can
    run it; the maps are drawn from `seed` alone, so every process computes the same one."""

    def __init__(self, params):
        self.ngram = params.ngram
        rng = np.random.default_rng(params.seed)
        # a < 2^31 and a 32-bit shingle hash keep a*h + b under 2^64: the modulus is
        # taken on the exact value, never on a wrapped one.
        self.a = rng.integers(1, 1 << 31, size=params.num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, size=params.num_perm, dtype=np.uint64)

    def __call__(self, text):
        digests = b"".join(hashlib.sha1(s.encode("utf-8")).digest()[:4] for s in shingles(text, self.ngram))
        hashes = np.frombuffer(digests, dtype="<u4").astype(np.uint64)
        signature = np.full(self.a.shape, MAX_HASH, dtype=np.uint64)
        for start in range(0, hashes.size, _SHINGLE_BLOCK):
            block = hashes[start:start + _SHINGLE_BLOCK, None]
            signature = np.minimum(signature, (((block * self.a + self.b) % MERSENNE_PRIME) & MAX_HASH).min(axis=0))
        return signature.astype(np.uint32)


def _mix64(z):
    """splitmix64's finalizer, elementwise over uint64 (wrapping multiply)."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def band_keys(signature, bands, rows):
    """One 64-bit key per band. The band's index is mixed in, so all bands share one table
    without two bands' equal rows colliding; 0 is the table's empty mark, so it moves to 1."""
    table = signature[:bands * rows].astype(np.uint64).reshape(bands, rows)
    keys = _mix64(np.arange(1, bands + 1, dtype=np.uint64))
    for j in range(rows):
        keys = _mix64(keys ^ table[:, j])
    keys[keys == 0] = 1
    return keys


class KeySet:
    """A set of nonzero 64-bit keys in one numpy array: open addressing, linear probing,
    doubled when more than half full (the memory bound in the module docstring)."""

    def __init__(self, slots=None, count=0, capacity=1 << 20):
        self.slots = np.zeros(capacity, dtype=np.uint64) if slots is None else slots
        self.count = int(count)

    def contains_any(self, keys):
        mask = np.uint64(self.slots.size - 1)
        pending, idx = keys, keys & mask
        while pending.size:
            found = self.slots[idx]
            if np.any(found == pending):
                return True
            occupied = found != 0
            pending, idx = pending[occupied], (idx[occupied] + np.uint64(1)) & mask
        return False

    def add(self, keys):
        """Insert keys none of which is in the set yet."""
        keys = np.unique(keys)
        size = self.slots.size
        while 2 * (self.count + keys.size) > size:
            size *= 2
        if size != self.slots.size:
            grown = np.zeros(size, dtype=np.uint64)
            _place(grown, self.slots[self.slots != 0])
            self.slots = grown
        _place(self.slots, keys)
        self.count += keys.size


def _place(slots, keys):
    """Linear-probe `keys` into `slots`, all at once: each round, every key whose slot is
    free and first in line takes it, and the rest step one slot on."""
    mask = np.uint64(slots.size - 1)
    idx = keys & mask
    while keys.size:
        free = np.flatnonzero(slots[idx] == 0)
        _, first = np.unique(idx[free], return_index=True)
        won = free[first]
        slots[idx[won]] = keys[won]
        lost = np.ones(keys.size, dtype=bool)
        lost[won] = False
        keys, idx = keys[lost], (idx[lost] + np.uint64(1)) & mask


class NearDedup:
    """One source's streaming near-dedup: judges documents in order, keeps the index and
    the counts the corpus's manifest records."""

    COUNTERS = ("docs_seen", "docs_dropped", "chars_seen", "chars_dropped")

    def __init__(self, params, keys=None, counts=None):
        self.params = params
        self.hasher = MinHasher(params)
        self.keys = KeySet() if keys is None else keys
        self.counts = dict.fromkeys(self.COUNTERS, 0) if counts is None else dict(counts)

    def admit(self, signature, size):
        """True keeps the document (and indexes it); False drops it as a near-duplicate.
        `size` is its length in characters, for the manifest's share of text removed."""
        keys = band_keys(signature, self.params.bands, self.params.rows)
        duplicate = self.keys.contains_any(keys)
        if not duplicate:
            self.keys.add(keys)
        self.counts["docs_seen"] += 1
        self.counts["chars_seen"] += size
        self.counts["docs_dropped"] += duplicate
        self.counts["chars_dropped"] += size * duplicate
        return not duplicate

    def keep(self, texts, map_fn=map):
        """The texts that survive, in order. `map_fn` computes the signatures (a worker
        pool's `map`); the judging is sequential, since each depends on the ones before."""
        signatures = list(map_fn(self.hasher, texts))
        return [text for text, signature in zip(texts, signatures) if self.admit(signature, len(text))]

    def record(self):
        """The manifest entry: the parameters, the counts and the share removed."""
        c = self.counts
        return {**self.params.record(), **c,
                "doc_removal_rate": c["docs_dropped"] / max(c["docs_seen"], 1),
                "char_removal_rate": c["chars_dropped"] / max(c["chars_seen"], 1)}

    def save(self, path):
        """Write the index and counts atomically: a crash mid-write leaves the old file."""
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, slots=self.keys.slots, count=self.keys.count,
                     counts=np.array([self.counts[k] for k in self.COUNTERS], dtype=np.int64),
                     params=json.dumps(self.params.record()))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, params):
        """The index `save` wrote, refused if it was built with other parameters."""
        with np.load(path) as saved:
            recorded = json.loads(str(saved["params"]))
            if recorded != params.record():
                raise ValueError(f"{path} was built with {recorded}, not {params.record()}")
            keys = KeySet(saved["slots"], int(saved["count"]))
            counts = dict(zip(cls.COUNTERS, (int(v) for v in saved["counts"])))
        return cls(params, keys, counts)
