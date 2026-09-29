"""The prefill's near-dedup (#486) on synthetic documents: copies go, distinct documents
stay, the same seed judges the same way, a resumed index remembers, and the index keeps
its memory bound. tests/core/test_prefill_dedup.py holds the prefill's use of it."""

import numpy as np
import pytest

from trm.data.dedup import DedupParams, KeySet, MinHasher, NearDedup, shingles
from trm.settings import Config

PARAMS = DedupParams.of(Config())
VOCAB = [f"w{i}" for i in range(5000)]


def document(rng, words=400):
    return " ".join(rng.choice(VOCAB, size=words))


def edited(rng, text, changes):
    """`text` with `changes` words replaced, spread out so each touches its own shingles."""
    words = text.split()
    for position in np.linspace(10, len(words) - 10, changes).astype(int):
        words[position] = "edit" + str(rng.integers(1 << 30))
    return " ".join(words)


def jaccard(a, b):
    sa, sb = shingles(a, PARAMS.ngram), shingles(b, PARAMS.ngram)
    return len(sa & sb) / len(sa | sb)


def test_exact_and_near_copies_are_dropped_and_distinct_documents_kept():
    rng = np.random.default_rng(0)
    originals = [document(rng) for _ in range(200)]
    exact = originals[:60]
    near = [edited(rng, text, 3) for text in originals[60:120]]
    assert min(jaccard(a, b) for a, b in zip(originals[60:120], near)) > 0.9, \
        "the near copies must sit where LSH catches all but ~1e-5 of pairs"

    dedup = NearDedup(PARAMS)
    assert dedup.keep(originals) == originals, "no two random documents look alike"
    assert dedup.keep(exact + near) == [], "every copy goes"
    record = dedup.record()
    assert (record["docs_seen"], record["docs_dropped"]) == (320, 120)
    assert record["doc_removal_rate"] == pytest.approx(120 / 320)
    assert record["chars_dropped"] == sum(map(len, exact + near))


def test_below_the_threshold_documents_are_kept():
    """At Jaccard <= 0.3, 25 bands of 10 match with probability ~1.5e-4 per pair."""
    rng = np.random.default_rng(1)
    pairs = []
    for _ in range(100):
        base = document(rng)
        other = edited(rng, base, 90)
        assert jaccard(base, other) < 0.3
        pairs += [base, other]
    assert NearDedup(PARAMS).keep(pairs) == pairs


def test_the_signature_estimates_jaccard():
    """What the LSH cut rests on: two signatures agree on a share of entries equal, in
    expectation, to their documents' Jaccard similarity (256 entries: sd <= 0.031)."""
    rng = np.random.default_rng(2)
    hasher = MinHasher(PARAMS)
    for changes in (2, 10, 25, 60):
        a = document(rng)
        b = edited(rng, a, changes)
        agree = float(np.mean(hasher(a) == hasher(b)))
        assert agree == pytest.approx(jaccard(a, b), abs=0.1), changes


def test_short_documents_match_only_their_exact_copy():
    dedup = NearDedup(PARAMS)
    texts = ["hello world", "hello there", "+++", "---", "hello   world!", "+++"]
    assert dedup.keep(texts) == ["hello world", "hello there", "+++", "---"]


def test_the_same_seed_judges_the_same_way():
    rng = np.random.default_rng(3)
    texts = [document(rng, 60) for _ in range(20)]
    texts += [edited(rng, t, 1) for t in texts[:10]]
    first, second = NearDedup(PARAMS), NearDedup(PARAMS)
    assert first.keep(texts) == second.keep(texts)
    assert np.array_equal(first.keys.slots, second.keys.slots)
    reseeded = MinHasher(DedupParams.of(Config(DEDUP_SEED=7)))
    assert not np.array_equal(MinHasher(PARAMS)(texts[0]), reseeded(texts[0]))


def test_a_saved_index_resumes_where_it_stopped(tmp_path):
    rng = np.random.default_rng(4)
    before = [document(rng) for _ in range(30)]
    dedup = NearDedup(PARAMS)
    dedup.keep(before)
    path = tmp_path / "dedup_index.npz"
    dedup.save(path)

    resumed = NearDedup.load(path, PARAMS)
    assert resumed.counts == dedup.counts
    fresh = document(rng)
    assert resumed.keep([before[3], edited(rng, before[7], 2), fresh]) == [fresh]
    with pytest.raises(ValueError, match="built with"):
        NearDedup.load(path, DedupParams.of(Config(DEDUP_SEED=7)))


def test_the_index_keeps_its_memory_bound():
    """At most 32 bytes per key once past the starting table (dedup.py's docstring):
    a table doubled only when over half full is never under a quarter full."""
    rng = np.random.default_rng(5)
    index = KeySet(capacity=1 << 10)
    added = rng.integers(1, 1 << 63, size=(4000, PARAMS.bands), dtype=np.uint64)
    for keys in added:
        index.add(keys)
        assert index.slots.nbytes <= max(1 << 13, 32 * index.count)
    assert index.count == added.size == np.count_nonzero(index.slots)
    assert all(index.contains_any(keys) for keys in added[::97])
    absent = rng.integers(1, 1 << 63, size=(200, PARAMS.bands), dtype=np.uint64)
    assert not any(index.contains_any(keys) for keys in absent)


def _optimal_bands_and_rows(threshold, num_perm):
    """datasketch's `_optimal_param` (lsh.py): the cut minimizing the equal-weight sum of
    the false-positive area below the threshold and the false-negative area above it."""
    s = np.linspace(0.0, 1.0, 4001)
    below, above = s <= threshold, s >= threshold
    best, cut = np.inf, None
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            match = 1 - (1 - s ** r) ** b
            error = np.trapezoid(match[below], s[below]) + np.trapezoid(1 - match[above], s[above])
            if error < best:
                best, cut = error, (b, r)
    return cut


def test_the_default_bands_and_rows_are_the_optimum_for_the_threshold():
    assert _optimal_bands_and_rows(PARAMS.threshold, PARAMS.num_perm) == (PARAMS.bands, PARAMS.rows)


@pytest.mark.parametrize("environment", [
    {"DEDUP_THRESHOLD": "0.85"},                       # threshold moved, bands left behind
    {"DEDUP_BANDS": "30"},                             # 30 x 10 > 256 permutations
])
def test_a_threshold_without_its_bands_refuses(environment):
    with pytest.raises(SystemExit, match="DEDUP_ROWS"):
        Config.from_env(environment)


def test_a_threshold_with_its_bands_is_accepted():
    """0.85 over 256 permutations is 13 x 19 by the same optimum."""
    config = Config.from_env({"DEDUP_THRESHOLD": "0.85", "DEDUP_BANDS": "13", "DEDUP_ROWS": "19"})
    assert (config.DEDUP_BANDS, config.DEDUP_ROWS) == (13, 19)
