"""instruments.corpus_overlap on toy buckets (#521, #485)."""

import numpy as np
import pytest

from instruments import corpus_overlap as co


@pytest.fixture
def bucket(tmp_path):
    """chunk_2 re-reads chunk_0 from token 1,000 on, then goes on with fresh text; chunk_1 is fresh."""
    rng = np.random.default_rng(0)
    first, second = (rng.integers(0, 50_000, 300_000, dtype=np.int32) for _ in range(2))
    np.save(tmp_path / "chunk_0.npy", first)
    np.save(tmp_path / "chunk_1.npy", second)
    np.save(tmp_path / "chunk_10.npy", rng.integers(0, 50_000, 300_000, dtype=np.int32))
    np.save(tmp_path / "chunk_2.npy", np.concatenate([first[1_000:200_000], rng.integers(0, 50_000, 50_000,
                                                                                          dtype=np.int32)]))
    return tmp_path, first


def test_chunks_are_read_in_the_order_the_prefill_wrote_them(bucket):
    assert [p.name for p in co.chunks(bucket[0])] == ["chunk_0.npy", "chunk_1.npy", "chunk_2.npy", "chunk_10.npy"]


def test_a_head_that_re_reads_an_earlier_chunk_is_found_and_a_fresh_one_is_not(bucket):
    rows = {name: (total, seen, where) for name, total, seen, where in co.heads(bucket[0], head_tokens=100_000)}
    total, seen, where = rows["chunk_2.npy"]
    assert seen == total and where == "chunk_0.npy"
    assert rows["chunk_1.npy"][1] == 0 and rows["chunk_10.npy"][1] == 0
    assert rows["chunk_0.npy"][1] == 0, "the first chunk has nothing before it"


def test_a_probe_is_found_in_the_bins_that_hold_it(bucket):
    path, first = bucket
    total, per_chunk = co.probe(path, first[150_000:160_000], bin_tokens=50_000)
    found = dict(per_chunk)
    assert total == 10_000 - co.WINDOW + 1
    assert found["chunk_0.npy"][3] == total, "tokens 150k-160k of chunk_0 sit in its fourth bin"
    assert sum(found["chunk_1.npy"]) == 0
    # chunk_2 re-read from token 1,000, so the stretch sits at 149k-159k there: split
    # across its third and fourth bins, and all of it found again.
    assert found["chunk_2.npy"][2:4] == [1_000, total - 1_000]


def test_windows_straddling_two_blocks_are_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "BLOCK_TOKENS", 1_000)
    tokens = np.random.default_rng(1).integers(0, 50_000, 5_000, dtype=np.int32)
    np.save(tmp_path / "chunk_0.npy", tokens)
    total, per_chunk = co.probe(tmp_path, tokens[990:1_040], bin_tokens=10_000)
    assert per_chunk[0][1][0] == total


def test_unique_counts_the_re_read_share(bucket):
    rows = {name: (sampled, repeats) for name, sampled, repeats in co.unique(bucket[0], keep_one_in=4)}
    assert rows["chunk_0.npy"][1] == 0 and rows["chunk_1.npy"][1] == 0
    sampled, repeats = rows["chunk_2.npy"]
    assert 0.75 < repeats / sampled < 0.85, "199k of chunk_2's 249k tokens re-read chunk_0"
