"""The prefill's near-dedup pass (#486), end to end on a fake stream: copies never reach
a chunk, the manifest says what was removed, a resume keeps the index, and a folder is
only continued the way it was started, so `--dedup` cannot write into runs/data/.
No network and no tokenizer download: the stream, the tokenizer and the pool are fakes."""

import glob
import json
import os

import numpy as np
import pytest

import trm.data.prefill as prefill
from trm.data.dedup import DedupParams
from trm.settings import Config

PARAMS = DedupParams.of(Config())
STRIDE = 9
EOT = 50256


class FakeStream:
    def __init__(self, items, start=0):
        self.items, self.start = items, start

    def __iter__(self):
        return iter(self.items[self.start:])

    def skip(self, n):
        return FakeStream(self.items, self.start + n)


class InlinePool:
    map = staticmethod(lambda fn, xs: list(map(fn, xs)))


def fake_tokens(text):
    """One id per word (its index in the vocabulary below), then the separator."""
    return [int(word[1:]) for word in text.split()] + [EOT]


@pytest.fixture
def prefill_into(tmp_path, monkeypatch):
    monkeypatch.setattr(prefill, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(prefill, "TOKENIZE_BATCH_ITEMS", 16)
    monkeypatch.setattr(prefill, "TOKENS_PER_FILE", 3000)
    monkeypatch.setattr(prefill, "tokenize_batch_parallel", fake_tokens)

    def run(items, params=PARAMS, target=10**9):
        monkeypatch.setattr(prefill, "load_dataset", lambda *a, **k: FakeStream(items))
        cfg = {"path": "fake/code", "target_tokens": target, "folder": "pretrain", "alias": "code"}
        prefill.process_dataset(InlinePool(), cfg, STRIDE, params)
        folder = tmp_path / "pretrain" / "code"
        with open(folder / "status.json") as f:
            return folder, json.load(f)
    return run


def documents(seed, n, words=120):
    rng = np.random.default_rng(seed)
    return [" ".join(f"w{i}" for i in rng.integers(0, 5000, size=words)) for _ in range(n)]


def near_copy(text):
    words = text.split()
    words[len(words) // 2] = "w4999"
    return " ".join(words)


def test_copies_never_reach_a_chunk_and_the_manifest_says_so(prefill_into):
    originals = documents(0, 80)
    copies = originals[:20] + [near_copy(t) for t in originals[20:40]]
    # Empty records are read from the stream but are not documents: the resume offset
    # must count them, or a resume re-reads what it already wrote.
    items = [{"content": t} for t in originals] + [{"content": ""}] * 5 + [{"content": t} for t in copies]
    folder, status = prefill_into(items)

    written = np.concatenate([np.load(f) for f in sorted(glob.glob(str(folder / "chunk_*.npy")))])
    kept_tokens = sum(len(fake_tokens(t)) for t in originals)
    assert kept_tokens - STRIDE < written.size <= kept_tokens, "only the originals were tokenized"
    assert status["items_processed"] == 80 and status["stream_offset"] == len(items)
    dedup = status["dedup"]
    assert {k: dedup[k] for k in PARAMS.record()} == PARAMS.record()
    assert (dedup["docs_seen"], dedup["docs_dropped"]) == (120, 40)
    assert dedup["doc_removal_rate"] == pytest.approx(1 / 3)
    assert sorted(os.listdir(folder)) == sorted(
        [os.path.basename(f) for f in glob.glob(str(folder / "chunk_*.npy"))] + ["dedup_index.npz", "status.json"]), \
        "the loaders read every .npy in a bucket as tokens: the index must not be one"


def test_a_resume_remembers_what_it_kept(prefill_into):
    first, later = documents(1, 60), documents(2, 30)
    items = [{"content": t} for t in first + [near_copy(t) for t in first[:15]] + later]
    _, stopped = prefill_into(items, target=3000)
    assert stopped["stream_offset"] < 60, "the first run stopped at its target, mid-stream"

    _, finished = prefill_into(items)
    assert finished["stream_offset"] == len(items)
    assert finished["dedup"]["docs_dropped"] == 15, "copies of documents kept before the stop still go"
    assert finished["items_processed"] == 90


def test_a_folder_is_only_continued_the_way_it_was_started(prefill_into):
    items = [{"content": t} for t in documents(3, 40)]
    prefill_into(items, params=None, target=1000)
    with pytest.raises(SystemExit, match="point DATA_ROOT at a new folder"):
        prefill_into(items)


def test_a_deduped_folder_is_not_continued_without_dedup(prefill_into):
    items = [{"content": t} for t in documents(4, 40)]
    prefill_into(items, target=1000)
    with pytest.raises(SystemExit, match="point DATA_ROOT at a new folder"):
        prefill_into(items, params=None)
    other = DedupParams.of(Config(DEDUP_SEED=7))
    with pytest.raises(SystemExit, match="point DATA_ROOT at a new folder"):
        prefill_into(items, params=other)


def test_a_legacy_status_is_a_corpus_written_without_dedup(prefill_into, tmp_path):
    """runs/data/'s folders carry a status.json from before #486: no dedup key, no
    stream_offset. `--dedup` must refuse them."""
    folder = tmp_path / "pretrain" / "code"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"file_idx": 32, "total_tokens": 4_000_000_000,
                                                    "items_processed": 1_234_567}))
    with pytest.raises(SystemExit, match="dedup=None"):
        prefill_into([])
    assert prefill.load_progress(str(folder), "code") == (32, 4_000_000_000, 1_234_567, 1_234_567)
