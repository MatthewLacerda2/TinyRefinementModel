"""The FineWeb val yardstick (#462) compares us with a published number, so it has
to read the shard the way the reference does: the pinned file, its header, every
target once, and a CE that is the plain mean over those targets."""

import math

import numpy as np
import pytest

from instruments.yardstick import fineweb_val


def write_shard(path, tokens):
    header = np.zeros(fineweb_val.HEADER_INT32S, dtype=np.int32)
    header[:3] = fineweb_val.MAGIC, 1, len(tokens)
    with open(path, "wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def test_the_header_is_read_and_refused_when_it_lies(tmp_path):
    write_shard(tmp_path / "s.bin", [fineweb_val.EOT, 5, 6, 7])
    assert fineweb_val.read_tokens(tmp_path / "s.bin", 3).tolist() == [fineweb_val.EOT, 5, 6]
    with pytest.raises(ValueError, match="fewer"):
        fineweb_val.read_tokens(tmp_path / "s.bin", 5)
    (tmp_path / "bad.bin").write_bytes(b"\0" * 2048)
    with pytest.raises(ValueError, match="magic"):
        fineweb_val.read_tokens(tmp_path / "bad.bin", 1)


def test_windows_score_every_target_exactly_once():
    tokens = np.arange(23)
    inputs, targets = fineweb_val.windows(tokens, 5)
    assert inputs.shape == targets.shape == (4, 5)
    assert targets.ravel().tolist() == list(range(1, 21))
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])  # each window predicts its own next tokens
    with pytest.raises(ValueError, match="no 30-token window"):
        fineweb_val.windows(tokens, 30)


def test_score_is_the_plain_mean_ce():
    """A model whose log-prob of token k is -k - log Z: the mean CE is the mean target
    id plus log Z, over every target, none excluded."""
    vocab = 8

    def logits_fn(inputs):
        return np.tile(-np.arange(vocab, dtype=np.float64), inputs.shape + (1,)) - np.log(
            np.exp(-np.arange(vocab)).sum())

    tokens = np.array([0, 3, 1, 2, 5, 4, 7])
    out = fineweb_val.score(logits_fn, tokens, seq=3, batch=1)
    assert out["targets"] == 6
    assert math.isclose(out["val_ce"], np.mean([3, 1, 2, 5, 4, 7]) + np.log(np.exp(-np.arange(vocab)).sum()))


def test_overlap_finds_a_shared_document_and_only_that_one(tmp_path):
    e = fineweb_val.EOT
    shared = list(range(100, 130))
    val = np.array([e, *shared, e, *range(200, 230), e, 1])
    corpus = tmp_path / "chunk_0.npy"
    np.save(corpus, np.array([7, 7, e, *shared, e, *range(300, 330)], dtype=np.int32))
    out = fineweb_val.overlap(val, [corpus], width=24)
    assert out == {"documents": 2, "found_in_corpus": 1, "width": 24}


def test_the_pin_is_the_hubs_object_id():
    assert len(fineweb_val.FINEWEB_VAL_SHA256) == 64 and fineweb_val.SPEEDRUN_VAL_TOKENS == 10 * 2 ** 20
