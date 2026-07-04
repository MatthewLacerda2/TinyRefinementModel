"""The yardstick eval (#48) must be trustworthy before any verdict read off it is.

Covers the three ways an eval silently lies: wrong example prep (split /
truncation), wrong metric math (the paper-number comparison inherits any slip),
and batching artifacts (padding or bucket membership changing a score). Plus the
real-model path: the tiny refiner through the exact adapter the runner uses.
All offline — a fake whitespace tokenizer stands in for tiktoken.
"""

import hashlib

import numpy as np
import pytest

from tools.yardstick import (
    encode_example,
    score_examples,
    split_last_word,
    summarize,
    verify_sha256,
)

VOCAB = 11
PAD = 0


class FakeEnc:
    """Whitespace tokenizer over a tiny closed vocabulary (ids 1..9)."""

    def encode(self, text):
        return [int(w) for w in text.split()]


def rule_logits_fn(tokens):
    """Causal-by-construction fake model: position i predicts (7 * token_i) % VOCAB.

    Logits at i depend only on token i, so padding and batch composition
    provably cannot change any example's score — the invariant the batching
    tests lean on. A cosine bump makes the non-argmax mass smooth and
    token-dependent so log-probs are informative, not degenerate.
    """
    tokens = np.asarray(tokens)
    b, s = tokens.shape
    logits = np.cos(np.arange(VOCAB)[None, None, :] * (tokens[:, :, None] + 1)).astype(np.float32)
    winner = (7 * tokens) % VOCAB
    logits[np.arange(b)[:, None], np.arange(s)[None, :], winner] += 5.0
    return logits


def test_split_last_word():
    assert split_last_word("the quick brown fox") == ("the quick brown", " fox")
    assert split_last_word("one two\n") == ("one", " two")


def test_encode_example_truncates_context_keeps_target():
    enc = FakeEnc()
    ctx, tgt = encode_example(enc, "1 2 3 4 5 6 7 8 9", max_seq_len=4)
    assert tgt == [9]
    assert ctx == [6, 7, 8]  # most recent context wins; total fits max_seq_len


def test_encode_example_rejects_degenerate():
    enc = FakeEnc()
    assert encode_example(enc, "9", max_seq_len=8) is None  # nothing to split


def test_metric_math_by_hand():
    """One 2-token context, 1-token target, vocab 3: exact log-softmax check."""
    fixed = np.array([[[0.0, 1.0, 2.0], [3.0, 1.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    scores = score_examples(lambda t: fixed, [([1, 2], [0])],
                            pad_token_id=PAD, buckets=(3,))
    row = fixed[0, 1].astype(np.float64)  # position len(ctx)-1 predicts the target
    expected = row[0] - np.log(np.exp(row).sum())
    assert scores[0].greedy_hit  # argmax of [3,1,0] is id 0, the target
    assert scores[0].logprob == pytest.approx(expected, rel=1e-6)
    assert np.exp(-expected) == pytest.approx(summarize(scores)["lambada_ppl"], rel=1e-6)


def test_accuracy_counts_exactly_the_rule_hits():
    # Under rule_logits_fn the greedy next token after t is (7*t) % VOCAB:
    # 3->10, 2->3, 4->6 (hits); 5->2, so a target of 8 is the engineered miss.
    encoded = [([3], [10]), ([2], [3]), ([4], [6]), ([5], [8])]
    scores = score_examples(rule_logits_fn, encoded, pad_token_id=PAD, buckets=(4,))
    assert [s.greedy_hit for s in scores] == [True, True, True, False]
    assert summarize(scores)["lambada_acc"] == pytest.approx(0.75)


def test_multi_token_target_needs_every_token():
    hit = ([2], [3, 10])   # 7*2=14%11=3, then 7*3=21%11=10 — both argmaxes
    miss = ([2], [3, 9])   # second token off — one wrong token sinks the word
    scores = score_examples(rule_logits_fn, [hit, miss], pad_token_id=PAD, buckets=(8,))
    assert scores[0].greedy_hit and not scores[1].greedy_hit


def test_batching_and_buckets_change_nothing():
    """Same example alone vs jammed in batches with longer neighbors: identical
    score. This is the causal-padding assumption the whole batcher rests on."""
    probe = ([1, 2, 3], [10])
    neighbors = [([i % 9 + 1] * 30, [(7 * (i % 9 + 1)) % VOCAB]) for i in range(7)]
    alone = score_examples(rule_logits_fn, [probe], pad_token_id=PAD, buckets=(4,))
    crowded = score_examples(rule_logits_fn, [probe] + neighbors, pad_token_id=PAD,
                             batch_size=3, buckets=(8, 32))
    assert crowded[0].greedy_hit == alone[0].greedy_hit
    assert crowded[0].logprob == pytest.approx(alone[0].logprob, rel=1e-6)


def test_sha256_gate(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "a b"}\n')
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    verify_sha256(str(p), good)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_sha256(str(p), "0" * 64)


def test_tiny_refiner_through_the_runner_adapter():
    """The real path at toy scale: RefinerForTraining -> make_logits_fn ->
    score_examples. Finite, in-range, and deterministic across calls."""
    from flax import nnx

    from plan_a_trainer import RefinerForTraining
    from tools.eval_yardstick import make_logits_fn

    pad = 63
    model = RefinerForTraining(
        32, nnx.Rngs(0), vocab_size=64, num_heads=2, encoder_layers=1,
        max_depth=2, max_seq_len=64, pad_token_id=pad,
    )
    rng = np.random.default_rng(3)
    encoded = [(list(rng.integers(1, 62, size=n)), list(rng.integers(1, 62, size=2)))
               for n in (5, 11, 20)]
    logits_fn = make_logits_fn(model, depth=2)
    runs = [summarize(score_examples(logits_fn, encoded, pad_token_id=pad,
                                     batch_size=2, buckets=(16, 32)))
            for _ in range(2)]
    assert runs[0] == runs[1]  # same model, same examples, same numbers
    assert 0.0 <= runs[0]["lambada_acc"] <= 1.0
    assert np.isfinite(runs[0]["lambada_ppl"]) and runs[0]["lambada_ppl"] > 1.0
    assert runs[0]["num_examples"] == 3
