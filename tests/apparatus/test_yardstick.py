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

from instruments.yardstick.yardstick import (
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

    from trm.model.refiner_lm import RefinerForTraining
    from instruments.yardstick.eval_yardstick import make_logits_fn

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


TINY_PLAIN = dict(dim=32, num_heads=2, num_layers=1, max_seq_len=64)


@pytest.mark.parametrize("arch_flag", [[], ["--arch", "plain"]], ids=["default-arch", "explicit-plain"])
def test_the_runner_restores_and_scores_a_plain_checkpoint(tmp_path, monkeypatch, arch_flag):
    """#313: `plain` is the default architecture, and the runner could not restore it —
    `--arch` offered only reasoner/refiner and the restore map raised KeyError on the
    default. So the base run's milestone and completion scoring died on the live arch.

    Drives main() end to end on a tiny plain checkpoint: the real arch flag, the real
    restore (only the skeleton's size is shrunk), the real scorer and model-card row.
    LAMBADA itself is a two-line local file and the tokenizer a fake, so it runs offline."""
    import json
    import types

    import jax
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx

    from instruments.arch import build
    from instruments.yardstick import eval_yardstick
    from trm.config import MODEL_ARCH
    from trm.runtime import checkpoints as ck
    from trm.runtime.monitor import LossMonitor
    from trm.runtime.restore import restore_arch

    if not arch_flag and MODEL_ARCH != "plain":
        pytest.skip(f"the default-arch case needs MODEL_ARCH=plain, this process has {MODEL_ARCH}")

    saved = build("plain", **TINY_PLAIN)
    mngr = ocp.CheckpointManager(str(tmp_path / "checkpoints"), item_names=ck.CHECKPOINT_ITEMS,
                                 options=ocp.CheckpointManagerOptions(create=True))
    ck.save_checkpoint(mngr, 7, saved, nnx.Optimizer(saved, optax.adam(1e-3), wrt=nnx.Param),
                       LossMonitor(), False, "run_tiny")

    restored = {}

    def tiny_restore(arch, checkpoint_path, step=None):
        restored["arch"] = arch
        restored["model"], step = restore_arch(arch, checkpoint_path, step=step, **TINY_PLAIN)
        return restored["model"], step

    monkeypatch.setattr(eval_yardstick, "restore_arch", tiny_restore)
    monkeypatch.setattr(eval_yardstick, "tiktoken", types.SimpleNamespace(get_encoding=lambda name: FakeEnc()))
    data = tmp_path / "lambada.jsonl"
    data.write_text('{"text": "1 2 3 4"}\n{"text": "5 6 7 8 9"}\n')
    out = tmp_path / "row.json"

    eval_yardstick.main(["--checkpoint-path", str(tmp_path / "checkpoints"), "--data-path", str(data),
                         "--limit", "2", "--batch", "2", "--no-heldout", "--json-out", str(out), *arch_flag])

    assert restored["arch"] == "plain"
    saved_leaves = jax.tree_util.tree_leaves(nnx.state(saved))
    restored_leaves = jax.tree_util.tree_leaves(nnx.state(restored["model"]))
    assert len(saved_leaves) == len(restored_leaves), "a restore that drops leaves must not pass"
    assert all(np.array_equal(a, b) for a, b in zip(saved_leaves, restored_leaves)), \
        "the checkpoint's weights, not the skeleton's own initialization"
    row = json.loads(out.read_text())
    assert row["arch"] == "plain" and row["checkpoint"]["step"] == 7
    assert row["lambada"]["num_examples"] == 2 and 0.0 <= row["lambada"]["lambada_acc"] <= 1.0


def test_a_named_step_restores_that_step_not_the_newest(tmp_path):
    """#328: a milestones dir holds every milestone, and restore used to take the newest
    step of whatever dir it was given. Scoring milestone M must load exactly M."""
    import jax
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx

    from instruments.arch import build
    from trm.runtime import checkpoints as ck
    from trm.runtime.monitor import LossMonitor
    from trm.runtime.restore import restore_arch

    mngr = ocp.CheckpointManager(str(tmp_path), item_names=ck.CHECKPOINT_ITEMS,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True))
    older, newer = build("plain", seed=1, **TINY_PLAIN), build("plain", seed=2, **TINY_PLAIN)
    for step, model in ((3, older), (9, newer)):
        ck.save_checkpoint(mngr, step, model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param),
                           LossMonitor(), False, "run_tiny")

    def leaves(model):
        return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))

    at_3, step = restore_arch("plain", str(tmp_path), step=3, **TINY_PLAIN)
    assert len(leaves(older)) == len(leaves(at_3))
    assert step == 3 and all(np.array_equal(a, b) for a, b in zip(leaves(older), leaves(at_3)))
    assert not all(np.array_equal(a, b) for a, b in zip(leaves(newer), leaves(at_3))), \
        "the two saved models must differ, or this test cannot tell the steps apart"
    _, default_step = restore_arch("plain", str(tmp_path), **TINY_PLAIN)
    assert default_step == 9, "with no step named, the newest stays the default"
    with pytest.raises(SystemExit, match="step 5"):
        restore_arch("plain", str(tmp_path), step=5, **TINY_PLAIN)
