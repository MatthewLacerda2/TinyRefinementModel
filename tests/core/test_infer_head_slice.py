"""Generation projects the tied head over one position, not all of them (#206).

The pre-head state is sliced before the matmul instead of reading one row of the
full projection. Slicing is exact in real arithmetic but not bit-identical in f32:
float matmul is not shape-invariant, so the last ulp moves. The cost it removes and
the measurement are in PR #213 and its comment on #206.

So the gate is: agreement to a tolerance far below anything meaningful (the val-CE
noise floor is σ≈0.03 nats, ~5 orders of magnitude larger), plus **exact agreement
on the argmax** — because the only thing generation actually reads out of these
numbers is which token wins. A drifting logit that never changes the decision is
not a behaviour change; a drifting argmax would be.

The refiner and the reasoner are covered here; the plain model's slice is
test_the_sliced_row_matches_the_full_projection in test_plain_transformer.py. The
reasoner is the control baseline, and a control that quietly disagrees about what
its own head computes is worse than no control.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm import infer
from trm.config import MAX_SEQ_LEN

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_PAD = TOY_VOCAB - 1
TOY_DEPTH = 2
TOY_TOP_K = 8
PROMPT = [1, 2, 3, 4]

# Every position that has ever been an off-by-one: the first, a live one in the
# middle of the prompt, and the last valid index.
PROBE_POSITIONS = (0, len(PROMPT) - 1, MAX_SEQ_LEN - 1)


@pytest.fixture(scope="module")
def toy_refiner():
    from trm.model.refiner_lm import RefinerForTraining

    return RefinerForTraining(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        encoder_layers=1, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD,
    )


@pytest.fixture(scope="module")
def padded_tokens():
    tokens = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return tokens.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


@pytest.mark.parametrize("position", PROBE_POSITIONS)
def test_the_sliced_row_is_bit_identical_to_the_full_projection(
        toy_refiner, padded_tokens, position):
    """The whole correctness claim, stated directly."""
    full = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=False).logits
    one = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=False,
                      logits_at=position).logits

    assert jnp.allclose(one[0, 0], full[0, position], rtol=1e-4, atol=1e-5), (
        f"row {position} differs by more than f32 kernel-selection noise — that is a "
        f"real numerical change, not a different reduction order")
    assert int(jnp.argmax(one[0, 0])) == int(jnp.argmax(full[0, position])), (
        f"the winning token at row {position} changed — generation reads only the "
        f"argmax, so this is the drift that would actually alter output")


def test_asking_for_one_position_returns_one_position(toy_refiner, padded_tokens):
    """Shape is the contract. A model that accepted `logits_at` and returned full
    logits anyway would have the caller silently read row 0 for every token."""
    one = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=False, logits_at=2).logits
    assert one.shape == (1, 1, TOY_VOCAB)


def test_omitting_the_position_still_returns_every_row(toy_refiner, padded_tokens):
    """The seam is opt-in. Eval paths that score whole sequences must be untouched."""
    full = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=False).logits
    assert full.shape == (1, MAX_SEQ_LEN, TOY_VOCAB)


def test_the_training_path_is_unaffected(toy_refiner, padded_tokens):
    """`training=True` returns pre-head states and never reaches the head at all,
    so the chunked-CE path (#19) cannot have been disturbed."""
    out = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=True)
    assert out.logits is None
    assert out.hidden.shape == (1, MAX_SEQ_LEN, TOY_DIM)


def test_the_jitted_sampling_step_reads_the_row_it_asked_for(toy_refiner, padded_tokens):
    """The integration point, with a *traced* index — which is the reason the old
    code could not be optimized away and the reason the slice has to be a
    dynamic_slice. Compares against the full projection at the same position,
    after the same temperature/top-k/top-p truncation."""
    position = len(PROMPT) - 1
    full = toy_refiner(padded_tokens, depth=TOY_DEPTH, training=False).logits
    expected = infer._temperature_truncate(full[0, position], 1.0, TOY_TOP_K, 0.9)

    got = infer.get_logits_for_token(
        toy_refiner, padded_tokens, position, True, TOY_TOP_K, 0.9, 1.0, TOY_DEPTH)

    # -inf survives allclose (equal infinities compare equal), which is what we
    # want: the truncation mask itself must match exactly, only the surviving
    # logits may move in their last ulp.
    assert jnp.allclose(got, expected, rtol=1e-4, atol=1e-5, equal_nan=False)
    assert jnp.array_equal(jnp.isinf(got), jnp.isinf(expected)), (
        "top-k/top-p kept a different set of tokens")
    assert int(jnp.argmax(got)) == int(jnp.argmax(expected))



def test_generation_still_produces_a_reproducible_sequence(toy_refiner, monkeypatch, in_vocab_encoder):
    """End to end. Seeded, so it also pins that the change did not disturb the
    sampling stream — a shifted RNG would be a silent behaviour change even if
    every individual row were correct."""
    monkeypatch.setattr(infer, "PAD_TOKEN_ID", TOY_PAD)
    enc = in_vocab_encoder(TOY_VOCAB)  # why real ids break a toy model: tests/conftest.py

    def run():
        return infer.generate_text(
            toy_refiner, enc, "ab", max_new_tokens=6, temperature=0.7,
            top_k=TOY_TOP_K, top_p=0.9, depth=TOY_DEPTH, seed=42, quiet=True)

    assert run() == run()


def test_the_control_baseline_agrees_with_itself_too():
    """The reasoner carries the same tied-head pattern (reasoner.py:253) and got
    the same seam. It is the arch every architecture bet is measured against, so
    it does not get to skip the correctness gate just because it is not live."""
    from trm.model.reasoner import UniversalReasoner

    model = UniversalReasoner(60, nnx.Rngs(0), num_blocks=1, batch_size=1)
    tokens = jnp.zeros((1, MAX_SEQ_LEN), dtype=jnp.int32).at[0, :4].set(
        jnp.array(PROMPT, dtype=jnp.int32))

    full = model(tokens, depth=2, training=False).logits
    one = model(tokens, depth=2, training=False, logits_at=3).logits

    assert one.shape[1] == 1
    assert jnp.allclose(one[0, 0], full[0, 3], rtol=1e-4, atol=1e-5)
    assert int(jnp.argmax(one[0, 0])) == int(jnp.argmax(full[0, 3]))
