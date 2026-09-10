"""Generation refuses to sample from logits that are not a distribution (#229).

A checkpoint 3.20B tokens into the base run returns **all-NaN logits on ~10% of
code documents** under the f16 compute policy, and on 0% of prose documents:

    checkpoint                corpus         NaN rows / 300
    best/3121279 (3.20B)      codeparrot     29/300   (9.67%)
    best/3121279 (3.20B)      fineweb-edu     0/300   (0.00%)
    champion 3899391 (3.99B)  codeparrot      0/300   (0.00%)

Every parameter array in that checkpoint is finite (150/150 checked) and depth 1
is enough to trigger it, so this is an overflow in the encoder or the head at
f16, not corrupt weights and not the refine loop.

The reason it needs a guard rather than a note is that **nothing failed**. Both
samplers accept a NaN row and return an ordinary token id — `jnp.argmax` yields
index 0, `jax.random.categorical` yields whatever the cumulative sum lands on.
Training has non-finite guards and skipped 14,163 steps on them during the base
run; the inference path had none at all, so it turned an overflow into fluent
nonsense with no signal anywhere.

The counter-test matters as much as the rejections: `-inf` is the *normal* state
of nearly every entry after top-k/top-p, so a guard written with `jnp.isfinite`
would reject every healthy row in the codebase.
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
WHERE = "token 3, position 11, depth 2"


def _healthy_row(size=TOY_VOCAB):
    return jnp.linspace(-4.0, 4.0, size)


# --- the three unsampleable shapes -------------------------------------------

def test_an_all_nan_row_is_rejected():
    """The measured #229 failure, stated directly."""
    with pytest.raises(infer.NonFiniteLogits):
        infer.reject_unsampleable(jnp.full((TOY_VOCAB,), jnp.nan), where=WHERE)


def test_a_single_nan_is_rejected():
    """Partial corruption is the more dangerous case: the row still has a
    plausible argmax, so it survives every downstream check we have."""
    poisoned = _healthy_row().at[7].set(jnp.nan)
    with pytest.raises(infer.NonFiniteLogits):
        infer.reject_unsampleable(poisoned, where=WHERE)


def test_positive_infinity_is_rejected():
    """f16 overflow arrives as +inf before softmax turns it into NaN. Catching
    it here names the overflow instead of the NaN one step downstream."""
    overflowed = _healthy_row().at[3].set(jnp.inf)
    with pytest.raises(infer.NonFiniteLogits):
        infer.reject_unsampleable(overflowed, where=WHERE)


def test_a_row_that_is_entirely_minus_infinity_is_rejected():
    """Not the #229 failure, but the same silent kind: truncation ate the whole
    row and there is no distribution left to sample."""
    with pytest.raises(infer.NonFiniteLogits):
        infer.reject_unsampleable(jnp.full((TOY_VOCAB,), -jnp.inf), where=WHERE)


# --- the counter-tests: healthy rows must pass -------------------------------

def test_an_ordinary_row_is_accepted():
    infer.reject_unsampleable(_healthy_row(), where=WHERE)


@pytest.mark.parametrize("top_k,top_p", [(8, 0.9), (1, 1.0), (0, 0.5), (TOY_VOCAB, 0.9)])
def test_real_truncated_logits_are_accepted(top_k, top_p):
    """The guard-the-guard test. After top-k/top-p almost every entry is -inf —
    with top_k=1, all but one of them. A guard built on `jnp.isfinite` would
    reject every one of these, which is to say every healthy row generation has
    ever produced."""
    truncated = infer._temperature_truncate(_healthy_row(), 0.7, top_k, top_p)
    assert jnp.isneginf(truncated).any(), "the fixture stopped exercising truncation"
    infer.reject_unsampleable(truncated, where=WHERE)


def test_the_message_says_where_and_what():
    """An error that does not name the position is a bug report nobody can act
    on — the whole point is that this failure is input-specific."""
    with pytest.raises(infer.NonFiniteLogits) as caught:
        infer.reject_unsampleable(jnp.full((TOY_VOCAB,), jnp.nan), where=WHERE)
    message = str(caught.value)
    assert WHERE in message
    assert str(TOY_VOCAB) in message
    assert "FORCE_F32_COMPUTE" in message, "the message must name the next thing to try"


# --- the guard is actually on the generation path ----------------------------

class _StubEncoder:
    """`generate_text` needs only encode/decode. A real tokenizer emits ids far
    outside the toy vocab, which JAX would silently clamp rather than reject."""

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_generation_raises_instead_of_emitting_a_token(monkeypatch):
    """Wiring, not logic: a guard that exists but sits off the path is why #229
    went unnoticed for a whole run."""
    monkeypatch.setattr(
        infer, "get_logits_for_token",
        lambda *a, **k: jnp.full((TOY_VOCAB,), jnp.nan))

    with pytest.raises(infer.NonFiniteLogits):
        infer.generate_text(object(), _StubEncoder(), "hello",
                            max_new_tokens=4, quiet=True)


def test_a_healthy_model_still_generates(monkeypatch):
    """The end-to-end counter-test: a real forward pass, real truncation, real
    sampling, and the guard stays out of the way.

    `generate_text` pads with the *config* PAD_TOKEN_ID (50256), which is outside
    the toy vocab — and one out-of-range id makes this model return all-NaN
    logits for the whole window (#233). That is a real defect, but it is not the
    one under test here, so the pad id is aligned with the toy model instead of
    letting an unrelated bug decide whether this test passes.
    """
    from trm.model.refiner_lm import RefinerForTraining

    monkeypatch.setattr(infer, "PAD_TOKEN_ID", TOY_PAD)

    calls = []
    real_guard = infer.reject_unsampleable
    monkeypatch.setattr(infer, "reject_unsampleable",
                        lambda logits, **kw: (calls.append(1),
                                              real_guard(logits, **kw))[1])

    model = RefinerForTraining(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        encoder_layers=1, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD,
    )
    infer.generate_text(model, _StubEncoder(), "hello", max_new_tokens=3,
                        top_k=8, depth=TOY_DEPTH, seed=0, quiet=True)

    assert calls, "the guard never ran, so this proved nothing"
