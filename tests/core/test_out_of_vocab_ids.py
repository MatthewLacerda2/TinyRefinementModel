"""One out-of-vocabulary token id used to destroy the whole window (#233).

`nnx.Embed` lowers to `jnp.take`, whose default `mode="fill"` returns **NaN** for an
out-of-range index — direct indexing clamps instead, which is why a quick probe
suggests the opposite. And one NaN reaches every position, because attention masking
is additive: `NaN + (-1e9)` is `NaN`, so a single poisoned key makes every query's
softmax row all-NaN, including causally earlier ones.

Measured before the fix: one bad id turned **18,944 of 18,944** logits non-finite.

This is not hypothetical plumbing. It is what let two determinism tests compare NaN
to NaN and pass for months — they fed a real `r50k_base` tokenizer to a 37-token toy
vocab, so every id was out of range, and `run() == run()` held because NaN is
deterministic.

Two halves, and both are needed:

- the model **clamps**, because it cannot raise from inside jit. A wrong token at one
  position beats a destroyed window.
- the data loader **raises**, because it runs in numpy outside any trace and is the
  one place the real cause can be named.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from trm.model.plain import PlainTransformer

TOY_DIM, TOY_VOCAB, TOY_HEADS, TOY_LAYERS = 32, 37, 4, 3
TOY_PAD = TOY_VOCAB - 1


@pytest.fixture(scope="module")
def toy():
    return PlainTransformer(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)


@pytest.fixture(scope="module")
def clean():
    return jax.random.randint(
        jax.random.PRNGKey(0), (1, MAX_SEQ_LEN), 0, TOY_PAD, dtype=jnp.int32)


@pytest.mark.parametrize("bad_id", [TOY_VOCAB, 50256, 2**30])
def test_an_out_of_range_id_does_not_produce_a_single_nan(toy, clean, bad_id):
    """The defect, stated directly. Before the clamp this was 18,944 of 18,944."""
    poisoned = clean.at[0, 7].set(bad_id)
    logits = toy(poisoned, depth=1, training=False).logits
    assert int(jnp.sum(~jnp.isfinite(logits))) == 0


def test_earlier_positions_are_untouched_by_a_later_bad_id(toy, clean):
    """The part that made this catastrophic rather than local: additive masking
    spread the NaN backwards through the softmax, so a bad id at position 7 changed
    position 0. Causality must hold even for garbage input."""
    poisoned = clean.at[0, 7].set(50256)
    a = toy(clean, depth=1, training=False).logits
    b = toy(poisoned, depth=1, training=False).logits
    assert jnp.array_equal(a[0, :7], b[0, :7])


def test_a_negative_id_is_handled_too(toy, clean):
    """`jnp.take` treats negatives as wrap-around indices, so they do not NaN — but
    they silently read the wrong row, and the clamp should make that explicit rather
    than leaving two different out-of-range behaviours."""
    logits = toy(clean.at[0, 3].set(-5), depth=1, training=False).logits
    assert int(jnp.sum(~jnp.isfinite(logits))) == 0


def test_in_range_ids_are_completely_unaffected(toy, clean):
    """The counter-test. A clamp that altered ordinary input would be a silent
    behaviour change on every token the model has ever seen."""
    before = toy(clean, depth=1, training=False).logits
    assert int(jnp.sum(~jnp.isfinite(before))) == 0
    assert jnp.array_equal(
        before, toy(clean, depth=1, training=False).logits)


def test_the_loader_refuses_a_shard_with_an_impossible_id(tmp_path):
    """The loud half. The model clamps and so cannot name the cause; the loader runs
    in numpy, outside any trace, and is the only place that can."""
    from trm.config import VOCAB_SIZE
    from trm.data.loaders import TextDataGenerator

    shard = np.full(4 * MAX_SEQ_LEN + 8, 5, dtype=np.int32)
    shard[100] = VOCAB_SIZE + 1
    np.save(tmp_path / "chunk_0.npy", shard)

    gen = TextDataGenerator(str(tmp_path))
    with pytest.raises(ValueError, match="exceeds VOCAB_SIZE"):
        gen.get_batch(1)


def test_the_loader_accepts_an_ordinary_shard(tmp_path):
    """Counter-test: the check must not reject the corpus we actually train on."""
    from trm.config import VOCAB_SIZE
    from trm.data.loaders import TextDataGenerator

    shard = np.full(4 * MAX_SEQ_LEN + 8, VOCAB_SIZE - 1, dtype=np.int32)
    np.save(tmp_path / "chunk_0.npy", shard)

    rows, _ = TextDataGenerator(str(tmp_path)).get_batch(1)
    assert rows is not None and rows.shape[0] == 1
