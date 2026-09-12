"""The plain transformer is what remains after depth recurrence was retired.

Plan A looped ONE shared block K times. That works on sequential composition
([[plan-a-depth-recurrence-works]], unretracted) and is actively suppressed on
language — the trained gate routes to 6 of 960 channels on prose, the second refine
pass costs 5.7 nats at 0.66B and nothing at 3.99B, and bounding the activation scale
does not recover it (docs/findings/2026-09-12-...).

What these guard is that the removal was a REMOVAL: no loop, no gate, no time
signal, no depth dial, and `depth` accepted-and-ignored rather than quietly doing
something. The last one matters most — an architecture that accepted `depth` and
half-honoured it would be the worst of both.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from trm.model.plain import PlainTransformer

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_LAYERS = 3
TOY_PAD = TOY_VOCAB - 1
PROMPT = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def toy():
    return PlainTransformer(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


# --- the removal actually happened -------------------------------------------

@pytest.mark.parametrize("gone", ["gate", "time_embed", "time_signal_norm", "refine_block"])
def test_the_refinement_machinery_is_gone(toy, gone):
    """Structural. Each of these existed only to serve the loop, and a leftover
    would be dead weight carried by every future run."""
    assert not hasattr(toy, gone)
    assert all(not hasattr(b, gone) for b in toy.blocks)


def test_depth_is_accepted_and_ignored(toy, tokens):
    """The contract passes `depth`; a plain transformer has one fixed amount of
    compute per token. Accepting and ignoring it keeps the training loop free of an
    arch-specific branch — but it must be ignored EXACTLY, not partially."""
    at_one = toy(tokens, depth=1, training=False).logits
    at_eight = toy(tokens, depth=8, training=False).logits
    at_none = toy(tokens, depth=None, training=False).logits

    assert jnp.array_equal(at_one, at_eight)
    assert jnp.array_equal(at_one, at_none)


def test_new_document_is_ignored_too(toy, tokens):
    """No state crosses windows here, so the flag cannot matter. The refiner carried
    a vestigial buffer for exactly this for months (#105)."""
    assert jnp.array_equal(
        toy(tokens, depth=1, new_document=True, training=False).logits,
        toy(tokens, depth=1, new_document=False, training=False).logits)


# --- the contract -------------------------------------------------------------

def test_training_returns_pre_head_states_and_no_logits(toy, tokens):
    """The chunked-CE path (#19) projects the tied head per chunk; materializing
    [b, s, vocab] here would reintroduce the f32 logit peak it exists to avoid."""
    out = toy(tokens, depth=1, training=True)
    assert out.logits is None
    assert out.hidden.shape == (1, MAX_SEQ_LEN, TOY_DIM)


def test_inference_returns_logits_for_every_position(toy, tokens):
    out = toy(tokens, depth=1, training=False)
    assert out.hidden is None
    assert out.logits.shape == (1, MAX_SEQ_LEN, TOY_VOCAB)


def test_asking_for_one_position_returns_one_position(toy, tokens):
    """The generation seam (#206). A model that accepted `logits_at` and returned
    full logits would have the caller silently read row 0 for every token."""
    out = toy(tokens, depth=1, training=False, logits_at=3)
    assert out.logits.shape == (1, 1, TOY_VOCAB)


def test_the_sliced_row_matches_the_full_projection(toy, tokens):
    full = toy(tokens, depth=1, training=False).logits
    one = toy(tokens, depth=1, training=False, logits_at=3).logits
    assert jnp.allclose(one[0, 0], full[0, 3], rtol=1e-4, atol=1e-5)
    assert int(jnp.argmax(one[0, 0])) == int(jnp.argmax(full[0, 3]))


# --- causality ----------------------------------------------------------------

def test_a_later_token_cannot_change_an_earlier_prediction(toy):
    """The property no amount of architecture simplification is allowed to lose.
    Bit-identical, not close: a causal mask either holds or it does not."""
    base = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    base = base.at[0, :8].set(jnp.arange(1, 9, dtype=jnp.int32))
    perturbed = base.at[0, 5].set(TOY_VOCAB - 2)

    a = toy(base, depth=1, training=False).logits
    b = toy(perturbed, depth=1, training=False).logits

    assert jnp.array_equal(a[0, :5], b[0, :5]), "a token at position 5 moved logits before it"
    assert not jnp.array_equal(a[0, 5:8], b[0, 5:8]), "the perturbation did nothing at all"


# --- size ---------------------------------------------------------------------

def test_layer_count_is_the_only_depth_knob(toy):
    assert len(toy.blocks) == TOY_LAYERS
    deeper = PlainTransformer(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS + 2, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)

    def count(m):
        return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(m, nnx.Param)))

    assert count(deeper) > count(toy), "more layers must mean more parameters"


def test_blocks_do_not_share_weights(toy):
    """The whole point of the retirement: distinct layers, not one block looped."""
    first = toy.blocks[0].gate_proj.kernel[...]
    second = toy.blocks[1].gate_proj.kernel[...]
    assert not jnp.array_equal(first, second)
