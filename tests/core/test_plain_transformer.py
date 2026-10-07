"""The plain transformer: what a forward pass returns, causality, and its size knob.

The generation seam (`logits_at`, #206) is guarded in test_infer_head_slice.py.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.model.plain import PlainTransformer
from trm.settings import CONFIG

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_LAYERS = 3
TOY_PAD = TOY_VOCAB - 1
PROMPT = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def toy():
    return PlainTransformer(
        TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=CONFIG.MAX_SEQ_LEN, pad_token_id=TOY_PAD)


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, CONFIG.MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


# --- the contract -------------------------------------------------------------

def test_training_returns_pre_head_states_and_no_logits(toy, tokens):
    """The chunked-CE path (#19) projects the tied head per chunk; materializing
    [b, s, vocab] here would reintroduce the f32 logit peak it exists to avoid."""
    out = toy(tokens, training=True)
    assert out.logits is None
    assert out.hidden.shape == (1, CONFIG.MAX_SEQ_LEN, TOY_DIM)


def test_inference_returns_logits_for_every_position(toy, tokens):
    out = toy(tokens, training=False)
    assert out.hidden is None
    assert out.logits.shape == (1, CONFIG.MAX_SEQ_LEN, TOY_VOCAB)


# --- causality ----------------------------------------------------------------

def test_a_later_token_cannot_change_an_earlier_prediction(toy):
    """The property no amount of architecture simplification is allowed to lose.
    Bit-identical, not close: a causal mask either holds or it does not."""
    base = jnp.full((1, CONFIG.MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    base = base.at[0, :8].set(jnp.arange(1, 9, dtype=jnp.int32))
    perturbed = base.at[0, 5].set(TOY_VOCAB - 2)

    a = toy(base, training=False).logits
    b = toy(perturbed, training=False).logits

    assert jnp.array_equal(a[0, :5], b[0, :5]), "a token at position 5 moved logits before it"
    assert not jnp.array_equal(a[0, 5:8], b[0, 5:8]), "the perturbation did nothing at all"


# --- size ---------------------------------------------------------------------

def test_layer_count_is_the_only_depth_knob(toy, n_params):
    assert len(toy.blocks) == TOY_LAYERS
    deeper = PlainTransformer(
        TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS + 2, max_seq_len=CONFIG.MAX_SEQ_LEN, pad_token_id=TOY_PAD)

    assert n_params(deeper) > n_params(toy), "more layers must mean more parameters"


def test_blocks_do_not_share_weights(toy):
    """The whole point of the retirement: distinct layers, not one block looped."""
    first = toy.blocks[0].gate_proj.kernel[...]
    second = toy.blocks[1].gate_proj.kernel[...]
    assert not jnp.array_equal(first, second)
