"""The production model can hand back every state it passed through (#225, #391).

The load-bearing test here is `test_the_trajectory_ends_where_the_forward_pass_does`.
An instrument whose trajectory ends somewhere the ordinary forward pass does not
would be measuring a parallel computation, and every conclusion drawn from it —
#227's readout probe, #228's visualiser — would inherit that gap silently. It is
bit-identical, not merely close, so the test asserts equality rather than a
tolerance.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.settings import CONFIG

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_PAD = TOY_VOCAB - 1
TOY_LAYERS = 3
PROMPT = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, CONFIG.MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


@pytest.fixture(scope="module")
def toy_plain():
    from trm.model.plain import PlainTransformer

    return PlainTransformer(TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
                            num_layers=TOY_LAYERS, max_seq_len=CONFIG.MAX_SEQ_LEN, pad_token_id=TOY_PAD)


def test_the_plain_trajectory_has_one_state_per_block_plus_the_embedding(toy_plain, tokens):
    states = toy_plain.capture_trajectory(tokens)

    assert states.shape == (TOY_LAYERS + 1, 1, CONFIG.MAX_SEQ_LEN, TOY_DIM)
    assert states.dtype == jnp.float32
    assert jnp.array_equal(states[0], toy_plain.embed(tokens).astype(jnp.float32))


def test_the_trajectory_ends_where_the_forward_pass_does(toy_plain, tokens):
    """The last captured state, through the out-norm, is bit-for-bit the hidden state
    the training forward returns. Anything less and the instrument measures a
    parallel computation."""
    states = toy_plain.capture_trajectory(tokens)
    hidden = toy_plain(tokens, training=True).hidden

    last = states[-1].astype(hidden.dtype)
    assert jnp.array_equal(toy_plain.out_norm(last), hidden)


def test_the_per_block_readings_are_the_trajectorys_own(toy_plain, tokens):
    """#392: act_max per state and residual RMS per state, read off the forward pass,
    match the captured states one by one, and the scalar every reader since #235
    uses is still exactly their max."""
    import numpy as np

    diag = toy_plain(tokens, training=True).diag
    states = toy_plain.capture_trajectory(tokens)
    states = np.asarray(states)

    assert diag["act_max_blocks"].shape == (TOY_LAYERS + 1,)
    assert float(diag["act_max"]) == float(jnp.max(diag["act_max_blocks"]))
    for k in range(TOY_LAYERS + 1):
        assert np.isclose(float(diag["act_max_blocks"][k]), np.abs(states[k]).max())
        assert np.isclose(float(diag["act_rms_blocks"][k]), np.sqrt(np.mean(np.square(states[k]))), rtol=1e-5)
