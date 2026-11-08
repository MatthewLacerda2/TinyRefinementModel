"""The training forward is the program main traced before the KV cache (#153).

The accordion line (#620) compares new runs against base run 002, so while it runs
the training forward must compute exactly what 002's did. A numeric pin cannot say
that across machines (the golden run tolerates ~1e-7 of CPU noise, tests/README.md),
but the traced program can: two forwards with the same jaxpr run the same ops in
the same order on the same shapes and dtypes, so they give the same bits on any one
machine.

The pin is the sha256 of the training forward's jaxpr at toy size, f32 throughout so
the f16 GPU suite traces the same program. It was read off `main` at 1ee0232, before
#153 added the decode path beside the forward (the PR records the CI run). A change
that moves it changes what training computes: if that is the point of the change,
re-pin it in that PR and say so; while #620 runs, it is not.
"""

import hashlib
import re

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from trm.model.plain import PlainTransformer
from trm.settings import CONFIG

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_LAYERS = 2
TOY_SEQ_LEN = 16
TOY_PAD = TOY_VOCAB - 1

# post_norm -> sha256 of the training forward's jaxpr, read off main at 1ee0232.
PINNED = {
    False: "unpinned",
    True: "unpinned",
}


def _training_jaxpr(post_norm):
    model = PlainTransformer(
        TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=TOY_SEQ_LEN, pad_token_id=TOY_PAD,
        dtype=jnp.float32, post_norm=post_norm, residual_dtype="float32")
    graphdef, state = nnx.split(model)

    def forward(state, tokens):
        return nnx.merge(graphdef, state)(tokens, training=True)

    tokens = jnp.zeros((1, TOY_SEQ_LEN), jnp.int32)
    text = str(jax.make_jaxpr(forward)(state, tokens))
    # A callable's repr carries its address, which moves from process to process.
    return re.sub(r" at 0x[0-9a-f]+", "", text)


@pytest.mark.parametrize("post_norm", [False, True])
def test_the_training_forward_traces_the_program_main_did(post_norm):
    digest = hashlib.sha256(_training_jaxpr(post_norm).encode()).hexdigest()
    assert digest == PINNED[post_norm], (
        f"post_norm={post_norm}: the training forward's jaxpr is {digest}, not the pinned "
        f"{PINNED[post_norm]}. It no longer runs the program base run 002 trained with "
        f"(#620). Print _training_jaxpr({post_norm}) here and on main to see what moved.")


def test_the_digest_is_stable_within_a_process():
    """A digest that moves between two traces of one model pins nothing."""
    assert _training_jaxpr(False) == _training_jaxpr(False)
