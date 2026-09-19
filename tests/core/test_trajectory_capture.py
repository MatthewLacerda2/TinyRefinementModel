"""The production model can hand back every state it passed through (#225).

Depth inertness was found at the *end* of a 10-day run by eyeballing eight
prompts, because `return_all_states` lived only on the toy `CausalRefiner` and
was entangled there with `return_all_iters` — which drags a
`[depth, b, s, vocab]` logit tensor along and is genuinely unaffordable at real
vocab. States alone never were: 8.8 MB at the live config.

The load-bearing test here is `test_the_final_state_is_the_one_the_model_actually_computes`.
An instrument whose trajectory ends somewhere the ordinary forward pass does not
would be measuring a parallel computation, and every conclusion drawn from it —
#227's readout probe, #228's visualiser — would inherit that gap silently. It is
bit-identical, not merely close, so the test asserts equality rather than a
tolerance.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_PAD = TOY_VOCAB - 1
TOY_DEPTH = 4
PROMPT = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def toy_refiner():
    from trm.model.refiner_lm import RefinerForTraining

    return RefinerForTraining(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        encoder_layers=1, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD,
    )


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


def test_the_trajectory_has_one_state_per_pass_plus_the_origin(toy_refiner, tokens):
    states, _ = toy_refiner.capture_trajectory(tokens, depth=TOY_DEPTH)
    assert states.shape == (TOY_DEPTH + 1, 1, MAX_SEQ_LEN, TOY_DIM), (
        "depth+1 states: the encoder output plus one per refine pass")


def test_the_final_state_is_the_one_the_model_actually_computes(toy_refiner, tokens):
    """The whole instrument rests on this. Bit-identical, not close: same ops in
    the same order, so anything less would mean the capture path diverged."""
    states, _ = toy_refiner.capture_trajectory(tokens, depth=TOY_DEPTH)
    hidden = toy_refiner(tokens, depth=TOY_DEPTH, training=True).hidden

    assert jnp.array_equal(states[-1], hidden), (
        "the trajectory ends somewhere the ordinary forward pass does not — the "
        "capture is measuring a different computation than the model runs")


def test_the_first_state_is_the_encoder_output_before_any_refinement(toy_refiner, tokens):
    """z_0 is what no other return path exposes. A depth-0 forward runs the
    encoder and skips the loop entirely, so it is the independent witness."""
    states, _ = toy_refiner.capture_trajectory(tokens, depth=TOY_DEPTH)
    encoder_only = toy_refiner(tokens, depth=0, training=True).hidden

    assert jnp.array_equal(states[0], encoder_only)


def test_refinement_actually_moves_the_state(toy_refiner, tokens):
    """A capture that returned depth+1 copies of one array would satisfy every
    shape assertion above and be worthless."""
    states, _ = toy_refiner.capture_trajectory(tokens, depth=TOY_DEPTH)
    for k in range(1, TOY_DEPTH + 1):
        assert not jnp.array_equal(states[k], states[k - 1]), f"pass {k} was a no-op"


def test_gate_openness_comes_back_one_per_pass(toy_refiner, tokens):
    """Already computed inside the loop and previously thrown away."""
    _, gates = toy_refiner.capture_trajectory(tokens, depth=TOY_DEPTH)
    assert gates is not None and gates.shape == (TOY_DEPTH,)


@pytest.mark.parametrize("depth", [1, 2, 8])
def test_depth_is_honoured(toy_refiner, tokens, depth):
    states, _ = toy_refiner.capture_trajectory(tokens, depth=depth)
    assert states.shape[0] == depth + 1


def test_capture_does_not_drag_the_per_pass_logits_along(toy_refiner, tokens):
    """The coupling is the reason this was unaffordable. `return_all_iters`
    returns [depth, b, s, VOCAB]; the trajectory must return [depth+1, b, s, DIM]
    and never build the logit tensor."""
    states, _ = toy_refiner.refiner(
        tokens, depth=TOY_DEPTH, pad_mask=tokens != TOY_PAD, return_trajectory=True)
    logits_all, _ = toy_refiner.refiner(
        tokens, depth=TOY_DEPTH, pad_mask=tokens != TOY_PAD, return_all_iters=True)

    assert states.shape[-1] == TOY_DIM
    assert logits_all.shape[-1] == TOY_VOCAB
    assert states.shape[0] == logits_all.shape[0] + 1, (
        "the trajectory carries the origin state that the per-pass logits path does not")


def test_an_architecture_without_a_refine_loop_refuses():
    """The control baseline has no trajectory. Returning a flat one would invite
    exactly the wrong conclusion — that depth does nothing — from an instrument
    pointed at a model that has no depth to begin with."""
    from trm.model.reasoner import UniversalReasoner

    model = UniversalReasoner(60, nnx.Rngs(0), num_blocks=1)
    toks = jnp.zeros((1, MAX_SEQ_LEN), dtype=jnp.int32)

    with pytest.raises(NotImplementedError, match="no trajectory to capture"):
        model.capture_trajectory(toks, depth=2)


# ── the plain model walks its blocks (#391) ──────────────────────────────────

TOY_LAYERS = 3


@pytest.fixture(scope="module")
def toy_plain():
    from trm.model.plain import PlainTransformer

    return PlainTransformer(TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
                            num_layers=TOY_LAYERS, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)


def test_the_plain_trajectory_has_one_state_per_block_plus_the_embedding(toy_plain, tokens):
    states, gates = toy_plain.capture_trajectory(tokens)

    assert states.shape == (TOY_LAYERS + 1, 1, MAX_SEQ_LEN, TOY_DIM)
    assert states.dtype == jnp.float32
    assert gates is None, "the plain stack has no gate"
    assert jnp.array_equal(states[0], toy_plain.embed(tokens).astype(jnp.float32))


def test_the_plain_trajectory_ends_where_the_forward_pass_does(toy_plain, tokens):
    """The same load-bearing property as the refiner's: the last captured state,
    through the out-norm, is bit-for-bit the hidden state the training forward
    returns. Anything less and the instrument measures a parallel computation."""
    states, _ = toy_plain.capture_trajectory(tokens)
    hidden = toy_plain(tokens, training=True).hidden

    last = states[-1].astype(hidden.dtype)
    assert jnp.array_equal(toy_plain.out_norm(last), hidden)
