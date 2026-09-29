"""POSITION_ENCODING=nope deletes RoPE and nothing else (#444).

The pair that judges NoPE (experiments/recipe/specs/444-nope-pair.toml) attributes its
difference to position encoding only if the knob changes exactly that. So these hold:
the default is still RoPE; "nope" applies no rotation anywhere, which leaves position
to the causal mask alone; and the two arms have the same parameters. That the default
path computes what it always did is the golden run's job (tests/expensive/).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from trm.model import refiner
from trm.model.plain import PlainTransformer
from trm.settings import CONFIG, Config

DIM, VOCAB, HEADS, LAYERS, SEQ = 32, 37, 4, 2, 16
PAD = VOCAB - 1


def _toy(position_encoding, *, dim=DIM, layers=LAYERS, seed=0):
    return PlainTransformer(dim, nnx.Rngs(seed), CONFIG, vocab_size=VOCAB, num_heads=HEADS,
                            num_layers=layers, max_seq_len=SEQ, pad_token_id=PAD,
                            position_encoding=position_encoding)


def _tokens(seed=0):
    return jax.random.randint(jax.random.PRNGKey(seed), (2, SEQ), 0, PAD, dtype=jnp.int32)


def _logits(model, tokens):
    return model(tokens, training=False).logits


def test_the_default_is_still_rope():
    """An unset environment builds what every run so far trained: RoPE in every block."""
    model = PlainTransformer(DIM, nnx.Rngs(0), Config.from_env({}),
                             vocab_size=VOCAB, num_heads=HEADS, num_layers=LAYERS, max_seq_len=SEQ,
                             pad_token_id=PAD)
    assert all(b.attn.rope and hasattr(b.attn, "cos") for b in model.blocks)


def test_nope_is_read_from_the_config_it_is_handed():
    model = PlainTransformer(DIM, nnx.Rngs(0), CONFIG.model_copy(update={"POSITION_ENCODING": "nope"}),
                             vocab_size=VOCAB, num_heads=HEADS, num_layers=LAYERS, max_seq_len=SEQ,
                             pad_token_id=PAD)
    assert not any(b.attn.rope for b in model.blocks)


def test_nope_applies_no_rotation_anywhere(monkeypatch):
    """Structural and behavioural: no block keeps a cos/sin table, and a forward pass
    never reaches apply_rope. The same trap fires under RoPE, so it is live."""
    nope, rope = _toy("nope"), _toy("rope")
    assert not any(hasattr(b.attn, "cos") or hasattr(b.attn, "sin") for b in nope.blocks)

    def trap(*_):
        raise AssertionError("apply_rope reached")

    monkeypatch.setattr(refiner, "apply_rope", trap)
    _logits(nope, _tokens())
    with pytest.raises(AssertionError, match="apply_rope reached"):
        _logits(rope, _tokens())


def test_nope_is_rope_with_the_rotation_set_to_identity():
    """The one variable: the same init, with RoPE's tables replaced by angle zero
    (cos 1, sin 0), computes exactly what NoPE computes."""
    nope, rope = _toy("nope"), _toy("rope")
    for block in rope.blocks:
        block.attn.cos = jnp.ones_like(block.attn.cos)
        block.attn.sin = jnp.zeros_like(block.attn.sin)
    np.testing.assert_array_equal(_logits(nope, _tokens()), _logits(rope, _tokens()))
    assert not np.array_equal(_logits(nope, _tokens()), _logits(_toy("rope"), _tokens())), \
        "the real tables must change the output, or the test above proves nothing"


def test_the_two_arms_have_the_same_parameters():
    """NoPE adds and removes no parameter: same tree, same shapes, same init."""
    nope, rope = (nnx.state(_toy(p), nnx.Param) for p in ("nope", "rope"))
    assert jax.tree_util.tree_structure(nope) == jax.tree_util.tree_structure(rope)
    for a, b in zip(jax.tree_util.tree_leaves(nope), jax.tree_util.tree_leaves(rope)):
        np.testing.assert_array_equal(a, b)


def test_under_nope_only_the_mask_carries_position():
    """One layer, NoPE: position t >= 2 attends to positions 0..t as a set, so swapping
    the tokens at positions 0 and 1 cannot change its output. Under RoPE it must."""
    tokens = _tokens()
    swapped = tokens.at[:, 0].set(tokens[:, 1]).at[:, 1].set(tokens[:, 0])
    nope, rope = _toy("nope", layers=1), _toy("rope", layers=1)
    np.testing.assert_allclose(_logits(nope, tokens)[:, 2:], _logits(nope, swapped)[:, 2:],
                               rtol=1e-5, atol=1e-5)
    assert not np.allclose(_logits(rope, tokens)[:, 2:], _logits(rope, swapped)[:, 2:],
                           rtol=1e-5, atol=1e-5)


def test_nope_lifts_the_even_head_dim_constraint():
    """RoPE rotates feature pairs, so it needs an even head_dim; NoPE does not."""
    odd = DIM + HEADS   # head_dim 9
    assert np.isfinite(_logits(_toy("nope", dim=odd), _tokens())).all()
    with pytest.raises(AssertionError, match="even for RoPE"):
        _toy("rope", dim=odd)


def test_a_nope_forward_and_backward_is_finite():
    model = _toy("nope")
    tokens = _tokens()
    tokens = tokens.at[1, -4:].set(PAD)   # pad keys on the path too

    def loss_fn(m):
        logits = m(tokens[:, :-1], training=False).logits
        return optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:]).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert np.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(np.isfinite(g).all() for g in leaves)
    assert any(np.abs(g).max() > 0 for g in leaves), "a gradient must reach the parameters"


@pytest.mark.parametrize("saved, restored", [("rope", "nope"), ("nope", "rope")])
def test_a_checkpoint_does_not_restore_into_the_other_arm(tmp_path, saved, restored):
    """The parameters match, but RoPE's cos/sin tables are part of the saved state
    (raw arrays are nnx state leaves), so a restore across arms is a structure
    mismatch and fails loudly — a NoPE checkpoint scored without POSITION_ENCODING=nope
    cannot silently be scored as RoPE. This is the path trm.infer and the trainer use."""
    import orbax.checkpoint as ocp

    from trm.runtime.checkpoints import restore_tolerating_legacy

    mngr = ocp.CheckpointManager(tmp_path / "checkpoints", item_names=("model",),
                                 options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))
    mngr.save(0, args=ocp.args.Composite(model=ocp.args.StandardSave(nnx.state(_toy(saved)))))
    mngr.wait_until_finished()

    def read(target):
        return mngr.restore(0, args=ocp.args.Composite(model=ocp.args.StandardRestore(target)))

    restore_tolerating_legacy(read, _toy(saved, seed=1))   # the same arm restores
    with pytest.raises(ValueError):
        restore_tolerating_legacy(read, _toy(restored, seed=1))
