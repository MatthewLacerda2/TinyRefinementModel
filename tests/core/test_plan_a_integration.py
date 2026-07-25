"""Plan A wired into the production trainer (plan_a_trainer.RefinerForTraining).

test_plan_a.py proves the pure CausalRefiner; this proves the *integration* — that
the adapter drives the real grad step, that it speaks the trainer's neutral contract
without carrying any of the reasoner's machinery (#105), and that the refiner's param
tree survives the Orbax round-trip the trainer uses. These are the failure modes a
config flag flip would otherwise hit only mid-run on the GPU.

Runs on CPU in f32 like the rest of the suite. Windows are sliced at MAX_SEQ_LEN, so
the batch must be 2*MAX_SEQ_LEN+1 wide; dims are kept tiny otherwise.
"""

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from checkpoint_utils import restore_tolerating_legacy
from config import MAX_SEQ_LEN
from grad_step import compute_grad_step, apply_grads
from plan_a_trainer import RefinerForTraining

DIM = 64
VOCAB = 37


def _tiny_adapter(seed=0):
    return RefinerForTraining(
        DIM, nnx.Rngs(seed), vocab_size=VOCAB, num_heads=4,
        encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN,
    )


def _fixed_batch(seed=3):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, VOCAB, size=(1, 2 * MAX_SEQ_LEN + 1))
    return jnp.asarray(tokens.astype(np.int32))


def test_adapter_speaks_the_neutral_contract():
    """The adapter returns the LMOutput the trainer consumes — and nothing else.

    #105 turned the contract around: the refiner no longer impersonates the
    reasoner. It reports no auxiliary objectives (an empty dict, not zeros for a
    schedule to multiply) and allocates no state buffer for the trainer to write
    into. Both are asserted here because both used to exist purely as costume.
    """
    m = _tiny_adapter()
    tokens = _fixed_batch()[:, :MAX_SEQ_LEN]
    out = m(tokens, depth=4, training=True, new_document=True)

    # Training returns pre-head states (logits=None); the loss does the chunked
    # LM-head projection (#19). Inference still returns full logits (checked below).
    assert out.logits is None
    assert out.hidden.shape == (1, MAX_SEQ_LEN, DIM)
    assert np.isfinite(np.asarray(out.hidden)).all()
    infer_out = m(tokens, depth=4, training=False, new_document=True)
    assert infer_out.logits.shape == (1, MAX_SEQ_LEN, VOCAB)
    assert np.isfinite(np.asarray(infer_out.logits)).all()

    assert out.aux == {}, "the refiner has no auxiliary objectives to grade"
    assert m.grade_aux([out.aux, out.aux], 1000) == {}
    assert not hasattr(m, "hunch_cache"), "no vestigial state buffer"
    # The contract's state hooks are inert here, and must stay callable anyway —
    # the shared loop calls them without asking which architecture it has.
    m.reset_state()
    m.end_step(True)
    with m.isolated_state():
        pass


def test_stateless_forward_ignores_document_boundaries():
    """Plan A carries nothing between windows, so `new_document` cannot change a
    prediction. If this ever fails, the refiner grew cross-window state and the
    trainer's window sequencing has to be revisited."""
    m = _tiny_adapter()
    tokens = _fixed_batch()[:, :MAX_SEQ_LEN]
    fresh = np.asarray(m(tokens, depth=3, training=False, new_document=True).logits)
    continued = np.asarray(m(tokens, depth=3, training=False, new_document=False).logits)
    np.testing.assert_array_equal(fresh, continued)


def test_grad_step_runs_and_reduces_loss():
    """The real production grad step (compute_grad_step + apply_grads) drives the
    adapter: loss and grad norm are finite, gradients flow, and the optimizer
    reduces the loss on a fixed batch — the end-to-end integration signal."""
    m = _tiny_adapter()
    opt = nnx.Optimizer(m, optax.adamw(3e-3), wrt=nnx.Param)
    batch = _fixed_batch()

    loss0, out, grads, gnorm0 = compute_grad_step(m, batch, jnp.array(1), 2)
    assert np.isfinite(float(loss0)) and np.isfinite(float(gnorm0))
    assert float(gnorm0) > 0.0, "no gradient flowed through the refiner"
    assert out.logits is None  # grad step nulls logits in the returned aux

    last = float(loss0)
    apply_grads(opt, grads, m)
    for s in range(2, 14):
        loss, _, grads, gnorm = compute_grad_step(m, batch, jnp.array(s), 2)
        assert np.isfinite(float(loss)) and np.isfinite(float(gnorm))
        apply_grads(opt, grads, m)
        last = float(loss)

    assert last < 0.9 * float(loss0), f"loss did not decrease: {float(loss0):.3f} -> {last:.3f}"


def test_checkpoint_roundtrip_preserves_forward(tmp_path):
    """Save the refiner's state and restore it into a freshly-initialized adapter;
    forwards must match bit-for-bit. Catches state-tree drift in the new param
    layout before a real run trusts a resume."""
    m = _tiny_adapter(seed=0)
    tokens = _fixed_batch()[:, :MAX_SEQ_LEN]
    reference = np.asarray(m(tokens, depth=2, training=False, new_document=True).logits)

    mngr = ocp.CheckpointManager(
        tmp_path / "checkpoints",
        item_names=("model",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    mngr.save(0, args=ocp.args.Composite(model=ocp.args.StandardSave(nnx.state(m))))
    mngr.wait_until_finished()

    other = _tiny_adapter(seed=99)  # different init, must be overwritten by restore
    restored = mngr.restore(
        0, args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(other)))
    )
    nnx.update(other, restored["model"])

    roundtripped = np.asarray(other(tokens, depth=2, training=False, new_document=True).logits)
    np.testing.assert_array_equal(reference, roundtripped)


def test_pre_105_checkpoint_still_restores(tmp_path):
    """Every refiner checkpoint on disk — including the champion base run and the
    worlds the #134 time machine revives — was written while the adapter still
    carried the vestigial `hunch_cache`. Dropping the buffer must not orphan them:
    restore matches the legacy shape, discards it, and reproduces the forward.
    """
    m = _tiny_adapter(seed=0)
    tokens = _fixed_batch()[:, :MAX_SEQ_LEN]
    reference = np.asarray(m(tokens, depth=2, training=False).logits)

    # A pre-#105 save: the real state plus the buffer the old trainer wrote into.
    legacy_state = nnx.state(m)
    legacy_state["hunch_cache"] = nnx.Variable(jnp.zeros((1, 1, DIM)))

    mngr = ocp.CheckpointManager(
        tmp_path / "checkpoints", item_names=("model",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    mngr.save(0, args=ocp.args.Composite(model=ocp.args.StandardSave(legacy_state)))
    mngr.wait_until_finished()

    other = _tiny_adapter(seed=99)  # different init, must be overwritten by restore
    restored = restore_tolerating_legacy(
        lambda target: mngr.restore(
            0, args=ocp.args.Composite(model=ocp.args.StandardRestore(target))),
        other,
    )
    assert "hunch_cache" not in restored["model"], "legacy buffer leaked into the model"
    nnx.update(other, restored["model"])

    np.testing.assert_array_equal(
        reference, np.asarray(other(tokens, depth=2, training=False).logits))


def test_restore_still_fails_loudly_on_genuinely_missing_weights(tmp_path):
    """The legacy tolerance must not become a blanket 'load whatever fits'. A
    checkpoint that lacks weights the model needs is a real error — silently
    keeping freshly-initialized values there is how a resume quietly restarts
    from noise."""
    complete = _tiny_adapter(seed=0)
    truncated = nnx.state(complete)
    del truncated["refiner"]["embed"]

    mngr = ocp.CheckpointManager(
        tmp_path / "checkpoints", item_names=("model",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    mngr.save(0, args=ocp.args.Composite(model=ocp.args.StandardSave(truncated)))
    mngr.wait_until_finished()

    with pytest.raises(ValueError):
        restore_tolerating_legacy(
            lambda target: mngr.restore(
                0, args=ocp.args.Composite(model=ocp.args.StandardRestore(target))),
            _tiny_adapter(seed=99),
        )


def test_adapter_honors_time_signal():
    """#86: the production adapter must pass the time-signal choice through —
    sinusoidal builds no learned table (different param tree), table does. The
    default comes from config.TIME_SIGNAL, so the base run trains whatever the
    launch environment says."""
    import config
    sin = RefinerForTraining(DIM, nnx.Rngs(0), vocab_size=VOCAB, num_heads=4,
                             encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN,
                             time_signal="sinusoidal")
    tab = RefinerForTraining(DIM, nnx.Rngs(0), vocab_size=VOCAB, num_heads=4,
                             encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN,
                             time_signal="table")
    default = RefinerForTraining(DIM, nnx.Rngs(0), vocab_size=VOCAB, num_heads=4,
                                 encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN)
    assert "time_embed" not in nnx.state(sin.refiner, nnx.Param)
    assert "time_embed" in nnx.state(tab.refiner, nnx.Param)
    assert default.refiner.time_signal == config.TIME_SIGNAL
