"""Plan A wired into the production trainer (plan_a_trainer.RefinerForTraining).

test_plan_a.py proves the pure CausalRefiner; this proves the *integration* — that
the adapter drives the real grad step, that the no-op regularizers stay zero, and
that the refiner's param tree survives the Orbax round-trip the trainer uses. These
are the failure modes a config flag flip would otherwise hit only mid-run on the GPU.

Runs on CPU in f32 like the rest of the suite. Windows are sliced at MAX_SEQ_LEN, so
the batch must be 2*MAX_SEQ_LEN+1 wide; dims are kept tiny otherwise.
"""

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from config import MAX_SEQ_LEN, PAD_TOKEN_ID
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


def test_adapter_interface_and_zero_regularizers():
    """The adapter returns the ReasonerOutput the trainer consumes, with the
    hunch/forget/diversity terms as exact zeros (so their schedules add nothing)."""
    m = _tiny_adapter()
    tokens = _fixed_batch()[:, :MAX_SEQ_LEN]
    out = m(tokens, max_steps=4, training=True, should_refresh=True)

    # Training returns pre-head states (logits=None); the loss does the chunked
    # LM-head projection (#19). Inference still returns full logits (checked below).
    assert out.logits is None
    assert out.hidden.shape == (1, MAX_SEQ_LEN, DIM)
    assert np.isfinite(np.asarray(out.hidden)).all()
    infer_out = m(tokens, max_steps=4, training=False, should_refresh=True)
    assert infer_out.logits.shape == (1, MAX_SEQ_LEN, VOCAB)
    assert np.isfinite(np.asarray(infer_out.logits)).all()
    assert float(out.forget_cost) == 0.0
    assert float(out.diversity_loss) == 0.0
    # Vestigial buffer exists for the trainer's bookkeeping writes.
    assert m.hunch_cache[...].shape == (1, 1, DIM)


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
    reference = np.asarray(m(tokens, max_steps=2, training=False, should_refresh=True).logits)

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

    roundtripped = np.asarray(other(tokens, max_steps=2, training=False, should_refresh=True).logits)
    np.testing.assert_array_equal(reference, roundtripped)


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
