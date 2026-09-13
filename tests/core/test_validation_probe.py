"""The in-loop validation probe: scores held-out data without touching training.

Also serves as the import canary for trainer.py and its split-out modules — a
syntax or wiring error there would otherwise only surface at the next launch.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def test_trainer_imports():
    from trm.train import trainer  # noqa: F401
    from trm.train import optimizers  # noqa: F401
    from trm.train import validation  # noqa: F401


def test_validation_probe_scores_and_preserves_training_state(tiny_model, monkeypatch):
    from trm.train import trainer

    from trm.train import validation


    if not trainer.DATA_ROOT:
        pytest.skip("DATA_ROOT not set")
    monkeypatch.setattr(validation, "VAL_ROWS", 1)

    probe = validation.ValidationProbe(trainer.DATA_ROOT)
    sentinel = jnp.ones_like(tiny_model.hunch_cache[...]) * 0.123
    tiny_model.hunch_cache[...] = sentinel

    val_ce = probe.run(tiny_model)

    assert val_ce is not None and np.isfinite(val_ce), f"validation CE not finite: {val_ce}"
    assert 0.0 < val_ce < 20.0, f"validation CE out of any plausible range: {val_ce}"
    np.testing.assert_array_equal(
        np.asarray(tiny_model.hunch_cache[...]), np.asarray(sentinel),
        err_msg="validation perturbed the training stream's carried hunch",
    )


# --- the probe scores without building full logits (#208) ----------------------

@pytest.mark.parametrize("arch", ["plain", "refiner", "reasoner"])
def test_chunked_probe_matches_full_logit_scoring(arch):
    """The probe used to build two [1, 512, 50304] f32 logit tensors inside the
    trainer's allocator every probe. Chunked scoring must measure the same CE: the
    reference below is the old full-logit computation, kept here and nowhere else."""
    import optax
    from flax import nnx

    from instruments.arch import build
    from trm.config import MAX_SEQ_LEN, PAD_TOKEN_ID
    from trm.train import validation

    model = build(arch, dim=60, seed=3, **({"num_layers": 2} if arch == "plain" else {}))
    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.integers(1, 50000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)
    batch = batch.at[0, -40:].set(PAD_TOKEN_ID)  # a padded tail, so the mask is exercised

    @nnx.jit
    def full_logit_sums(model, batch):
        seq1_in, seq1_out = batch[:, :MAX_SEQ_LEN], batch[:, 1:MAX_SEQ_LEN + 1]
        seq2_in, seq2_out = batch[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN], batch[:, MAX_SEQ_LEN + 1:]
        out1 = model(seq1_in, depth=validation.VAL_FIXED_DEPTH, training=False, new_document=True)
        out2 = model(seq2_in, depth=validation.VAL_FIXED_DEPTH, training=False, new_document=False)
        total, count = 0.0, 0
        for logits, targets in ((out1.logits, seq1_out), (out2.logits, seq2_out)):
            mask = targets != PAD_TOKEN_ID
            ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
            total, count = total + jnp.sum(ce * mask), count + jnp.sum(mask)
        return total, count

    with model.isolated_state():
        model.reset_state()
        ref_sum, ref_count = full_logit_sums(model, batch)
        model.reset_state()
        got_sum, got_count = validation._val_ce_sums(model, batch)

    assert int(got_count) == int(ref_count)
    np.testing.assert_allclose(float(got_sum) / float(got_count), float(ref_sum) / float(ref_count),
                               rtol=1e-5)


def test_the_probe_never_asks_for_full_logits():
    import inspect
    from trm.train import validation
    source = inspect.getsource(validation._val_ce_sums)
    assert "training=False" not in source and ".logits" not in source
