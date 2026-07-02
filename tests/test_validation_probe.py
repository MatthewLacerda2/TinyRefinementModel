"""The in-loop validation probe: scores held-out data without touching training.

Also serves as the import canary for trainer.py — a syntax or wiring error
there would otherwise only surface at the next training launch.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def test_trainer_imports():
    import trainer  # noqa: F401


def test_validation_probe_scores_and_preserves_training_state(tiny_model, monkeypatch):
    import trainer

    if not trainer.DATA_ROOT:
        pytest.skip("DATA_ROOT not set")
    monkeypatch.setattr(trainer, "VAL_BATCHES", 1)

    probe = trainer.ValidationProbe()
    sentinel = jnp.ones_like(tiny_model.hunch_cache[...]) * 0.123
    tiny_model.hunch_cache[...] = sentinel

    val_ce = probe.run(tiny_model)

    assert val_ce is not None and np.isfinite(val_ce), f"validation CE not finite: {val_ce}"
    assert 0.0 < val_ce < 20.0, f"validation CE out of any plausible range: {val_ce}"
    np.testing.assert_array_equal(
        np.asarray(tiny_model.hunch_cache[...]), np.asarray(sentinel),
        err_msg="validation perturbed the training stream's carried hunch",
    )
