"""The training loop speaks plain LM, and keeps speaking it (#105).

Before this, the loop's idea of "a model" was the shape of the original reasoner:
it did hunch bookkeeping, scheduled a forget cost and a diversity loss, and sent a
refresh signal — so every live architecture had to wear that costume, and Plan A
was integrated by dressing up as the arch we had already proved inert.

Two things keep the inversion from quietly rotting back:
  1. the shared loop names no architecture's machinery, and
  2. both architectures really do train through the same call.

The second is the one that matters; the first is what stops the seam from being
re-opened one convenient attribute at a time.

The first half, the source scan, lives in test_neutral_contract_source.py (#325).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from trm.train.grad_step import compute_grad_step
from trm.model.contract import LanguageModel


def test_contract_defaults_describe_a_plain_stateless_lm():
    """A new architecture must be able to implement `__call__` and stop. If these
    defaults ever stop being no-ops, every future model inherits a chore."""

    class Minimal(LanguageModel):
        def __call__(self, tokens, depth, training=False, new_document=True):
            raise AssertionError("not called")

    m = Minimal()
    assert m.grade_aux([{}, {}], 0) == {}
    assert m.legacy_checkpoint_variables() == {}
    m.reset_state()
    m.end_step(True)
    with m.isolated_state():
        pass


def _tiny_refiner():
    from trm.model.refiner_lm import RefinerForTraining

    return RefinerForTraining(
        128, nnx.Rngs(0), vocab_size=37, num_heads=4,
        encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN,
    )


@pytest.mark.parametrize("arch", ["refiner", "reasoner"])
def test_both_arches_train_through_the_identical_call(arch, make_reasoner_model):
    """The same `compute_grad_step` invocation — no arch branch, no adapter
    shim — drives a model that carries state and grades two regularizers, and one
    that does neither."""
    model = _tiny_refiner() if arch == "refiner" else make_reasoner_model(seed=5)
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.integers(1, 37, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)

    loss, out, grads, grad_norm = compute_grad_step(
        model, batch, step=0, depth=2, doc_boundary=True)

    assert np.isfinite(float(loss)) and np.isfinite(float(grad_norm))
    assert float(grad_norm) > 0.0, "no gradient flowed"
    assert out.logits is None, "training must return pre-head states, not logits"
    # Whatever the arch reported got graded and added; nothing else was required.
    assert isinstance(out.aux, dict)
