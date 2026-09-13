"""The underflow watch reads the gradient that updates the weights (#191).

It read one micro-step's grads. On #157 that column hit exactly 0.500 on 13% of
steps: a depth-1 draw zeroes half of one norm's gradient, in that micro-step only,
while the 128-step mean that the optimizer applies had no such zeros. Anyone
reading the column against #82's 0.05 bar would think underflow was blowing it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from trm.train.grad_step import applied_gradient, dense_zero_frac_max, grad_zero_fractions


class Toy(nnx.Module):
    def __init__(self):
        self.dense = nnx.Param(jnp.zeros(4))


def _grads(model, values):
    return jax.tree_util.tree_map(lambda _: jnp.asarray(values, dtype=jnp.float32), nnx.state(model, nnx.Param))


def test_it_is_exactly_what_the_optimizer_applies():
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.MultiSteps(optax.sgd(1.0), every_k_schedule=4, use_grad_mean=True),
                              wrt=nnx.Param)
    window = [[1.0, 2.0, 0.0, 4.0], [3.0, 0.0, 0.0, 4.0], [5.0, 2.0, 0.0, 4.0], [7.0, 0.0, 6.0, 4.0]]
    for values in window[:-1]:
        optimizer.update(model, _grads(model, values))
    last = _grads(model, window[-1])
    predicted = np.asarray(jax.tree_util.tree_leaves(applied_gradient(optimizer, last))[0])

    before = np.asarray(model.dense[...])
    optimizer.update(model, last)
    applied = before - np.asarray(model.dense[...])   # sgd(1.0): the step IS the gradient
    np.testing.assert_allclose(predicted, np.mean(window, axis=0), rtol=1e-6)
    np.testing.assert_allclose(predicted, applied, rtol=1e-6)


def test_a_micro_step_artifact_does_not_reach_the_applied_reading():
    """The #157 shape: one micro-step with half its entries zero, a window whose mean has none."""
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.MultiSteps(optax.sgd(1.0), every_k_schedule=4, use_grad_mean=True),
                              wrt=nnx.Param)
    for _ in range(3):
        optimizer.update(model, _grads(model, [1.0, 1.0, 1.0, 1.0]))
    artifact = _grads(model, [1.0, 1.0, 0.0, 0.0])

    micro = dense_zero_frac_max(grad_zero_fractions(artifact))
    applied = dense_zero_frac_max(grad_zero_fractions(applied_gradient(optimizer, artifact)))
    assert float(micro) == 0.5 and float(applied) == 0.0


def test_the_trainer_logs_both_and_names_them_apart():
    import inspect
    from trm.runtime.metrics import MetricsLogger
    from trm.train import trainer

    source = inspect.getsource(trainer.train_loop)
    assert "applied_gradient(optimizer, grads)" in source
    assert "applied_zero_frac_dense_max=" in source and "zero_frac_dense_max=" in source
    fields = MetricsLogger("/dev/null").fields
    assert "applied_zero_frac_dense_max" in fields and "zero_frac_dense_max" in fields
