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
    assert "applied_gradient_stats(optimizer, grads)" in source, "jitted stats, never the materialized tree (#26)"
    assert "applied_gradient(optimizer, grads)" not in source
    # That both are LOGGED is observed, not read off the source: a real run's row
    # carries them (tests/apparatus/test_trainer_end_to_end.py).
    fields = MetricsLogger("/dev/null").fields
    assert "applied_zero_frac_dense_max" in fields and "zero_frac_dense_max" in fields


# --- the norm the clip sees (#180) -----------------------------------------------

def test_the_logged_norm_is_the_applied_gradients_the_one_the_clip_acts_on():
    """grad_norm_avg (~11 on the 4B run, per micro-step) was never comparable to the
    1.0 clip, which acts on the 128-step mean. The trainer now logs that mean's norm."""
    import inspect
    from trm.train import optimizers

    # The logged column is observed in a real run (tests/apparatus/test_trainer_end_to_end.py);
    # the clip it is read against is the optimizer's.
    assert "clip_by_global_norm(CLIP_NORM)" in inspect.getsource(optimizers)


def test_a_window_of_large_micro_steps_can_have_a_small_applied_norm():
    """Why the micro-step norm misleads: opposing micro-step gradients cancel in the mean."""
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.MultiSteps(optax.sgd(1.0), every_k_schedule=4, use_grad_mean=True),
                              wrt=nnx.Param)
    for values in ([10.0, 0, 0, 0], [-10.0, 0, 0, 0], [10.0, 0, 0, 0]):
        optimizer.update(model, _grads(model, values))
    last = _grads(model, [-10.0, 0, 0, 0])
    assert float(optax.global_norm(last)) == 10.0
    assert float(optax.global_norm(applied_gradient(optimizer, last))) == 0.0


def test_the_jitted_stats_equal_the_materialized_tree():
    """Same numbers as applied_gradient() + grad_zero_fractions + global_norm, without
    building the 564 MiB tree eagerly (which OOM'd every Muon arm, #26)."""
    from trm.train.grad_step import applied_gradient_stats
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.MultiSteps(optax.sgd(1.0), every_k_schedule=4, use_grad_mean=True),
                              wrt=nnx.Param)
    for _ in range(3):
        optimizer.update(model, _grads(model, [1.0, 1.0, 1.0, 1.0]))
    last = _grads(model, [1.0, 1.0, 0.0, 0.0])
    fracs, norm = applied_gradient_stats(optimizer, last)
    applied = applied_gradient(optimizer, last)
    assert {k: float(v) for k, v in fracs.items()} == {k: float(v) for k, v in grad_zero_fractions(applied).items()}
    assert float(norm) == float(optax.global_norm(applied))
    import inspect
    from trm.train import grad_step
    assert "@jax.jit" in inspect.getsource(grad_step).split("def _applied_stats")[0][-40:]
