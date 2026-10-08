"""HotPath == compute_grad_step + apply_grads in every number (#474, #293).

The hot path changes how often Python walks the NNX graph, where the accumulation
window's counter is read, and where the gradient is folded: inside the program that
computes it, with the window's update emitted from the folded mean. The math is the
same fold and the same inner step, so the bar is bit-identical: losses, gradient
norms, parameters and optimizer state, across two accumulation windows and a
partial third.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from trm.model import build_model
from trm.settings import CONFIG
from trm.train.accumulate import multi_steps
from trm.train.grad_step import HotPath, apply_grads, compute_grad_step
from trm.train.optimizers import _adamw, _muon

K, STEPS, WINDOW = 3, 7, 8


def _batches():
    rng = np.random.default_rng(0)
    return [jnp.array(rng.integers(0, 97, size=(2, 2 * WINDOW + 1)), dtype=jnp.int32) for _ in range(STEPS)]


def _setup(inner):
    model = build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1, vocab_size=97, max_seq_len=WINDOW)
    return model, nnx.Optimizer(model, multi_steps(inner, every_k_schedule=K), wrt=nnx.Param)


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")]


def _run(inner, hot, touch_at=()):
    model, opt = _setup(inner)
    # The shipped z-loss (#369), so the identity covers its backward too.
    path = HotPath(model, opt, z_loss_weight=CONFIG.Z_LOSS_WEIGHT) if hot else None
    seen = []
    for i, batch in enumerate(_batches()):
        if hot:
            if i in touch_at:  # what a checkpoint or the validation probe does mid-run
                _leaves(nnx.state(path.model, nnx.Param))
            loss, _, norm, _ = path.step(batch)
            path.commit()
        else:
            loss, _, grads, norm = compute_grad_step(model, batch, z_loss_weight=CONFIG.Z_LOSS_WEIGHT)
            apply_grads(opt, grads, model)
        seen += [np.asarray(loss), np.asarray(norm)]
    if hot:
        path.check_counter()
        model, opt = path.model, path.optimizer
    return seen, _leaves(nnx.state(model, nnx.Param)), _leaves(opt.opt_state)


def _identical(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("inner", [
    lambda: optax.chain(optax.clip_by_global_norm(1.0), _adamw(CONFIG, lambda s: 1e-3)),
    lambda: optax.chain(optax.clip_by_global_norm(1.0), _muon(CONFIG, lambda s: 1e-3)),
], ids=["adamw", "muon"])
def test_the_hot_path_is_bit_identical_across_two_windows_and_a_partial_third(inner):
    ref = _run(inner(), hot=False)
    got = _run(inner(), hot=True)
    for a, b in zip(ref, got, strict=True):
        assert _identical(a, b)


def test_reading_the_modules_mid_run_changes_nothing():
    inner = lambda: optax.chain(optax.clip_by_global_norm(1.0), _adamw(CONFIG, lambda s: 1e-3))  # noqa: E731
    ref = _run(inner(), hot=False)
    got = _run(inner(), hot=True, touch_at=(1, 3, 4))
    for a, b in zip(ref, got, strict=True):
        assert _identical(a, b)


def test_a_change_made_through_the_module_is_seen_by_the_next_step():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt, z_loss_weight=0.0)
    batch = _batches()[0]
    before, *_ = path.step(batch)
    for leaf in nnx.state(path.model, nnx.Param).flat_state():
        leaf[1].set_value(leaf[1].get_value() * 0)  # a change made through the module object
    after, *_ = path.step(batch)
    assert not np.array_equal(np.asarray(before), np.asarray(after))


def test_the_host_counter_follows_the_device_and_a_skipped_micro_step_does_not_move_it():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt, z_loss_weight=0.0)
    batches = _batches()
    for i, batch in enumerate(batches[:5]):
        if i == 2:  # an overflowed micro-step: the trainer skips it and commits nothing
            loss, *_ = path.step(batch, loss_scale=np.float32(np.inf))
            assert not np.isfinite(float(loss))
        else:
            path.step(batch)
            path.commit()
        path.check_counter()


def test_a_non_finite_micro_step_folds_nothing():
    """#293 moved the skip onto the device: the fold happens in the step's own program,
    so a non-finite gradient must leave the accumulator exactly as it was (#199, #355)."""
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt, z_loss_weight=0.0)
    path.step(_batches()[0])
    path.commit()
    before = _leaves(path.optimizer.opt_state)
    path.step(_batches()[1], loss_scale=np.float32(np.inf))
    assert _identical(before, _leaves(path.optimizer.opt_state))


def test_it_refuses_an_optimizer_that_does_not_accumulate():
    model = build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1, vocab_size=97, max_seq_len=WINDOW)
    with pytest.raises(TypeError, match="accumulating optimizer"):
        HotPath(model, nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param), z_loss_weight=0.0)


def test_a_drifted_counter_is_caught():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt, z_loss_weight=0.0)
    path._mini += 1
    with pytest.raises(RuntimeError, match="counter drift"):
        path.check_counter()
