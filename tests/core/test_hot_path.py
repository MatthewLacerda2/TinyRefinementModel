"""HotPath == compute_grad_step + apply_grads in every number (#474).

The hot path only changes how often Python walks the NNX graph and where the
accumulation window's counter is read; the jitted functions, their arguments and
their donation are the same. So the bar is bit-identical: losses, gradient norms,
parameters and optimizer state, across two accumulation windows and a partial third.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from instruments.arch import build
from trm.settings import CONFIG
from trm.train.accumulate import multi_steps
from trm.train.grad_step import HotPath, apply_grads, compute_grad_step
from trm.train.optimizers import _adamw, _muon

K, STEPS, WINDOW = 3, 7, 8


def _batches():
    rng = np.random.default_rng(0)
    return [jnp.array(rng.integers(0, 97, size=(2, 2 * WINDOW + 1)), dtype=jnp.int32) for _ in range(STEPS)]


def _setup(inner):
    model = build("plain", dim=60, num_layers=1, vocab_size=97, max_seq_len=WINDOW)
    return model, nnx.Optimizer(model, multi_steps(inner, every_k_schedule=K), wrt=nnx.Param)


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")]


def _run(inner, hot, touch_at=()):
    model, opt = _setup(inner)
    path = HotPath(model, opt) if hot else None
    seen = []
    for i, batch in enumerate(_batches()):
        step = jnp.array(i // K)
        if hot:
            if i in touch_at:  # what a checkpoint or the validation probe does mid-run
                _leaves(nnx.state(path.model, nnx.Param))
            loss, _, grads, norm = path.grad_step(batch, step, None)
            path.apply(grads)
        else:
            loss, _, grads, norm = compute_grad_step(model, batch, step, None)
            apply_grads(opt, grads, model)
        seen += [np.asarray(loss), np.asarray(norm)]
    if hot:
        path.check_counter()
        model, opt = path.model, path.optimizer
    return seen, _leaves(nnx.state(model, nnx.Param)), _leaves(opt.opt_state)


def _identical(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("inner", [
    lambda: optax.chain(optax.clip_by_global_norm(1.0), _adamw(CONFIG, lambda s: 1e-3)),
    lambda: optax.chain(optax.clip_by_global_norm(1.0), _muon(CONFIG, lambda s: 1e-3)),
], ids=["adamw", "muon"])
def test_the_hot_path_is_bit_identical_across_two_windows_and_a_partial_third(inner):
    ref = _run(inner(), hot=False)
    got = _run(inner(), hot=True)
    for a, b in zip(ref, got):
        assert _identical(a, b)


def test_reading_the_modules_mid_run_changes_nothing():
    inner = lambda: optax.chain(optax.clip_by_global_norm(1.0), _adamw(CONFIG, lambda s: 1e-3))  # noqa: E731
    ref = _run(inner(), hot=False)
    got = _run(inner(), hot=True, touch_at=(1, 3, 4))
    for a, b in zip(ref, got):
        assert _identical(a, b)


def test_a_change_made_through_the_module_is_seen_by_the_next_step():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt)
    batch = _batches()[0]
    before, *_ = path.grad_step(batch, jnp.array(0), None)
    for leaf in nnx.state(path.model, nnx.Param).flat_state():
        leaf[1].set_value(leaf[1].get_value() * 0)  # a reset done through the object, as reset_state does
    after, *_ = path.grad_step(batch, jnp.array(0), None)
    assert not np.array_equal(np.asarray(before), np.asarray(after))


def test_the_host_counter_follows_the_device_and_a_skipped_micro_step_does_not_move_it():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt)
    batches = _batches()
    for i, batch in enumerate(batches[:5]):
        _, _, grads, _ = path.grad_step(batch, jnp.array(i // K), None)
        if i != 2:  # the trainer skips the update on a non-finite micro-step
            path.apply(grads)
        path.check_counter()


def test_a_drifted_counter_is_caught():
    model, opt = _setup(optax.sgd(1e-3))
    path = HotPath(model, opt)
    path._mini += 1
    with pytest.raises(RuntimeError, match="counter drift"):
        path.check_counter()
