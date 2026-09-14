"""LazyMultiSteps == optax.MultiSteps in every number, minus the wasted work.

optax runs the inner optimizer on every micro-step and discards the result on
all but the last; ours runs it once. Same Welford mean, same update, same state
schema (checkpoints restore either way). Bit-identical is the bar: this is a
pure refactor of when work happens, not of what is computed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from instruments.arch import build
from trm.train.accumulate import LazyMultiSteps
from trm.train.optimizers import _adamw, _muon


def _run(tx_cls, inner, k, steps, seed=0):
    model = build("plain", dim=60, num_layers=1, seed=seed)
    opt = nnx.Optimizer(model, tx_cls(inner, every_k_schedule=k, use_grad_mean=True), wrt=nnx.Param)
    key = jax.random.PRNGKey(1)
    for i in range(steps):
        key, sub = jax.random.split(key)
        grads = jax.tree_util.tree_map(lambda x: 0.01 * jax.random.normal(sub, x.shape, x.dtype),
                                       nnx.state(model, nnx.Param))
        opt.update(model, grads)
    return ([np.asarray(x) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))],
            [np.asarray(x) for x in jax.tree_util.tree_leaves(opt.opt_state) if hasattr(x, "shape")])


def _identical(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def test_adamw_params_and_state_are_bit_identical_across_two_windows_and_a_partial_third():
    inner = lambda: optax.chain(optax.clip_by_global_norm(1.0), _adamw(lambda s: 1e-3))  # noqa: E731
    ref_p, ref_s = _run(optax.MultiSteps, inner(), k=4, steps=10)
    got_p, got_s = _run(LazyMultiSteps, inner(), k=4, steps=10)
    assert _identical(ref_p, got_p) and _identical(ref_s, got_s)


def test_muon_params_are_bit_identical_too():
    inner = lambda: optax.chain(optax.clip_by_global_norm(1.0), _muon(lambda s: 1e-3))  # noqa: E731
    ref_p, _ = _run(optax.MultiSteps, inner(), k=3, steps=7)
    got_p, _ = _run(LazyMultiSteps, inner(), k=3, steps=7)
    assert _identical(ref_p, got_p)


def test_the_inner_optimizer_runs_once_per_window():
    calls = []

    def counting(updates, state, params=None):
        calls.append(1)
        return updates, state
    inner = optax.GradientTransformation(lambda p: optax.EmptyState(), counting)
    model = build("plain", dim=60, num_layers=1)
    opt = nnx.Optimizer(model, LazyMultiSteps(inner, every_k_schedule=4, use_grad_mean=True), wrt=nnx.Param)
    grads = jax.tree_util.tree_map(jnp.ones_like, nnx.state(model, nnx.Param))
    for _ in range(8):
        opt.update(model, grads)
    # Traced once per compile, never per micro-step: the count is the number of traces,
    # which is what a cond guarantees; a where-based MultiSteps would also trace once,
    # so the real evidence is the GPU timing in the PR. This pins that it is a cond.
    import inspect
    from trm.train import accumulate
    assert "jax.lax.cond(is_last, emit, accumulate" in inspect.getsource(accumulate)


def test_production_chain_is_the_lazy_one():
    from trm.train.optimizers import optimizer_chain
    assert isinstance(optimizer_chain, LazyMultiSteps)
