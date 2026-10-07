"""Muon's partition, selector and chain (#26, stage 0).

The guard that matters most is the partition: optax routes every 2-D array to
Muon by default, which sends the tied token embedding — 50,304 lookup rows, not
a linear map — through Newton-Schulz. A wrong partition still trains and the
loss still falls, so nothing downstream would notice.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from trm.model import build_model
from trm.settings import CONFIG, Config
from trm.train import optimizers


def _labels(model):
    params = nnx.state(model, nnx.Param)
    labels = optimizers.muon_partition(params)
    return {jax.tree_util.keystr(p): (lab, leaf.ndim) for (p, lab), (_, leaf) in
            zip(jax.tree_util.tree_flatten_with_path(labels)[0], jax.tree_util.tree_flatten_with_path(params)[0])}


def test_matrices_go_to_muon_and_the_embedding_norms_and_biases_to_adam():
    labels = _labels(build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=2))
    embed = [k for k in labels if "embed" in k]
    assert embed and all(labels[k][0] == "adam" for k in embed), "the embedding table is not a linear map"
    kernels = [k for k, (lab, nd) in labels.items() if nd == 2 and "embed" not in k]
    assert kernels and all(labels[k][0] == "muon" for k in kernels)
    assert all(lab == "adam" for k, (lab, nd) in labels.items() if nd != 2)
    n_muon = sum(lab == "muon" for lab, _ in labels.values())
    assert n_muon >= 4 * 2, "every attention/MLP kernel must be on Muon"


def test_the_muon_chain_takes_a_step_and_stores_bf16_momentum_on_both_partitions():
    model = build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1)
    tx = optax.MultiSteps(optax.chain(optax.clip_by_global_norm(1.0), optimizers._muon(CONFIG, lambda s: 1e-3)),
                          every_k_schedule=1, use_grad_mean=True)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    before = jax.tree_util.tree_map(lambda x: np.asarray(x), nnx.state(model, nnx.Param))
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, nnx.state(model, nnx.Param))
    opt.update(model, grads)
    after = nnx.state(model, nnx.Param)
    moved = [not np.array_equal(b, np.asarray(a)) for b, a in
             zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after))]
    assert all(moved), "every parameter must receive an update"
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(after))
    mus = [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(opt.opt_state)[0]
           if ".mu[" in jax.tree_util.keystr(path)]
    mus = [m for m in mus if hasattr(m, "dtype") and m.ndim >= 1]
    assert mus and all(m.dtype == jnp.bfloat16 for m in mus), {m.dtype for m in mus}


def test_the_matrix_partition_runs_at_the_multiplied_lr():
    """Muon's update has RMS ~1 per element, so it wants a far larger LR than Adam's;
    the multiplier is what the #26 sweep turns."""
    model = build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1)
    steps = {}
    for mult in (1.0, 10.0):
        m = build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1)
        opt = nnx.Optimizer(m, optimizers._muon(CONFIG, lambda s: 1e-3, lr_mult=mult), wrt=nnx.Param)
        def kernel(state):
            return [np.asarray(leaf) for p, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
                    if leaf.ndim == 2 and "embed" not in jax.tree_util.keystr(p)][0]
        before = kernel(nnx.state(m, nnx.Param))
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, nnx.state(m, nnx.Param))
        opt.update(m, grads)
        steps[mult] = float(np.abs(kernel(nnx.state(m, nnx.Param)) - before).mean())
    del model
    assert steps[10.0] > 5 * steps[1.0]


def test_the_selector_fails_closed_and_the_default_is_muon():
    with pytest.raises(SystemExit) as typo:
        Config.from_env({"TRM_OPTIMIZER": "moun"})
    error = str(typo.value.code)
    assert "moun" in error and "adamw" in error and "muon" in error, "a typo'd TRM_OPTIMIZER must refuse"
    assert Config.from_env({}).TRM_OPTIMIZER == "muon"
    assert optimizers.inner_optimizer(Config.from_env({"TRM_OPTIMIZER": "adamw"}), lambda s: 1e-3) is not None
