"""The tied embedding decays at the standard rate under Muon (#360), and nothing else moved.

Before #360 the embedding sat in Adam's partition under WEIGHT_DECAY x LR: 6e-6 per step
at the 6e-4 peak, ten times below nanoGPT/GPT-3's 0.1 x 6e-4. The matrices under Muon
already decay above Moonlight's figure, so they must not move, and neither may the
optimizer state's tree: a checkpoint written before this change still has to resume.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.model import build_model
from trm.settings import CONFIG
from trm.train import optimizers

PEAK = CONFIG.PEAK_LR


def _model():
    return build_model(CONFIG, nnx.Rngs(0), dim=60, num_layers=1)


def _params_by_path(model):
    return {jax.tree_util.keystr(p): np.asarray(leaf, np.float64) for p, leaf in
            jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]}


def _after_one_step(tx, grad_of):
    """The params of a fresh model after one update under `tx`, by path."""
    model = _model()
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    params = nnx.state(model, nnx.Param)
    keys = iter(jax.random.split(jax.random.key(1), len(jax.tree_util.tree_leaves(params))))
    opt.update(model, jax.tree_util.tree_map(lambda x: grad_of(next(keys), x), params))
    return _params_by_path(model)


def _embedding_shrink(tx):
    """The fraction of each embedding weight one step removes, with a zero gradient:
    Adam's step is then exactly zero, so what moves the table is the decay alone.
    A least-squares fit over the table, so f32 rounding of single entries averages out."""
    before = _params_by_path(_model())
    after = _after_one_step(tx, lambda _, x: jnp.zeros_like(x))
    (embed,) = [k for k in before if "embed" in k]
    return 1.0 - float((after[embed] * before[embed]).sum() / (before[embed] ** 2).sum())


def _pre_360_muon(monkeypatch, lr):
    """The chain as it was: Adam's partition under WEIGHT_DECAY's schedule."""
    with monkeypatch.context() as m:
        m.setattr(optimizers, "_embedding_decay", optimizers._weight_decay)
        return optimizers._muon(CONFIG, lr)


def test_the_embedding_decays_by_the_knob_per_step_at_the_peak_and_follows_the_schedule():
    np.testing.assert_allclose(_embedding_shrink(optimizers._muon(CONFIG, lambda s: PEAK)),
                               CONFIG.EMBED_WEIGHT_DECAY, rtol=1e-3)
    # The schedule's shape, not its scale: half the LR, half the decay.
    np.testing.assert_allclose(_embedding_shrink(optimizers._muon(CONFIG, lambda s: PEAK / 2)),
                               CONFIG.EMBED_WEIGHT_DECAY / 2, rtol=1e-3)


def test_every_other_parameter_steps_exactly_as_before(monkeypatch):
    """Matrices, norms and biases: bit-identical to the pre-#360 chain on the same
    gradient. Only the embedding differs."""
    def grad_of(key, x):
        return 0.01 * jax.random.normal(key, x.shape, x.dtype)

    new = _after_one_step(optimizers._muon(CONFIG, lambda s: PEAK), grad_of)
    old = _after_one_step(_pre_360_muon(monkeypatch, lambda s: PEAK), grad_of)
    assert new.keys() == old.keys()
    changed = [k for k in new if not np.array_equal(new[k], old[k])]
    assert changed and all("embed" in k for k in changed), changed
    assert len(new) - len(changed) >= 4, "the matrices and norms must be in the comparison"


def test_the_optimizer_state_keeps_its_tree_so_old_checkpoints_resume(monkeypatch):
    params = nnx.state(_model(), nnx.Param)
    new = optimizers._muon(CONFIG, lambda s: PEAK).init(params)
    old = _pre_360_muon(monkeypatch, lambda s: PEAK).init(params)
    assert jax.tree_util.tree_structure(new) == jax.tree_util.tree_structure(old)
    assert ([(x.shape, x.dtype) for x in jax.tree_util.tree_leaves(new)]
            == [(x.shape, x.dtype) for x in jax.tree_util.tree_leaves(old)])


def test_adamw_keeps_the_pre_muon_recipe():
    """Under TRM_OPTIMIZER=adamw the embedding decays by WEIGHT_DECAY x LR like every
    matrix, as every AdamW run on record did."""
    np.testing.assert_allclose(_embedding_shrink(optimizers._adamw(CONFIG, lambda s: PEAK)),
                               CONFIG.WEIGHT_DECAY * PEAK, rtol=1e-3)
