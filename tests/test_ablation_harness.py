"""The #34 additions to the proof instrument: the memorization probe and the
matched-compute vanilla arm. Tiny configs — these guard wiring, not results."""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import pytest

from ablation_harness import VanillaTransformer, memorize_task, train_one
from plan_a_model import CausalRefiner


def test_memorize_dictionary_is_fixed_across_calls():
    """Same key → same value on every call: recall over TRAINED pairs is only a
    capacity metric if the dictionary never resamples."""
    fn_a, fn_b = memorize_task(64), memorize_task(64)
    xa, ta, _ = fn_a(jax.random.PRNGKey(0), 8, 12)
    xb, tb, _ = fn_b(jax.random.PRNGKey(1), 8, 12)
    lut = {}
    for x, t in [(xa, ta), (xb, tb)]:
        for k, v in zip(np.asarray(x).ravel(), np.asarray(t).ravel()):
            assert lut.setdefault(int(k), int(v)) == int(v), f"key {k} mapped to two values"


def test_vanilla_arm_shapes_and_param_scaling():
    """The vanilla arm produces logits of the right shape, and its parameters grow
    with depth (distinct blocks) while the refiner's stay flat (shared block)."""
    def params(m):
        return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(m, nnx.Param)))

    kw = dict(dim=32, vocab_size=16, num_heads=4, num_encoder_layers=1, max_seq_len=16)
    v2 = VanillaTransformer(**kw, num_blocks=2, rngs=nnx.Rngs(0))
    v4 = VanillaTransformer(**kw, num_blocks=4, rngs=nnx.Rngs(0))
    r2 = CausalRefiner(**kw, max_depth=2, rngs=nnx.Rngs(0))
    r4 = CausalRefiner(**kw, max_depth=4, rngs=nnx.Rngs(0))

    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    assert v4(tokens).shape == (2, 16, 16)
    assert params(v4) > params(v2), "distinct blocks must add parameters with depth"
    # The refiner's block is shared: depth only grows the (tiny) time embedding.
    assert params(r4) - params(r2) < params(v4) - params(v2)


def test_train_one_runs_both_arms_on_memorize():
    """The harness's full train/eval path works for both arms on the new task and
    beats chance on a trivially small dictionary."""
    fn = memorize_task(8)
    for arch in ("refiner", "vanilla"):
        acc, ce, n_params = train_one(fn, 8, 2, arch=arch, dim=32, heads=4, enc=1,
                                      steps=60, batch=64, n_pool=512, n_test=128,
                                      train_seq=8)
        assert np.isfinite(ce), f"{arch}: CE not finite"
        assert n_params > 0
        assert acc > 1.0 / 8, f"{arch}: recall {acc} not above chance"


def test_per_depth_loss_requires_refiner_arm():
    """The vanilla arm has distinct, non-shared blocks per depth — there is no
    single K-loop to grade at every iteration, so #74's per-depth loss must
    refuse to run on it rather than silently doing something ill-defined."""
    fn = memorize_task(8)
    with pytest.raises(AssertionError):
        train_one(fn, 8, 2, arch="vanilla", per_depth_loss=True, dim=32, heads=4,
                 enc=1, steps=1, batch=64, n_pool=512, n_test=128, train_seq=8)


def test_per_depth_loss_trains_and_reports_depth_curve():
    """#74 wiring: --per-depth-loss trains without error and beats chance, and
    --depth-curve reports one finite accuracy per intermediate depth for both
    the final-only and per-depth-loss arms."""
    fn = memorize_task(8)
    depth = 3
    for per_depth_loss in (False, True):
        out = train_one(fn, 8, depth, arch="refiner", per_depth_loss=per_depth_loss,
                        depth_curve=True, dim=32, heads=4, enc=1, steps=80, batch=64,
                        n_pool=512, n_test=128, train_seq=8)
        acc, ce, n_params, curve = out
        assert np.isfinite(ce)
        assert acc > 1.0 / 8, f"per_depth_loss={per_depth_loss}: recall {acc} not above chance"
        assert [d for d, _, _ in curve] == list(range(1, depth + 1))
        assert all(np.isfinite(a) and np.isfinite(c) for _, a, c in curve)
        assert curve[-1][1] == pytest.approx(acc), "depth-curve's last point must match the top-level eval at full depth"
