"""The #34 additions to the proof instrument: the memorization probe and the
matched-compute vanilla arm. Tiny configs — these guard wiring, not results."""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

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
