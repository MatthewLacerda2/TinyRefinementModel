"""The three settled adoptions (#357 f32 residual stream, #359 Adam b2 0.95, #369 z-loss
1e-4): each knob does what it says, and setting all three back reproduces the numbers
main computed before them.

The reproduction is against a trajectory recorded on main at the commit before the
adoptions (tests/core/fixtures/pre_adoption_trajectory.json), by `_trajectory` below
run on that commit's code: a tiny plain model trained through the shipped optimizer
chain (Muon + Adam) for three accumulation windows. Compared across machines, so within
RTOL, ten times the golden run's measured cross-CPU floor (#330); the same test checks
that the adopted defaults move the trajectory by far more than that, so the tolerance
cannot hide a knob that was silently left on.
"""

import json
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from trm.model.plain import PlainTransformer
from trm.runtime.run_tracker import RunTracker
from trm.settings import Config
from trm.train.grad_step import apply_grads, compute_grad_step
from trm.train.losses import chunked_cross_entropy
from trm.train.optimizers import optimizer_chain

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "pre_adoption_trajectory.json"
RTOL = 1.8e-6  # 10x tests/expensive/test_golden_run.py's NOISE_FLOOR
WINDOW, STEPS = 8, 12
ADOPTED = {"RESIDUAL_DTYPE": "float32", "ADAM_B2": 0.95, "Z_LOSS_WEIGHT": 1e-4}
SET_BACK = {"RESIDUAL_DTYPE": "float16", "ADAM_B2": 0.999, "Z_LOSS_WEIGHT": 0.0,
            # the fixture predates #375 too, so the Newton-Schulz coefficients go back with them
            "MUON_NS_COEFFS": "keller"}


def _trajectory(config):
    """(losses, grad norms) of six micro-steps, accumulation 2, through the run's own
    optimizer chain at a constant LR. Kept identical to the recording script."""
    model = PlainTransformer(32, nnx.Rngs(3), config, vocab_size=97, num_heads=4, num_layers=2,
                             max_seq_len=WINDOW)
    opt = nnx.Optimizer(model, optimizer_chain(config, lambda step: 1e-3), wrt=nnx.Param)
    rng = np.random.default_rng(5)
    losses, norms = [], []
    for _ in range(STEPS):
        batch = jnp.array(rng.integers(1, 97, size=(2, 2 * WINDOW + 1)), dtype=jnp.int32)
        loss, _, grads, norm = compute_grad_step(model, batch, z_loss_weight=config.Z_LOSS_WEIGHT)
        apply_grads(opt, grads, model)
        losses.append(float(loss))
        norms.append(float(norm))
    return np.array(losses), np.array(norms)


def _config(**knobs):
    return Config(BATCH_SIZE=64, FORCE_F32_COMPUTE=True, **knobs)  # accumulation 2


def test_the_defaults_are_the_adopted_values_and_the_run_records_them():
    config = Config()
    assert {k: getattr(config, k) for k in ADOPTED} == ADOPTED
    recorded = RunTracker.get_hyperparameters(config)
    assert {k: recorded[k] for k in ADOPTED} == ADOPTED


@pytest.mark.skipif(bool(os.environ.get("RUN_TESTS_ON_GPU")), reason=(
    "the fixture is a CPU f32 trajectory; GPU kernels reduce in another order, so it "
    "cannot match to RTOL there (#452)"))
def test_setting_all_three_back_reproduces_main_before_them():
    recorded = json.loads(FIXTURE.read_text())
    losses, norms = _trajectory(_config(**SET_BACK))
    np.testing.assert_allclose(losses, recorded["losses"], rtol=RTOL)
    np.testing.assert_allclose(norms, recorded["grad_norms"], rtol=RTOL)


@pytest.mark.parametrize("knob", ["ADAM_B2", "Z_LOSS_WEIGHT"])
def test_each_adopted_value_moves_the_trajectory_past_the_tolerance(knob):
    """The f32 test path cannot see RESIDUAL_DTYPE (the stream is never narrower than
    compute); test_the_stream_has_the_knobs_dtype covers it under f16 compute."""
    recorded = json.loads(FIXTURE.read_text())
    got = np.concatenate(_trajectory(_config(**{**SET_BACK, knob: ADOPTED[knob]})))
    ref = np.array(recorded["losses"] + recorded["grad_norms"])
    assert np.max(np.abs(got - ref) / np.abs(ref)) > 3 * RTOL


@pytest.mark.parametrize("residual", ["float16", "float32"])
def test_the_stream_has_the_knobs_dtype(residual):
    """Built with f16 compute explicitly, so the check does not depend on the suite's
    FORCE_F32_COMPUTE. Every state of the stream is RESIDUAL_DTYPE, and the matmuls
    still get f16: the loss head reads f16 either way. float16 is the pre-#357 stream."""
    config = Config(RESIDUAL_DTYPE=residual)
    model = PlainTransformer(32, nnx.Rngs(0), config, vocab_size=37, num_heads=4, num_layers=2,
                             max_seq_len=16, dtype=jnp.float16)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 37
    *_, states = model._stream(tokens, keep_states=True)
    assert [s.dtype for s in states] == [jnp.dtype(residual)] * 3
    assert model(tokens, training=True).hidden.dtype == jnp.float16


def test_the_stream_knob_moves_no_weight():
    """The same seed draws the same parameters at either width: a run can switch the
    stream without a different init, and a checkpoint loads either way."""
    def params(residual):
        model = PlainTransformer(32, nnx.Rngs(0), Config(RESIDUAL_DTYPE=residual), vocab_size=37,
                                 num_heads=4, num_layers=2, max_seq_len=16, dtype=jnp.float16)
        return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    for a, b in zip(params("float16"), params("float32"), strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_the_stream_is_never_narrower_than_compute():
    """Under FORCE_F32_COMPUTE (the suite's) a float16 stream is f32, so the test path
    and the golden run stay all-f32."""
    model = PlainTransformer(32, nnx.Rngs(0), Config(RESIDUAL_DTYPE="float16"), vocab_size=37,
                             num_heads=4, num_layers=1, max_seq_len=16, dtype=jnp.float32)
    *_, states = model._stream(jnp.zeros((1, 16), dtype=jnp.int32), keep_states=True)
    assert all(s.dtype == jnp.float32 for s in states)


def test_z_loss_gradient_matches_naive_and_the_reported_ce_excludes_it(ce_batch):
    """z_weight adds z * (log Z)^2 per position to the gradient only: the backward must
    equal jax.grad of the naive CE + z-loss mean, and the returned loss must stay the
    plain CE so every recorded CE stays comparable."""
    hidden, embedding, targets = ce_batch(seed=3)
    pad_id, z = 0, 1e-2

    def naive(h, e):
        logits = jnp.matmul(h, e.astype(h.dtype).T, preferred_element_type=jnp.float32)
        mask = (targets != pad_id).astype(jnp.float32)
        per = (optax.softmax_cross_entropy_with_integer_labels(logits, targets)
               + z * jax.nn.logsumexp(logits, axis=-1) ** 2)
        return jnp.sum(per * mask) / jnp.sum(mask).clip(min=1.0)

    naive_g = jax.grad(naive, argnums=(0, 1))(hidden, embedding)
    chunk_g = jax.grad(lambda h, e: chunked_cross_entropy(h, e, targets, pad_id, 13, z)[0],
                       argnums=(0, 1))(hidden, embedding)
    assert jnp.allclose(naive_g[0], chunk_g[0], rtol=1e-4, atol=1e-6), "grad wrt hidden differs"
    assert jnp.allclose(naive_g[1], chunk_g[1], rtol=1e-4, atol=1e-6), "grad wrt embedding differs"

    plain, _ = chunked_cross_entropy(hidden, embedding, targets, pad_id, 13)
    with_z, _ = chunked_cross_entropy(hidden, embedding, targets, pad_id, 13, z)
    assert float(plain) == float(with_z)
    without_g = jax.grad(lambda h: chunked_cross_entropy(h, embedding, targets, pad_id, 13)[0])(hidden)
    assert not jnp.allclose(chunk_g[0], without_g, atol=1e-7), "z-loss changed no gradient"


def test_the_trainer_hands_the_z_loss_to_the_hot_path():
    import inspect

    from trm.train import loop
    assert "HotPath(model, optimizer, z_loss_weight=config.Z_LOSS_WEIGHT)" in inspect.getsource(loop)
