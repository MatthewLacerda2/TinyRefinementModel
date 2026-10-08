"""On the card, the trainer's hot path must match the reference path bit for bit (#540).

    RUN_TESTS_ON_GPU=1 venv/bin/python -m pytest tests/expensive/test_gpu_identity.py -v

The shipped model and optimizer (`init_model_and_optimizer`, the trainer's own) are
built from the same seed and trained for two full accumulation windows and three
micro-steps of a third, on fixed tokens, through (a) `compute_grad_step` +
`apply_grads`, the reference, and (b) `HotPath`, what the trainer runs. Losses, grad
norms, parameters and optimizer state must be bitwise equal.

Why on the card: the CPU ignores buffer donation, so a donation or aliasing bug is
invisible there. #474 was bit-identical on every CPU test and still wrong on the 2060
(#533); only an ad-hoc old-vs-new script caught it. This is that script as a test.

Three runs, one at a time (two copies of the shipped model do not fit in 6 GB), each
reduced to a sha256 per leaf on the host:
- the reference twice: the control. If it differs from itself, the card is not
  deterministic and the identity test below it means nothing, not a regression.
- the hot path against the reference: the identity.
- the hot path with its update pointed at #533's split-state `jax.jit` apply: proof
  the comparison can see the bug class it exists for. That variant gave a different
  update on the card for the full model only (not on the CPU, not at dim 128), so this
  check, too, only means something here.

Any change to the hot path (trm/train/grad_step.py) runs this on the card before merging.
"""

import functools
import gc
import hashlib
import os

# Production's allocator (trm/train/start.py), set before jax loads: three full-size
# runs in one process must fit as the trainer's one does.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
# Bitwise identity needs deterministic kernels: by default the embedding's backward
# scatter-adds with atomics, and the reference differed from itself in exactly the
# embedding's gradient, moments and weights (#540's first card run).
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.config import VOCAB_SIZE
from trm.settings import CONFIG
from trm.train.grad_step import HotPath, apply_grads, compute_grad_step
from trm.train.loss_scale import DynamicLossScale

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not os.environ.get("RUN_TESTS_ON_GPU") or jax.default_backend() != "gpu",
                       reason="the identity that matters is the card's: set RUN_TESTS_ON_GPU=1 on a GPU"),
]

# Two optimizer updates and a partial third window: the second update is the first one
# whose inputs (mu, nu, the accumulator) were written by a donated update.
UPDATES = 2


def shipped():
    """The trainer's model and optimizer, at the shipped config and seed."""
    from trm.train.trainer import init_model_and_optimizer
    return init_model_and_optimizer(CONFIG)


def batches(n, batch_size, seq_len, vocab):
    rng = np.random.default_rng(0)
    return [jnp.array(rng.integers(1, vocab, size=(batch_size, 2 * seq_len + 1)), dtype=jnp.int32)
            for _ in range(n)]


# #533's split-state update, the known-bad variant: one jax.jit over the split model and
# optimizer state that merges both and runs `opt.update` on the window's last gradient,
# donating the params together with the optimizer state and the gradient (the set #533
# narrowed the divergence to). It runs on the reference path, not HotPath's: since #293
# HotPath folds that gradient before an emit that takes none, and the emit does not show
# the bug. Kept here only as the bug this test must be able to see.
@functools.partial(jax.jit, static_argnames=("graphdef", "opt_graphdef"),
                   donate_argnames=("params", "opt_state", "grads"))
def _apply_split_state(graphdef, params, rest, opt_graphdef, opt_state, grads):
    model = nnx.merge(graphdef, params, rest)
    opt = nnx.merge(opt_graphdef, opt_state)
    opt.update(model, grads)
    return nnx.state(model, nnx.Param), nnx.state(opt)


def apply_split_state(opt, grads, model):
    """apply_grads, with the window's update through `_apply_split_state`."""
    if not opt.tx.emits_next(nnx.pure(opt.opt_state)):
        return apply_grads(opt, grads, model)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    params, opt_state = _apply_split_state(graphdef, params, rest, nnx.graphdef(opt), nnx.state(opt), grads)
    nnx.update(model, params)
    nnx.update(opt, opt_state)


def fingerprint(tree, prefix):
    """{leaf path: sha256 of its bytes}: bitwise equality without a host copy of each run."""
    return {prefix + jax.tree_util.keystr(path): hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest()
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree) if hasattr(leaf, "shape")}


def train(path_kind, build=shipped, config=CONFIG):
    """One run from a fresh build: per-micro-step (loss, grad norm) and the final fingerprint."""
    model, opt = build()
    k = config.ACCUMULATION_STEPS
    tokens = batches(UPDATES * k + min(3, k - 1), config.BATCH_SIZE, config.MAX_SEQ_LEN, VOCAB_SIZE)
    # The trainer's first-step arguments: its loss scale, no clip ceiling yet.
    loss_scale = jnp.float32(DynamicLossScale(growth_interval=config.LOSS_SCALE_GROWTH_INTERVAL).value)
    clip_norm = jnp.float32(jnp.inf)
    hot = HotPath(model, opt, z_loss_weight=config.Z_LOSS_WEIGHT) if path_kind == "hot" else None
    apply = apply_split_state if path_kind == "split" else apply_grads
    scalars = []
    for step, batch in enumerate(tokens):
        if hot is None:
            loss, _, grads, norm = compute_grad_step(model, batch, loss_scale=loss_scale,
                                                     clip_norm=clip_norm, z_loss_weight=config.Z_LOSS_WEIGHT)
        else:
            loss, _, norm, _ = hot.step(batch, loss_scale=loss_scale, clip_norm=clip_norm)
        scalars.append((float(loss), float(norm)))
        assert np.isfinite(scalars[-1]).all(), f"non-finite micro-step {step}: {scalars[-1]}"
        if hot is None:
            apply(opt, grads, model)
        else:
            hot.commit()
    if hot is not None:
        model, opt = hot.model, hot.optimizer
    assert int(nnx.pure(opt.opt_state).gradient_step) == UPDATES, "the run must cross two updates"
    prints = {**fingerprint(nnx.state(model, nnx.Param), "params"), **fingerprint(opt.opt_state, "opt")}
    del model, opt, hot, tokens
    gc.collect()
    return scalars, prints


def differences(a, b):
    """What differs between two runs, first steps and leaves first; [] when bitwise equal."""
    (sa, pa), (sb, pb) = a, b
    out = [f"micro-step {i}: (loss, grad norm) {x} vs {y}" for i, (x, y) in enumerate(zip(sa, sb, strict=True)) if x != y]
    out += [f"{name} differs" for name in sorted(pa.keys() | pb.keys()) if pa.get(name) != pb.get(name)]
    return out


@pytest.fixture(scope="module")
def reference():
    return train("reference")


def test_control_the_reference_path_is_deterministic_on_this_card(reference):
    diff = differences(reference, train("reference"))
    assert not diff, ("the reference path differs from itself: this card is not deterministic, so the "
                      "identity below cannot be read as a regression either way\n  " + "\n  ".join(diff[:10]))


def test_the_hot_path_matches_the_reference_bit_for_bit(reference):
    diff = differences(reference, train("hot"))
    assert not diff, f"HotPath diverged from compute_grad_step + apply_grads ({len(diff)} differences)\n  " \
                     + "\n  ".join(diff[:10])


def test_the_comparison_sees_the_533_split_state_update(reference):
    # The test above must fail for this variant; if it passes, the identity is blind to
    # the donation bug class it exists for.
    assert differences(reference, train("split")), \
        "#533's split-state update matched the reference: the identity test cannot see that bug"
