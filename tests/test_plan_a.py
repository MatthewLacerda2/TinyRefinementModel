"""Test battery for the Plan A model (plan_a_model.CausalRefiner).

Self-contained (does not use the UniversalReasoner conftest fixtures) — constructs
tiny CausalRefiners directly. Runs on CPU in f32 by default, like the rest of the
suite.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from plan_a_model import CausalRefiner


def _tiny(vocab=37, dim=64, seq=32, enc=2):
    return CausalRefiner(dim=dim, vocab_size=vocab, num_heads=4, num_encoder_layers=enc,
                         max_depth=8, max_seq_len=seq, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_causality_no_future_leak(depth):
    """Perturbing token at position j must not change any logit at positions < j,
    at any depth — the leak the slot architecture could not close."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    j = 8
    a = np.asarray(m(tok, depth=depth))
    b = np.asarray(m(tok.at[0, j].set(20), depth=depth))
    assert np.max(np.abs(a[0, :j] - b[0, :j])) < 1e-5, "future token leaked into an earlier position"
    assert np.max(np.abs(a[0, j:] - b[0, j:])) > 1e-4, "perturbation had no effect (sanity)"


def test_init_loss_near_ln_vocab():
    """At init the model should be near-uniform: CE ~ ln(vocab)."""
    vocab = 50
    m = _tiny(vocab=vocab)
    tok = jax.random.randint(jax.random.PRNGKey(1), (8, 32), 0, vocab)
    logits = m(tok, depth=4)
    ce = float(jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tok)))
    assert abs(ce - math.log(vocab)) < 0.7, f"init CE {ce:.3f} far from ln(vocab)={math.log(vocab):.3f}"


def test_overfit_single_batch():
    """The looped block can actually learn: drive a fixed batch's loss far down."""
    vocab = 17
    m = _tiny(vocab=vocab, seq=16)
    opt = nnx.Optimizer(m, optax.adamw(3e-3), wrt=nnx.Param)
    inp = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, vocab)
    tgt = jax.random.randint(jax.random.PRNGKey(2), (4, 16), 0, vocab)

    @nnx.jit
    def step(m, opt, inp, tgt):
        def loss_fn(mm):
            lg = mm(inp, depth=4)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=lg, labels=tgt))
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    first = float(step(m, opt, inp, tgt))
    for _ in range(250):
        last = float(step(m, opt, inp, tgt))
    assert last < 0.15 * first, f"failed to overfit: {first:.3f} -> {last:.3f}"


def test_depth_is_static_and_varies_output():
    """Different depths produce different logits (the loop actually does something)."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    d1 = np.asarray(m(tok, depth=1))
    d8 = np.asarray(m(tok, depth=8))
    assert np.max(np.abs(d1 - d8)) > 1e-3, "depth had no effect on the output"
