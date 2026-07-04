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


@pytest.mark.parametrize("grad_last", [1, 2])
def test_truncated_backprop_gradient_routing(grad_last):
    """Truncated backprop through depth (#64): grad_last=j must (a) leave the
    forward pass numerically unchanged — the cut computes z_enc + (z - z_enc),
    which only re-associates float addition, so equal up to ~1e-6, not bit-exact —
    (b) zero the gradient of every time-embedding row used before the cut (those
    steps live inside the detached prefix), (c) keep a nonzero gradient on the
    last j rows and on the encoder (the identity bypass)."""
    depth = 8
    m = _tiny()
    tok = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, 37)
    tgt = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, 37)

    def loss(mm, gl):
        lg = mm(tok, depth=depth, grad_last=gl)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=lg, labels=tgt))

    full = np.asarray(m(tok, depth=depth))
    trunc = np.asarray(m(tok, depth=depth, grad_last=grad_last))
    np.testing.assert_allclose(trunc, full, atol=1e-5, rtol=0,
                               err_msg="stop_gradient changed the forward pass")

    grads = nnx.grad(loss)(m, grad_last)
    t_grad = np.asarray(grads["time_embed"]["embedding"][...])
    cut = depth - grad_last
    pre_norm = np.abs(t_grad[:cut]).max()
    post_norm = np.abs(t_grad[cut:depth]).max()
    assert pre_norm == 0.0, f"time-embed rows before the cut got gradient ({pre_norm})"
    assert post_norm > 0.0, "time-embed rows after the cut got no gradient"

    enc_norm = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(grads["encoder"]))
    assert enc_norm > 0.0, "encoder lost its gradient path under truncated backprop"


def test_depth_is_static_and_varies_output():
    """Different depths produce different logits (the loop actually does something)."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    d1 = np.asarray(m(tok, depth=1))
    d8 = np.asarray(m(tok, depth=8))
    assert np.max(np.abs(d1 - d8)) > 1e-3, "depth had no effect on the output"
