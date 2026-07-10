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

from plan_a_model import CausalRefiner, sinusoidal_step_encoding


def _tiny(vocab=37, dim=64, seq=32, enc=2, time_signal="learned"):
    return CausalRefiner(dim=dim, vocab_size=vocab, num_heads=4, num_encoder_layers=enc,
                         max_depth=8, max_seq_len=seq, time_signal=time_signal,
                         rngs=nnx.Rngs(0))


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


def test_return_all_iters_last_pass_matches_standard_forward():
    """The per-pass logits path (#75) must be a pure readout: pass k's logits for
    k = depth-1 equal the standard __call__ output, and gate means land in (0,1)."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    logits_all, gates = m(tok, depth=4, return_all_iters=True)
    std = m(tok, depth=4)
    assert logits_all.shape == (4,) + std.shape
    np.testing.assert_allclose(np.asarray(logits_all[-1]), np.asarray(std), atol=1e-5, rtol=0)
    assert gates.shape == (4,) and float(gates.min()) > 0.0 and float(gates.max()) < 1.0


def test_islands_gradient_routing():
    """Gradient islands (#75): with per-pass supervision every pass's time-embed
    row gets gradient (each island has its own test); with a final-only loss the
    islands cut reduces to last-pass-only credit. Encoder stays live in both."""
    depth = 8
    m = _tiny()
    tok = jax.random.randint(jax.random.PRNGKey(5), (2, 16), 0, 37)
    tgt = jax.random.randint(jax.random.PRNGKey(6), (2, 16), 0, 37)

    def per_pass_loss(mm):
        lg, _ = mm(tok, depth=depth, islands=True, return_all_iters=True)
        labels = jnp.broadcast_to(tgt, (depth,) + tgt.shape)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=lg, labels=labels))

    def final_only_loss(mm):
        lg, _ = mm(tok, depth=depth, islands=True, return_all_iters=True)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=lg[-1], labels=tgt))

    g_pp = nnx.grad(per_pass_loss)(m)
    rows = np.abs(np.asarray(g_pp["time_embed"]["embedding"][...])[:depth]).max(axis=1)
    assert (rows > 0.0).all(), "per-pass supervision left some pass without gradient"
    enc_norm = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g_pp["encoder"]))
    assert enc_norm > 0.0, "encoder lost its gradient path under islands"

    g_fo = nnx.grad(final_only_loss)(m)
    rows = np.abs(np.asarray(g_fo["time_embed"]["embedding"][...])[:depth]).max(axis=1)
    assert (rows[:-1] == 0.0).all(), "islands + final-only loss leaked credit into earlier passes"
    assert rows[-1] > 0.0


def test_depth_is_static_and_varies_output():
    """Different depths produce different logits (the loop actually does something)."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    d1 = np.asarray(m(tok, depth=1))
    d8 = np.asarray(m(tok, depth=8))
    assert np.max(np.abs(d1 - d8)) > 1e-3, "depth had no effect on the output"


# ── time-signal arms (#86) ───────────────────────────────────────────────────

def test_time_signal_arms_share_param_tree_and_init():
    """The three arms must be a matched pair's dream: identical param tree AND
    identical init values (the table is created even when unused, keeping the RNG
    stream aligned), so an ablation differs in exactly one thing — signal source."""
    states = [nnx.state(_tiny(time_signal=ts), nnx.Param) for ts in ("learned", "sinusoidal", "none")]
    flat = [jax.tree_util.tree_leaves_with_path(s) for s in states]
    assert [tuple(p for p, _ in f) for f in flat] == [tuple(p for p, _ in flat[0])] * 3
    for path_leaf in zip(*flat):
        arrays = [np.asarray(leaf) for _, leaf in path_leaf]
        assert all(np.array_equal(arrays[0], a) for a in arrays[1:]), path_leaf[0][0]


def test_learned_table_clamps_past_max_depth():
    """The premise of #86, pinned: past max_depth the learned arm reuses its LAST
    row (explicit clamp in the loop), so passes 9+ are indistinguishable from pass
    8 — but the forward stays finite. Without the clamp the raw gather NaN-fills
    on this JAX version (second assert documents the hazard the clamp guards)."""
    m = _tiny()
    tok = jnp.arange(1, 17)[None, :]
    out12 = np.asarray(m(tok, depth=12))
    assert np.isfinite(out12).all(), "depth overrun poisoned the forward"
    raw = np.asarray(m.time_embed(jnp.asarray(12)))
    assert not np.isfinite(raw).any(), \
        "raw out-of-range gather no longer NaN-fills — revisit whether the clamp is still needed"


def test_sinusoidal_signal_extends_past_max_depth():
    """The formula must keep discriminating where the table clamps: step 12's
    encoding differs from step 8's (and from every trained step's), stays finite,
    and is deterministic."""
    dim = 64
    codes = np.stack([np.asarray(sinusoidal_step_encoding(k, dim)) for k in range(17)])
    assert np.isfinite(codes).all()
    for k in range(9, 17):
        dists = np.abs(codes[:9] - codes[k]).max(axis=1)
        assert dists.min() > 1e-3, f"step {k}'s encoding collides with a trained step"
    np.testing.assert_array_equal(codes[12], np.asarray(sinusoidal_step_encoding(12, dim)))


@pytest.mark.parametrize("time_signal", ["sinusoidal", "none"])
def test_alternative_arms_ignore_the_table(time_signal):
    """Wiring check: in the sinusoidal/time-blind arms the table must be inert —
    zeroing it changes nothing, and it receives no gradient — while the learned
    arm demonstrably depends on it."""
    tok = jnp.arange(1, 17)[None, :]
    tgt = jax.random.randint(jax.random.PRNGKey(7), (1, 16), 0, 37)

    m = _tiny(time_signal=time_signal)
    before = np.asarray(m(tok, depth=4))
    m.time_embed.embedding[...] = jnp.zeros_like(m.time_embed.embedding[...])
    np.testing.assert_array_equal(before, np.asarray(m(tok, depth=4)))

    def loss(mm):
        lg = mm(tok, depth=4)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=lg, labels=tgt))

    grads = nnx.grad(loss)(m)
    assert float(jnp.abs(grads["time_embed"]["embedding"][...]).max()) == 0.0
    blk_norm = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(grads["refine_block"]))
    assert blk_norm > 0.0, "refine block lost its gradient in this arm"

    m_learned = _tiny()
    before = np.asarray(m_learned(tok, depth=4))
    m_learned.time_embed.embedding[...] = jnp.zeros_like(m_learned.time_embed.embedding[...])
    assert np.abs(before - np.asarray(m_learned(tok, depth=4))).max() > 1e-4, \
        "learned arm did not actually read the table (sanity)"
