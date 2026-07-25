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


def test_depth_overrun_fails_loud():
    """Depth past max_depth has no trained time-embedding rows and the row index
    silently clamps (findings 2026-06-18 caveats, 2026-07-05 readout 5) — the
    model must refuse it unless the caller opts into a deliberate probe."""
    m = _tiny()  # max_depth=8
    tok = jnp.arange(1, 17)[None, :]
    with pytest.raises(AssertionError, match="allow_depth_overrun"):
        m(tok, depth=9)
    # The opt-in must still run — degraded numbers are the probe's business
    # (at random init deep overrun can even overflow, so only shape is checked).
    out = np.asarray(m(tok, depth=12, allow_depth_overrun=True))
    assert out.shape == (1, 16, 37)


def test_sinusoidal_time_signal_extends_past_max_depth():
    """#86: the sinusoidal step encoding is a fixed function of the step index,
    so a sinusoidal-signal model has no depth ceiling — depth 12 on a
    max_depth=8 model must run WITHOUT the overrun opt-in (which exists only
    for the table's clamp hazard) and stay finite. The param tree must also
    carry no time_embed table."""
    m = CausalRefiner(dim=64, vocab_size=37, num_heads=4, num_encoder_layers=2,
                      max_depth=8, max_seq_len=32, time_signal="sinusoidal",
                      rngs=nnx.Rngs(0))
    tok = jnp.arange(1, 17)[None, :]
    out = np.asarray(m(tok, depth=12))
    assert out.shape == (1, 16, 37)
    assert np.isfinite(out).all()
    assert "time_embed" not in nnx.state(m, nnx.Param), \
        "sinusoidal mode must not build the learned table"


def test_sinusoidal_encoding_distinct_and_deterministic():
    """Each step must get its own signal (the whole point of a time signal),
    the same signal every call, defined arbitrarily far out."""
    from plan_a_model import sinusoidal_step_encoding
    encs = [np.asarray(sinusoidal_step_encoding(s, 64, jnp.float32)) for s in range(16)]
    for a in range(16):
        assert encs[a].shape == (64,)
        for b in range(a + 1, 16):
            assert np.max(np.abs(encs[a] - encs[b])) > 1e-3, f"steps {a},{b} indistinct"
    again = np.asarray(sinusoidal_step_encoding(3, 64, jnp.float32))
    np.testing.assert_array_equal(encs[3], again)
    far = np.asarray(sinusoidal_step_encoding(100, 64, jnp.float32))
    assert np.isfinite(far).all()


def test_none_time_signal_is_step_blind_and_unbounded():
    """#138: time_signal="none" removes the step signal entirely — the refine
    block conditions only on the state. Step-blindness is structural: the param
    tree must carry NO time-signal parameters (no table, no time_signal_norm),
    so there is no path for "which pass is this" to enter the graph. Like
    sinusoidal, no ceiling: depth 12 on a max_depth=8 model runs without the
    overrun opt-in, and a none model must be strictly smaller than sinusoidal
    (the dropped norm) which is smaller than table (the dropped rows)."""
    def build(sig):
        return CausalRefiner(dim=64, vocab_size=37, num_heads=4,
                             num_encoder_layers=2, max_depth=8, max_seq_len=32,
                             time_signal=sig, rngs=nnx.Rngs(0))
    m = build("none")
    tok = jnp.arange(1, 17)[None, :]
    out = np.asarray(m(tok, depth=12))
    assert out.shape == (1, 16, 37)
    assert np.isfinite(out).all()
    state = nnx.state(m, nnx.Param)
    assert "time_embed" not in state and "time_signal_norm" not in state, \
        "none mode must build no time-signal parameters at all"

    def n_params(mm):
        return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(mm, nnx.Param)))
    assert n_params(build("none")) < n_params(build("sinusoidal")) < n_params(build("table"))
