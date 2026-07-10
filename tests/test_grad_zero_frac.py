"""The underflow instrument (#82): per-group zero-gradient fractions.

f16 underflow rounds gradients to exact zeros silently — no NaN, no abort, just
a plateau — so the instrument counts exact zeros per top-level param group.
Hand-built grad trees pin the arithmetic (known zero counts per group) and the
grouping rules; one real grad step through the production adapter pins the
group naming for the arch we'd actually ship, and demonstrates the
pre-registered caveat: embedding-style groups are legitimately mostly zero
(rows for absent tokens/depths), while dense groups sit at ~0 on the f32 CPU
lane where underflow cannot occur.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from grad_step import compute_grad_step, dense_zero_frac_max, grad_zero_fractions


def _fracs(tree):
    return {k: float(v) for k, v in grad_zero_fractions(tree).items()}


def test_known_zero_counts_per_group():
    tree = {
        "embed": {"embedding": jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])},  # 3 of 5
        "attn": {
            "kernel": jnp.array([[1.0, 0.0], [3.0, 4.0]]),  # 1 of 4
            "bias": jnp.array([0.0, 5.0]),                  # 1 of 2 -> group 2 of 6
        },
    }
    fracs = _fracs(tree)
    assert set(fracs) == {"embed", "attn"}
    assert fracs["embed"] == pytest.approx(3 / 5)
    assert fracs["attn"] == pytest.approx(2 / 6)


def test_single_child_wrappers_are_stripped():
    """The refiner adapter holds all params under one 'refiner' attribute; the
    groups must be its children, not one useless mega-group."""
    inner = {
        "embed": jnp.zeros(4),
        "mlp": jnp.ones(4),
    }
    assert _fracs({"model": inner}) == _fracs(inner)
    assert set(_fracs({"wrapper": {"model": inner}})) == {"embed", "mlp"}


def test_list_entries_aggregate_into_one_group():
    tree = {
        "blocks": [jnp.zeros(4), jnp.ones(4)],  # 4 of 8 across the stack
        "norm": jnp.ones(3),
    }
    fracs = _fracs(tree)
    assert fracs["blocks"] == pytest.approx(0.5)
    assert fracs["norm"] == 0.0


def test_dense_max_excludes_embedding_style_groups():
    fracs = {"embed": 0.95, "time_embed": 0.7, "attn": 0.12, "norm": 0.0}
    assert dense_zero_frac_max(fracs) == pytest.approx(0.12)


def test_real_adapter_groups_and_interpretation_caveats():
    """One production grad step on a tiny refiner pins the group naming and the
    interpretation caveats the per-group split exists for:
      - the tied token embedding reads 0.0 — EVERY vocab row gets gradient
        through the CE head projection, so #82's 'absent rows' caveat applies
        only to the untied time_embed, whose unsampled-depth rows are zero;
      - at init the zero-init down_proj blocks all gradient to gate/up_proj, so
        the block groups carry a large *structural* zero fraction that must
        vanish once an optimizer update lands — zeros the reading has to
        attribute to structure, not underflow.
    f32 CPU lane throughout: underflow itself cannot occur here."""
    import optax
    from config import MAX_SEQ_LEN
    from grad_step import apply_grads
    from plan_a_trainer import RefinerForTraining

    vocab = 5000  # far more tokens than the ~60 the batch uses
    m = RefinerForTraining(64, nnx.Rngs(0), vocab_size=vocab, num_heads=4,
                           encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN)
    rng = np.random.default_rng(3)
    batch = jnp.asarray(rng.integers(1, 60, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))

    _, _, grads, _ = compute_grad_step(m, batch, jnp.array(1), 2)
    fracs = _fracs(grads)

    assert set(fracs) == {
        "embed", "time_embed", "encoder", "refine_block",
        "time_norm", "time_signal_norm", "out_norm", "gate",
    }
    assert fracs["embed"] == 0.0                              # tied head: all rows graded
    assert fracs["time_embed"] == pytest.approx(7 / 9, rel=1e-4)  # depth 2 of max_depth+1 rows
    assert fracs["encoder"] == pytest.approx(fracs["refine_block"])
    assert fracs["encoder"] > 0.4, "gate/up_proj should sit structurally zero behind down_proj == 0"
    for group in ("time_norm", "time_signal_norm", "out_norm", "gate"):
        assert fracs[group] == 0.0

    # One real update moves down_proj off zero; the structural zeros must clear.
    opt = nnx.Optimizer(m, optax.adamw(1e-2), wrt=nnx.Param)
    apply_grads(opt, grads, m)
    _, _, grads, _ = compute_grad_step(m, batch, jnp.array(2), 2)
    fracs = _fracs(grads)
    assert dense_zero_frac_max(fracs) < 0.05, "structural zeros must vanish after an update"
