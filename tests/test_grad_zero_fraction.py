"""Underflow instrument (#82): f16 gradient underflow rounds entries to
exactly zero without ever going non-finite, so it's invisible to the
NaN-streak abort and to the global grad norm. `zero_fraction_by_group` is the
guard — a hand-built grad tree with known zero counts per top-level group, so
the fraction math is pinned independent of any real model."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from grad_step import zero_fraction_by_group


def test_zero_fraction_per_group_matches_hand_count():
    grads = {
        # 3 of 4 entries zero.
        "embed": {"embedding": jnp.array([0.0, 0.0, 0.0, 1.0])},
        # 1 of 4 entries zero, split across two leaves in the same group.
        "encoder_stack": {
            "w": jnp.array([[1.0, 2.0], [0.0, 3.0]]),
        },
    }

    fractions = zero_fraction_by_group(grads)

    assert set(fractions.keys()) == {"embed", "encoder_stack"}
    assert float(fractions["embed"]) == 0.75
    assert float(fractions["encoder_stack"]) == 0.25


def test_zero_fraction_sums_leaves_within_a_group():
    # Two leaves in the same top-level group must pool into one fraction,
    # not report per-leaf independently.
    grads = {
        "block": {
            "attn": {"q": jnp.zeros((4,))},
            "mlp": {"up": jnp.ones((4,))},
        },
    }

    fractions = zero_fraction_by_group(grads)

    assert set(fractions.keys()) == {"block"}
    assert float(fractions["block"]) == 0.5


def test_all_nonzero_group_reads_zero():
    grads = {"norm": {"scale": jnp.array([1.0, 2.0, 3.0])}}
    fractions = zero_fraction_by_group(grads)
    assert float(fractions["norm"]) == 0.0
