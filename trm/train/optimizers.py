"""The training optimizer: clip -> AdamW (or Muon on the matrices) with masked
weight decay and a bf16 first moment, accumulated over ACCUMULATION_STEPS
micro-steps.
"""

import jax
import jax.numpy as jnp
import optax

from trm.config import (
    ACCUMULATION_STEPS,
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    CLIP_NORM,
    MUON_BETA,
    MUON_EPS,
    MUON_LR_MULT,
    MUON_NESTEROV,
    MUON_NS_STEPS,
    TRM_OPTIMIZER,
)
from trm.train.accumulate import multi_steps
from trm.train.schedules import learning_schedule, weight_decay_schedule


def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)


def muon_partition(params):
    """'muon' for every 2-D weight matrix, 'adam' for everything else (#26).

    optax's own default routes every 2-D array to Muon, which sends the tied token
    embedding there too. An embedding table is 50,304 independent lookup rows, not a
    linear map; orthogonalizing its update is wrong, and a wrong partition still
    trains and the loss still falls. Path-aware, so the table stays on Adam.
    """
    def label(path, leaf):
        name = jax.tree_util.keystr(path)
        return "muon" if getattr(leaf, "ndim", 0) == 2 and "embed" not in name else "adam"
    return jax.tree_util.tree_map_with_path(label, params)


def _adamw(learning_rate):
    # Every numeric knob passed by name, none left to optax's defaults (#358).
    return optax.adamw(
        learning_rate=learning_rate,
        b1=ADAM_B1,
        b2=ADAM_B2,
        eps=ADAM_EPS,
        weight_decay=weight_decay_schedule,
        mask=weight_decay_mask,
        # Store Adam's first moment in bf16 (upcast to f32 for the update math).
        # Storage-only, Turing-safe — tensor cores never see bf16. Frees ~2 bytes/
        # param (~0.23GB at dim960), which is exactly what lets the dim960 / 138.7M
        # model fit the 6GB card — it OOMs with f32 moments. The variance estimate
        # (nu) stays f32: bf16 is too coarse near zero there. Verified sound by
        # instruments.bf16_mu_smoke at commit 3859e57 (#37): a RefinerForTraining at
        # dim 512, 16 heads, 7 encoder layers, f32 compute, tracked an f32-mu run to
        # 0.06% of loss. Not re-measured at the dim-960 shipping config.
        mu_dtype=jnp.bfloat16,
    )


def _muon(learning_rate, lr_mult=MUON_LR_MULT):
    """Muon on the matrices at its own LR, the same AdamW as today on the rest.

    Not optax.contrib.muon(): that helper hands ONE learning rate to both
    partitions, and Muon's orthogonalized update needs a far larger one than
    Adam's. mu_dtype is storage-only here exactly as in _adamw — Newton-Schulz
    runs on the f32 bias-corrected momentum (optax/contrib/_muon.py), and the
    cast back to bf16 is for storage.
    """
    return optax.multi_transform(
        {
            "muon": optax.chain(
                optax.contrib.scale_by_muon(
                    # The 2024 coefficients, stated rather than defaulted (#375 asks
                    # whether per-step ones orthogonalize better).
                    ns_coeffs=(3.4445, -4.775, 2.0315),
                    ns_steps=MUON_NS_STEPS,
                    beta=MUON_BETA,
                    eps=MUON_EPS,
                    nesterov=MUON_NESTEROV,
                    mu_dtype=jnp.bfloat16,
                    # Every leaf that reaches this partition is a 2-D matrix
                    # (muon_partition says so); optax needs that stated per leaf,
                    # and a bare None breaks the tree walk against nnx's State.
                    weight_dimension_numbers=lambda updates: jax.tree_util.tree_map(
                        lambda _: optax.contrib.MuonDimensionNumbers(), updates),
                ),
                optax.add_decayed_weights(weight_decay_schedule, mask=weight_decay_mask),
                optax.scale_by_learning_rate(lambda step: learning_rate(step) * lr_mult),
            ),
            "adam": _adamw(learning_rate),
        },
        muon_partition,
    )


def inner_optimizer(learning_rate, kind=TRM_OPTIMIZER):
    return _muon(learning_rate) if kind == "muon" else _adamw(learning_rate)


def _make_chain(learning_rate):
    # LazyMultiSteps, not optax.MultiSteps: the same state and numbers, but the
    # inner optimizer runs once per window instead of on every micro-step
    # (trm/train/accumulate.py) — the ~69 ms/micro-step #24 measured, and the
    # Newton-Schulz-every-micro-step that OOM'd Muon (#26).
    return multi_steps(
        optax.chain(
            optax.clip_by_global_norm(CLIP_NORM),
            inner_optimizer(learning_rate),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True,
    )


optimizer_chain = _make_chain(learning_schedule)

