"""The training optimizer: clip -> AdamW (or Muon on the matrices) with masked
weight decay and a bf16 first moment, accumulated over ACCUMULATION_STEPS
micro-steps. Every knob comes from the run's Config, passed in (#475).
"""

import jax
import jax.numpy as jnp
import optax

from trm.train.accumulate import multi_steps


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


def _weight_decay(config):
    return optax.constant_schedule(config.WEIGHT_DECAY)


def _adamw(config, learning_rate):
    # Every numeric knob passed by name, none left to optax's defaults (#358).
    return optax.adamw(
        learning_rate=learning_rate,
        b1=config.ADAM_B1,
        b2=config.ADAM_B2,
        eps=config.ADAM_EPS,
        weight_decay=_weight_decay(config),
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


def _muon(config, learning_rate, lr_mult=None):
    """Muon on the matrices at its own LR, the same AdamW as today on the rest.

    Not optax.contrib.muon(): that helper hands ONE learning rate to both
    partitions, and Muon's orthogonalized update needs a far larger one than
    Adam's. mu_dtype is storage-only here exactly as in _adamw — Newton-Schulz
    runs on the f32 bias-corrected momentum (optax/contrib/_muon.py), and the
    cast back to bf16 is for storage. `lr_mult` overrides config.MUON_LR_MULT.
    """
    lr_mult = config.MUON_LR_MULT if lr_mult is None else lr_mult
    return optax.multi_transform(
        {
            "muon": optax.chain(
                optax.contrib.scale_by_muon(
                    # The 2024 coefficients, stated rather than defaulted (#375 asks
                    # whether per-step ones orthogonalize better).
                    ns_coeffs=(3.4445, -4.775, 2.0315),
                    ns_steps=config.MUON_NS_STEPS,
                    beta=config.MUON_BETA,
                    eps=config.MUON_EPS,
                    nesterov=config.MUON_NESTEROV,
                    mu_dtype=jnp.bfloat16,
                    # Every leaf that reaches this partition is a 2-D matrix
                    # (muon_partition says so); optax needs that stated per leaf,
                    # and a bare None breaks the tree walk against nnx's State.
                    weight_dimension_numbers=lambda updates: jax.tree_util.tree_map(
                        lambda _: optax.contrib.MuonDimensionNumbers(), updates),
                ),
                optax.add_decayed_weights(_weight_decay(config), mask=weight_decay_mask),
                optax.scale_by_learning_rate(lambda step: learning_rate(step) * lr_mult),
            ),
            "adam": _adamw(config, learning_rate),
        },
        muon_partition,
    )


def inner_optimizer(config, learning_rate):
    """The optimizer config.TRM_OPTIMIZER names, without the clip or the accumulation."""
    return _muon(config, learning_rate) if config.TRM_OPTIMIZER == "muon" else _adamw(config, learning_rate)


def optimizer_chain(config, learning_rate):
    """The training optimizer at `learning_rate` (the run's: Schedules.of(config).learning_rate)."""
    # LazyMultiSteps, not optax.MultiSteps: the same state and numbers, but the
    # inner optimizer runs once per window instead of on every micro-step
    # (trm/train/accumulate.py) — the ~69 ms/micro-step #24 measured, and the
    # Newton-Schulz-every-micro-step that OOM'd Muon (#26).
    return multi_steps(
        optax.chain(
            optax.clip_by_global_norm(config.CLIP_NORM),
            inner_optimizer(config, learning_rate),
        ),
        every_k_schedule=config.ACCUMULATION_STEPS,
        use_grad_mean=True,
    )

