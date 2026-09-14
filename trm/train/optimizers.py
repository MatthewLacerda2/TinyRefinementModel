"""The training optimizers: the base pretrain chain and the reduced-LR SFT
variant it hands over to at the phase switch. Both share the same structure
(clip -> AdamW with masked weight decay, bf16 first moment, MultiSteps
accumulation) so the SFT swap changes exactly one thing: the LR schedule.
"""

import gc

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from trm.config import ACCUMULATION_STEPS, MUON_LR_MULT, TRM_OPTIMIZER
from trm.train.accumulate import multi_steps
from trm.train.schedules import learning_schedule, weight_decay_schedule


# The global-norm clip, applied by MultiSteps to the window's MEAN gradient. The
# trainer logs that same mean's norm against it (#180).
CLIP_NORM = 1.0


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
    return optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay_schedule,
        mask=weight_decay_mask,
        # Store Adam's first moment in bf16 (upcast to f32 for the update math).
        # Storage-only, Turing-safe — tensor cores never see bf16. Frees ~2 bytes/
        # param (~0.23GB at dim960), which is exactly what lets the dim960 / 138.7M
        # model fit the 6GB card — it OOMs with f32 moments. The variance estimate
        # (nu) stays f32: bf16 is too coarse near zero there. Verified sound by
        # instruments.bf16_mu_smoke (tracks an f32-mu run to 0.06% of loss).
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


def create_sft_optimizer(model, old_state=None):
    """The SFT-phase optimizer: same chain at 10% LR. Pass `old_state` to
    preserve momentum across the swap. VRAM note (#30): the mid-training call
    site (trainer.train_loop) must stage the moments to HOST before calling —
    building this optimizer allocates fresh full-size mu/nu on the GPU. The
    startup resume path (start_training) may pass device state: nothing else
    holds VRAM yet, so the transient double-state is harmless there."""
    print("📉 Recreating optimizer with LR reduced to 10% for SFT phase...")

    def sft_lr_schedule(step):
        return learning_schedule(step) * 0.1

    new_opt = nnx.Optimizer(model, _make_chain(sft_lr_schedule), wrt=nnx.Param)
    if old_state is not None:
        nnx.update(new_opt, old_state)

    gc.collect()
    return new_opt
