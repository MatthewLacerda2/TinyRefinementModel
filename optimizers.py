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

from config import ACCUMULATION_STEPS
from schedules import learning_schedule, weight_decay_schedule


def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)


def _make_chain(learning_rate):
    return optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=weight_decay_schedule,
                mask=weight_decay_mask,
                # Store Adam's first moment in bf16 (upcast to f32 for the update math).
                # Storage-only, Turing-safe — tensor cores never see bf16. Frees ~2 bytes/
                # param (~0.23GB at dim960), which is exactly what lets the dim960 / 138.7M
                # model fit the 6GB card — it OOMs with f32 moments. The variance estimate
                # (nu) stays f32: bf16 is too coarse near zero there. Verified sound by
                # tools/bf16_mu_smoke.py (tracks an f32-mu run to 0.06% of loss).
                mu_dtype=jnp.bfloat16,
            ),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True,
    )


optimizer_chain = _make_chain(learning_schedule)


def create_sft_optimizer(model, old_state=None):
    """The SFT-phase optimizer: same chain at 10% LR. Pass `old_state` (moments
    pulled to HOST first — see the #30 OOM note at the call sites) to preserve
    momentum across the swap."""
    print("📉 Recreating optimizer with LR reduced to 10% for SFT phase...")

    def sft_lr_schedule(step):
        return learning_schedule(step) * 0.1

    new_opt = nnx.Optimizer(model, _make_chain(sft_lr_schedule), wrt=nnx.Param)
    if old_state is not None:
        nnx.update(new_opt, old_state)

    gc.collect()
    return new_opt
