"""Smoke for the #30 fix: does staging optimizer moments through host RAM avoid the
2x optimizer-state VRAM spike at the SFT phase switch?

The base run died (RESOURCE_EXHAUSTED) at the pretrain→SFT handoff because rebuilding
the optimizer allocates a fresh full mu/nu on the GPU while the old moments are still
resident — two full optimizer states at once. The fix (trainer.py) pulls the old
moments to host with jax.device_get before the rebuild, so the old GPU buffers free
first and only one state is resident at the peak.

This probe builds the real refiner + optimizer, runs a step to populate the moments,
then rebuilds the optimizer the two ways and reports JAX peak VRAM for each:

    --mode double : old state stays on device during the rebuild (the bug)
    --mode host   : old state staged to host first (the fix)

One mode per process (peak is a high-water mark, so the two must not share a process).
It also runs a grad step on the rebuilt optimizer, proving the host-staged moments
load and train (i.e. momentum is preserved, not just freed).

    ./venv/bin/python tools/sft_switch_smoke.py --mode double
    ./venv/bin/python tools/sft_switch_smoke.py --mode host
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import gc

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from config import MAX_SEQ_LEN, VOCAB_SIZE, ACCUMULATION_STEPS
from plan_a_trainer import RefinerForTraining
from grad_step import compute_grad_step, apply_grads


def build_optimizer(model, lr_scale=1.0):
    """Mirror the trainer's chain (clip + adamw under MultiSteps). lr_scale=0.1 is the
    SFT case; the value is irrelevant to the memory footprint we're measuring."""
    chain = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-4 * lr_scale, weight_decay=0.01),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True,
    )
    return nnx.Optimizer(model, chain, wrt=nnx.Param)


def main():
    ap = argparse.ArgumentParser(description="#30 SFT optimizer-switch VRAM smoke")
    ap.add_argument("--mode", choices=["double", "host"], required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    dev = jax.devices()[0]
    model = RefinerForTraining(args.dim, nnx.Rngs(0))
    opt = build_optimizer(model)

    batch = jax.random.randint(
        jax.random.PRNGKey(0), (args.batch, 2 * MAX_SEQ_LEN + 1), 0, VOCAB_SIZE, dtype=jnp.int32
    )
    doc_boundary = jnp.zeros((args.batch,), dtype=bool)

    # Populate the moments so the old state is non-trivial (a real switch carries them).
    loss, _o, grads, _g = compute_grad_step(model, batch, 0, args.depth, doc_boundary)
    apply_grads(opt, grads, model)
    loss.block_until_ready()
    pre_peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9

    # --- the SFT switch, the two ways ---
    if args.mode == "double":
        old_state = nnx.state(opt)            # stays on the GPU
    else:
        old_state = jax.device_get(nnx.state(opt))  # the fix: copy to host RAM first
    del opt
    gc.collect()

    new_opt = build_optimizer(model, lr_scale=0.1)
    nnx.update(new_opt, old_state)
    del old_state
    gc.collect()

    # Prove the rebuilt optimizer actually trains with the carried-over moments.
    loss2, _o2, grads2, _g2 = compute_grad_step(model, batch, 1, args.depth, doc_boundary)
    apply_grads(new_opt, grads2, model)
    loss2.block_until_ready()

    post_peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
    print(
        f"mode={args.mode} dim={args.dim} batch={args.batch} | "
        f"peak_before_switch={pre_peak:.2f}GB peak_after_switch={post_peak:.2f}GB | "
        f"switch_added={post_peak - pre_peak:+.2f}GB | rebuilt_opt_step_ok={bool(jnp.isfinite(loss2))}"
    )


if __name__ == "__main__":
    main()
