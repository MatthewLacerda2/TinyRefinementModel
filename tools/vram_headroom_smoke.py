"""Measure real training VRAM at a given model size / batch, to size the run to the GPU.

The question this answers: "how big a model + batch fills ~5 GB of the 6 GB card
without spiking past it?" We can't read that off param counts alone — the optimizer
state, the full-vocab logits, and the backward pass all cost memory. So this builds
the *real* refiner, runs the *real* grad step (`grad_step.compute_grad_step`) on
synthetic random tokens at the true shapes (seq 512, two windows = [B, 1025]) and the
deepest sampled depth, then reads JAX's peak in-use bytes.

No real data needed — VRAM depends on tensor shapes, not token values — so this runs
on the idle GPU while the corpus tokenizes. One config per invocation (a fresh process
per config keeps the peak measurement clean); sweep with a shell loop.

    XLA_PYTHON_CLIENT_PREALLOCATE=false  is set below so peak_bytes_in_use reflects
    actual usage instead of JAX's default 75 % land-grab.

Example:
    ./venv/bin/python tools/vram_headroom_smoke.py --dim 512 --batch 1
    ./venv/bin/python tools/vram_headroom_smoke.py --dim 640 --batch 2 --bf16-mu
"""

import os

# Must precede the jax import: keep the real BFC allocator (so the number reflects
# training) but skip preallocation so the stats track true peak usage.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Deliberately NOT setting FORCE_F32_COMPUTE — we want the real f16 GPU footprint.

import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from config import MAX_SEQ_LEN, VOCAB_SIZE, ACCUMULATION_STEPS
from plan_a_trainer import RefinerForTraining
from grad_step import compute_grad_step, apply_grads


def param_count(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def main():
    ap = argparse.ArgumentParser(description="training VRAM headroom probe")
    ap.add_argument("--dim", type=int, default=512, help="LATENT_DIM (must be divisible by --heads)")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--encoder-layers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=1, help="micro-batch (per accumulation step)")
    ap.add_argument("--depth", type=int, default=8, help="refinement depth; 8 = the deepest sampled, peak memory")
    ap.add_argument("--bf16-mu", action="store_true", help="store Adam's first moment in bf16 (#18)")
    ap.add_argument("--steps", type=int, default=4, help="grad steps to run (reach steady peak incl. opt state)")
    args = ap.parse_args()

    if args.dim % args.heads:
        raise SystemExit(f"--dim {args.dim} not divisible by --heads {args.heads}")

    model = RefinerForTraining(
        args.dim, nnx.Rngs(0), num_heads=args.heads, encoder_layers=args.encoder_layers
    )

    mu_dtype = jnp.bfloat16 if args.bf16_mu else jnp.float32
    chain = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-4, mu_dtype=mu_dtype),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True,
    )
    opt = nnx.Optimizer(model, chain, wrt=nnx.Param)

    # Synthetic batch at the exact training shape: two 512-token windows + 1.
    batch = jax.random.randint(
        jax.random.PRNGKey(0), (args.batch, 2 * MAX_SEQ_LEN + 1), 0, VOCAB_SIZE, dtype=jnp.int32
    )
    doc_boundary = jnp.zeros((args.batch,), dtype=bool)

    dev = jax.devices()[0]
    for s in range(args.steps):
        loss, _out, grads, _gn = compute_grad_step(model, batch, s, args.depth, doc_boundary)
        apply_grads(opt, grads, model)
    loss.block_until_ready()

    stats = dev.memory_stats()
    peak = stats.get("peak_bytes_in_use", 0) / 1e9
    inuse = stats.get("bytes_in_use", 0) / 1e9
    print(
        f"dim={args.dim} heads={args.heads} enc={args.encoder_layers} batch={args.batch} "
        f"depth={args.depth} bf16mu={args.bf16_mu} | params={param_count(model)/1e6:.1f}M "
        f"| peak={peak:.2f}GB inuse={inuse:.2f}GB"
    )


if __name__ == "__main__":
    main()
