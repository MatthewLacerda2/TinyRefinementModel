"""Micro-step training benchmark: measure a step before optimizing it.

Measures steps/sec and peak VRAM for the real compute_grad_step + apply_grads
path on synthetic data, for the model the trainer builds (MODEL_ARCH). Two modes:
  - loop:   mimics the real train loop, pulling loss/grad-norm/token-loss floats
            to the host every micro-step (the trainer's per-step sync)
  - kernel: dispatches all steps and syncs once at the end — the upper bound a
            fused step with deferred sync could reach

Allocator env vars must be set by the caller BEFORE this script runs, e.g.:
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      venv/bin/python -m instruments.bench_train_step
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from trm.config import ACCUMULATION_STEPS, BATCH_SIZE, MAX_SEQ_LEN, MAX_STEPS_LIMIT, MODEL_ARCH, VOCAB_SIZE
from trm.train.trainer import init_model_and_optimizer
from trm.train.grad_step import compute_grad_step, apply_grads

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "ms/micro-step, tok/s": ("measured", "wall-clock mean over --steps micro-steps after warmup; one run, no spread"),
    "peak / in use MB": ("measured", "memory_stats() high-water mark for the whole process so far, compile included; "
                                     "absent under the platform allocator"),
}


def report_memory(label):
    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        print(f"  [{label}] memory_stats unavailable under this allocator")
        return
    peak = stats.get("peak_bytes_in_use")
    inuse = stats.get("bytes_in_use")
    if peak is not None:
        print(f"  [{label}] peak {peak / 1024**2:.0f} MB | in use {inuse / 1024**2:.0f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60, help="timed micro-steps per mode")
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps (includes compile)")
    parser.add_argument("--depth", type=int, default=MAX_STEPS_LIMIT,
                        help="refinement/reasoning depth; inert for plain, which has no depth dial")
    parser.add_argument("--modes", type=str, default="loop,kernel")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help="micro-batch rows (default: config BATCH_SIZE). Sweeping this "
                             "measures how throughput scales with the forward's GEMM shape — "
                             "the 1->2 rung is what stacking the two windows would buy without "
                             "touching the data pipeline, the 2->4 rung is what real batching adds.")
    parser.add_argument("--attn-impl", type=str, default=None,
                        help="force jax.nn.dot_product_attention implementation (e.g. cudnn)")
    args = parser.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"allocator: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', '(default bfc)')} | "
          f"preallocate: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '(default true)')}")
    print(f"arch: {MODEL_ARCH} | depth: {args.depth} | batch: {args.batch} | steps: {args.steps} | "
          f"warmup: {args.warmup} | attn_impl: {args.attn_impl or '(default)'}")

    if args.attn_impl:
        import functools
        _orig = jax.nn.dot_product_attention
        jax.nn.dot_product_attention = functools.partial(_orig, implementation=args.attn_impl)

    # The trainer's own constructor, so the bench times the model a launch trains.
    model, optimizer = init_model_and_optimizer()

    rng = np.random.default_rng(0)
    batch = jnp.array(
        rng.integers(0, VOCAB_SIZE, size=(args.batch, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32
    )
    no_boundary = jnp.zeros((args.batch,), dtype=bool)

    def micro_step(i, sync):
        loss, out, grads, grad_norm = compute_grad_step(
            model, batch, jnp.array(i), args.depth, doc_boundary=no_boundary
        )
        apply_grads(optimizer, grads, model)
        if sync:
            # The real train loop pulls these to the host every micro-step.
            _ = float(loss)
            _ = float(grad_norm)
            _ = float(out.diag.get("token_loss", loss))
        return loss

    t0 = time.time()
    last = None
    for i in range(args.warmup):
        last = micro_step(i, sync=True)
    jax.block_until_ready(last)
    print(f"warmup (incl. compile): {time.time() - t0:.1f}s")
    report_memory("post-warmup")

    tokens_per_step = args.batch * 2 * MAX_SEQ_LEN
    for mode in args.modes.split(","):
        t0 = time.time()
        for i in range(args.steps):
            last = micro_step(args.warmup + i, sync=(mode == "loop"))
        jax.block_until_ready(last)
        dt = time.time() - t0
        ms = dt / args.steps * 1000
        print(
            f"mode={mode:6} : {ms:7.1f} ms/micro-step | {args.steps / dt:6.2f} steps/s | "
            f"{tokens_per_step * args.steps / dt / 1e3:6.1f}k tok/s | "
            f"opt-step (x{ACCUMULATION_STEPS}): {ms * ACCUMULATION_STEPS / 1000:5.2f}s"
        )
    report_memory("post-bench")


if __name__ == "__main__":
    main()
