"""Micro-step training benchmark (PERFORMANCE_PLAN.md rule zero).

Measures steps/sec and peak VRAM for the real compute_grad_step + apply_grads
path on synthetic data. Two modes:
  - loop:   mimics the real train loop, pulling loss/grad-norm/diag floats to
            the host every micro-step (the current per-step sync behavior)
  - kernel: dispatches all steps and syncs once at the end — the upper bound
            P1 (fused step, deferred sync) can reach

Allocator env vars must be set by the caller BEFORE this script runs, e.g.:
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      venv/bin/python tools/bench_train_step.py --depth 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from config import BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE
from trainer import init_model_and_optimizer
from grad_step import compute_grad_step, apply_grads


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
    parser.add_argument("--depth", type=int, default=8, help="reasoning steps (curriculum depth)")
    parser.add_argument("--modes", type=str, default="loop,kernel")
    parser.add_argument("--no-remat-encdec", action="store_true",
                        help="disable per-block remat in encoder/decoder stacks (P5 config c)")
    parser.add_argument("--no-remat-reasoning", action="store_true",
                        help="disable per-block remat in the reasoning stack (P5 config b)")
    parser.add_argument("--attn-impl", type=str, default=None,
                        help="force jax.nn.dot_product_attention implementation (e.g. cudnn)")
    args = parser.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"allocator: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', '(default bfc)')} | "
          f"preallocate: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '(default true)')}")
    print(f"depth: {args.depth} | steps: {args.steps} | warmup: {args.warmup} | "
          f"no_remat_encdec: {args.no_remat_encdec} | no_remat_reasoning: {args.no_remat_reasoning} | "
          f"attn_impl: {args.attn_impl or '(default)'}")

    if args.attn_impl:
        import functools
        _orig = jax.nn.dot_product_attention
        jax.nn.dot_product_attention = functools.partial(_orig, implementation=args.attn_impl)

    model, optimizer = init_model_and_optimizer()
    # Flip remat flags before the first trace; use_remat is read at call time.
    if args.no_remat_encdec:
        model.encoder_stack.use_remat = False
        model.decoder_stack.use_remat = False
    if args.no_remat_reasoning:
        model.reasoning_stack.use_remat = False

    rng = np.random.default_rng(0)
    batch = jnp.array(
        rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32
    )
    no_boundary = jnp.zeros((BATCH_SIZE,), dtype=bool)

    def micro_step(i, sync):
        loss, out, grads, grad_norm = compute_grad_step(
            model, batch, jnp.array(i), args.depth, doc_boundary=no_boundary
        )
        apply_grads(optimizer, grads, model)
        if sync:
            # The real train loop pulls these to the host every micro-step.
            _ = float(loss)
            _ = float(grad_norm)
            _ = float(out.diag.get("temporal_drift", 0.0))
        return loss

    t0 = time.time()
    last = None
    for i in range(args.warmup):
        last = micro_step(i, sync=True)
    jax.block_until_ready(last)
    print(f"warmup (incl. compile): {time.time() - t0:.1f}s")
    report_memory("post-warmup")

    tokens_per_step = BATCH_SIZE * 2 * MAX_SEQ_LEN
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
            f"opt-step (x128): {ms * 128 / 1000:5.2f}s"
        )
    report_memory("post-bench")


if __name__ == "__main__":
    main()
