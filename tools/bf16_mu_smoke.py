"""Correctness smoke for #18 (store Adam's first moment in bf16).

The VRAM win is already measured (it's what lets dim960 fit). The open question #18
names is "optax state-dtype handling": does optax actually keep mu in bf16, upcast it
for the update math, and still train *soundly* — or does the coarser moment quietly
derail the loss? bf16-mu is now load-bearing for the base run, so we check it head-on.

Two runs, identical seed and identical batches of real r50k tokens, differing only in
mu_dtype (f32 vs bf16). If bf16-mu is sound, its loss trajectory tracks f32-mu closely
(small, non-growing gap). We also walk the optimizer state and confirm mu is really
stored as bfloat16 while the variance (nu) stays f32.

    PYTHONPATH=. ./venv/bin/python tools/bf16_mu_smoke.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# On-demand allocator: frees the first run's buffers before the second run, and avoids
# the BFC fragmentation false-OOM that bites when two models live in one process.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# f32 compute so the f16 forward-pass overflow (real on a fresh-init Turing model) can't
# masquerade as an optimizer problem — this A/B is about mu *storage*, nothing else.
os.environ["FORCE_F32_COMPUTE"] = "1"

import argparse
import glob

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from config import MAX_SEQ_LEN
from plan_a_trainer import RefinerForTraining
from grad_step import compute_grad_step, apply_grads


def build_optimizer(model, mu_dtype, lr):
    # Plain adamw that updates EVERY step — no MultiSteps wrapper. Accumulation only
    # controls *when* updates land (orthogonal to mu storage); dropping it means every
    # step exercises the bf16 mu, which is the only thing this smoke is testing.
    chain = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, mu_dtype=mu_dtype),
    )
    return nnx.Optimizer(model, chain, wrt=nnx.Param)


def dtype_histogram(opt):
    """Count float leaves by dtype in the optimizer state — proves where bf16 lives."""
    hist = {}
    for leaf in jax.tree_util.tree_leaves(nnx.state(opt)):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            hist[str(leaf.dtype)] = hist.get(str(leaf.dtype), 0) + 1
    return hist


def load_batches(dim_stride, n_seqs):
    """First n_seqs sequences from the first real r50k chunk, shaped [n_seqs, stride]."""
    chunks = sorted(glob.glob("runs/data/pretrain/fineweb-edu/chunk_*.npy"))
    if not chunks:
        raise SystemExit("no r50k chunks on disk yet — run prefill first")
    flat = np.load(chunks[0], mmap_mode="r")
    need = n_seqs * dim_stride
    seqs = np.asarray(flat[:need], dtype=np.int32).reshape(n_seqs, dim_stride)
    return jnp.asarray(seqs)


def run(mu_dtype, batches, depth, steps, batch, lr):
    """Fresh model+opt at a fixed seed; same batches every call → only mu_dtype differs."""
    model = RefinerForTraining(512, nnx.Rngs(0), num_heads=16, encoder_layers=7)
    opt = build_optimizer(model, mu_dtype, lr)
    doc_boundary = jnp.zeros((batch,), dtype=bool)
    losses = []
    for s in range(steps):
        b = batches[s * batch:(s + 1) * batch]
        loss, _o, grads, _gn = compute_grad_step(model, b, s, depth, doc_boundary)
        apply_grads(opt, grads, model)
        losses.append(float(loss))
    return losses, dtype_histogram(opt)


def main():
    ap = argparse.ArgumentParser(description="#18 bf16-mu correctness smoke")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6, help="fixed refinement depth for a clean A/B")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    stride = 2 * MAX_SEQ_LEN + 1
    batches = load_batches(stride, args.steps * args.batch)

    f32_losses, f32_hist = run(jnp.float32, batches, args.depth, args.steps, args.batch, args.lr)
    bf16_losses, bf16_hist = run(jnp.bfloat16, batches, args.depth, args.steps, args.batch, args.lr)

    finite = all(np.isfinite(bf16_losses))
    has_bf16 = any("bfloat16" in d for d in bf16_hist)
    f32_clean = not any("bfloat16" in d for d in f32_hist)
    # Compare the back half (after both have moved off the shared init).
    half = args.steps // 2
    gap = np.mean(np.abs(np.array(bf16_losses[half:]) - np.array(f32_losses[half:])))
    rel = gap / np.mean(np.array(f32_losses[half:]))

    print(f"\nstep   f32_mu    bf16_mu")
    for s in range(0, args.steps, max(1, args.steps // 10)):
        print(f"{s:4d}  {f32_losses[s]:7.4f}   {bf16_losses[s]:7.4f}")
    print(f"{args.steps-1:4d}  {f32_losses[-1]:7.4f}   {bf16_losses[-1]:7.4f}")

    print(f"\noptimizer-state dtypes  f32-run: {f32_hist}")
    print(f"optimizer-state dtypes  bf16-run: {bf16_hist}")
    print(f"\nbf16 loss finite: {finite}")
    print(f"bf16 run stores mu in bf16: {has_bf16}    f32 run is all-f32: {f32_clean}")
    print(f"mean |loss gap| over back half: {gap:.4f}  ({rel*100:.2f}% of f32 loss)")
    verdict = finite and has_bf16 and f32_clean and rel < 0.02
    print(f"\nVERDICT: {'PASS — bf16-mu tracks f32 and is stored as bf16' if verdict else 'CHECK — see numbers above'}")


if __name__ == "__main__":
    main()
