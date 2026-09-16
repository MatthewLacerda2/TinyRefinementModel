"""Real-config GPU smoke, f16 path, for the architecture a launch would train.

What the CPU suite can't cover: CPU XLA cannot lower the f16-with-f32-accumulation
matmuls (config.py), so the real numerical risk — f16 underflow/overflow — and the
6GB VRAM fit only show up on the GPU. This drives the *production* grad step
(compute_grad_step + apply_grads) at LATENT_DIM/VOCAB_SIZE/MAX_SEQ_LEN, asserts finite
loss + finite, nonzero grads AND f16 activation headroom, then runs a few optimizer
steps.

`--arch` defaults to MODEL_ARCH. `plain` traces every block of its stack; `refiner`
traces its encoder (where #235 overflowed) and also sweeps depths 1..MAX_STEPS_LIMIT
through its unrolled refine loop. The reasoner has no trace here and is refused by
name. The file keeps its historical name because the doctrine and findings cite it.

**Token content is NOT irrelevant, and this file used to say it was.** The overflow
that eventually mattered (#229 -> #235) is corpus-specific: the encoder's output
reaches 65,120 on code against 47.7 on prose, with an f16 ceiling of 65,504. Random
tokens produce prose-like magnitudes, so this smoke ran clean through an entire
10-day run while the model trained itself to 0.6% of the ceiling.

Two consequences, both fixed here:

- with DATA_ROOT set the smoke reads REAL tokens and prefers **code**, the
  distribution that actually stresses activations; random tokens stay the fallback so
  the no-corpus path still works.
- finiteness is not a sufficient assertion. A model at 99.4% of the ceiling is finite
  and passes -- the champion passes today. So the smoke now measures peak activation
  as a fraction of the f16 max and fails below a headroom margin. That is the
  difference between a gate and a post-mortem.

Also reads the underflow instrument (#82) on every grad step: per-group zero-gradient
fractions. Embedding rows for absent tokens are legitimately zero; the dense groups
should sit at ~0 — elevated means f16 underflow, and loss scaling is the named fix.

    venv/bin/python -m instruments.smoke_refiner_gpu
"""

import os
# Match the real run's allocator arena (trm.train.start) — the smoke must see the
# same VRAM budget training does, not JAX's smaller 0.75 default.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from instruments.arch import add_arch_argument, build as arch_build
from trm.config import LATENT_DIM, MAX_SEQ_LEN, MAX_STEPS_LIMIT, VOCAB_SIZE
from trm.train.grad_step import compute_grad_step, apply_grads, grad_zero_fractions, dense_zero_frac_max
from trm.train.optimizers import optimizer_chain

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "f16 headroom": ("sampled", "worst |activation| over the probed rows and depths only, against the f16 max"),
    "loss, grad_norm": ("measured", "per probed step; checks finiteness, not quality"),
}


# f16's largest finite value. The margin is what the smoke demands is left unused.
F16_MAX = 65504.0
# The champion sits at 0.6% headroom and is one rounding from #229's whole-window
# NaN. 50% (a factor of two) is the smallest bar that would have caught it while
# leaving ordinary activation growth alone -- prose runs at 47.7, six orders below.
MIN_HEADROOM = 0.50


TRACED_ARCHES = ("plain", "refiner")


def refuse_untraced(arch):
    """The reasoner is a frozen control with no stack this smoke knows how to trace.
    Refusing by name beats an AttributeError halfway through a 138M build."""
    if arch not in TRACED_ARCHES:
        raise SystemExit(f"smoke_refiner_gpu has no f16 activation trace for arch {arch!r}; "
                         f"it supports {' and '.join(TRACED_ARCHES)}")


def traced_stack(model, arch):
    """(embedding, the blocks whose activations are traced) for `arch`.

    `plain` is its whole stack. The refiner's is its encoder, the stack #235 found at
    99.4% of the f16 ceiling; its shared refine block is exercised by the grad steps.
    """
    refuse_untraced(arch)
    if arch == "plain":
        return model.embed, list(model.blocks)
    refiner = model.refiner  # ARCH-SPECIFIC: refiner — the encoder lives inside the CausalRefiner
    return refiner.embed, list(refiner.encoder)


def residual_blocks(model, arch):
    """Every block whose zero-initialized down_proj the smoke wakes before reading
    zero-fractions: the traced stack, plus the refiner's shared refine block."""
    _, blocks = traced_stack(model, arch)
    if arch == "refiner":
        blocks.append(model.refiner.refine_block)  # ARCH-SPECIFIC: refiner — its one shared block
    return blocks


def block_headroom(model, tokens, arch):
    """Peak |activation| after each traced block, and the f16 headroom it leaves.

    Reported per block because #235's growth was not gradual: six blocks behaved
    identically on both corpora and the seventh multiplied by ~794. A single
    end-of-stack number says a run is unsafe; the per-block trace says where.
    """
    embed, blocks = traced_stack(model, arch)
    pad_bias = (((tokens != model.pad_token_id).astype(jnp.float32) - 1.0) * 1e9)[:, None, None, :]
    z = embed(tokens)
    peaks = []
    for blk in blocks:
        z = blk(z, pad_bias)
        peaks.append(float(jnp.max(jnp.abs(z.astype(jnp.float32)))))
    worst = max(peaks)
    return peaks, worst, 1.0 - worst / F16_MAX


def load_batch(rng):
    """Real tokens when a corpus is reachable, preferring code.

    Random ids are drawn uniformly over the vocabulary, which is nothing like text
    and -- crucially -- nothing like the code that actually drives the encoder to
    the f16 ceiling. Falling back to random keeps the no-corpus path (CI, a fresh
    clone) working, but it is the weaker test and says so.
    """
    root = os.environ.get("DATA_ROOT", "")
    if root:
        from trm.config import resolve_root
        from trm.train.validation import VAL_SKIP_SAMPLES, read_heldout_rows
        for source in ("codeparrot", "fineweb-edu"):
            path = f"{resolve_root(root)}/pretrain/{source}"
            if not os.path.isdir(path):
                continue
            rows = read_heldout_rows(path, 1, VAL_SKIP_SAMPLES)
            if rows:
                (row,) = rows
                print(f"📚 real tokens from {source} (the distribution that stresses f16)")
                return jnp.asarray(row[:, :2 * MAX_SEQ_LEN + 1].astype(np.int32))
    print("🎲 random tokens — DATA_ROOT unset, so the corpus-specific overflow "
          "(#235) CANNOT be caught by this run")
    return jnp.asarray(rng.integers(1, VOCAB_SIZE, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="real-config GPU smoke, f16 path")
    add_arch_argument(ap)
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    assert jax.default_backend() == "gpu", "smoke must run on GPU (unset JAX_PLATFORMS / FORCE_F32_COMPUTE)"

    refuse_untraced(args.arch)  # before building 138M params for nothing
    model = arch_build(args.arch, dim=LATENT_DIM, seed=42)
    n = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"📐 {args.arch}: {n / 1e6:.2f}M params")
    # Optimizer state (Adam m+v, MultiSteps grad accumulator) allocated up front, as
    # in training — the peak that matters is grad step + resident optimizer state.
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    # Wake the zero-init residual path before measuring zero-fracs: at init,
    # down_proj == 0 blocks all gradient to gate/up_proj, so nearly half of each
    # block group reads exactly zero for structural reasons (pinned in
    # tests/apparatus/test_grad_zero_frac.py) — and MultiSteps lands no real update within
    # this smoke to clear it. Scale 0.02 puts down_proj where early training
    # does, the regime the underflow reading has to certify. Finiteness and
    # VRAM-fit checks are unaffected.
    key = jax.random.PRNGKey(1)
    for blk in residual_blocks(model, args.arch):
        key, sub = jax.random.split(key)
        kernel = blk.down_proj.kernel[...]
        blk.down_proj.kernel[...] = 0.02 * jax.random.normal(sub, kernel.shape, kernel.dtype)

    rng = np.random.default_rng(0)
    batch = load_batch(rng)

    # Headroom BEFORE the grad steps: this is a property of the weights and the
    # tokens, and reading it first means a doomed run is refused before it spends
    # anything. Measured on one window, the shape the stack actually sees.
    peaks, worst, headroom = block_headroom(model, batch[:, :MAX_SEQ_LEN], args.arch)
    stack = "block" if args.arch == "plain" else "encoder block"
    print(f"📏 activation peak per {stack}: "
          + "  ".join(f"{v:,.0f}" for v in peaks))
    print(f"   worst {worst:,.1f} of f16 max {F16_MAX:,.0f} → headroom {headroom:.1%}")
    assert headroom >= MIN_HEADROOM, (
        f"{args.arch} activations reach {worst:,.0f}, leaving {headroom:.1%} of f16 headroom "
        f"(need {MIN_HEADROOM:.0%}). This is #235: the residual branches are unbounded, "
        f"and #229's whole-window NaN is what running out looks like. POST_NORM=1 bounds "
        f"the branch outputs.")

    # Worst-case VRAM and the deepest f16 unroll first: if depth-8 fits with the
    # optimizer resident, every shallower depth training samples does too. `plain`
    # has no depth dial, so for it this is simply the grad step. Loss is expected to
    # stay flat — optimizer_chain is MultiSteps, so no real update lands in a handful
    # of steps; this checks fit + finiteness, not descent.
    def read_zero_fracs(grads):
        zf = {k: float(v) for k, v in grad_zero_fractions(grads).items()}
        dmax = dense_zero_frac_max(zf)
        print("    zero-frac (#82): dense max=" + f"{dmax:.4f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in zf.items()))
        return dmax

    worst_dense = 0.0

    print(f"— grad steps at depth {MAX_STEPS_LIMIT} (worst case; inert for plain) with optimizer resident —")
    for s in range(1, 4):
        loss, _, grads, gnorm = compute_grad_step(model, batch, jnp.array(s), MAX_STEPS_LIMIT)
        loss_f, grad_f = float(loss), float(gnorm)
        ok = math.isfinite(loss_f) and math.isfinite(grad_f) and grad_f > 0
        print(f"  step {s}: loss={loss_f:.4f}  grad_norm={grad_f:.4f}  {'OK' if ok else '✗ NON-FINITE/ZERO'}")
        assert ok, f"depth {MAX_STEPS_LIMIT} step {s} non-finite/zero in f16"
        worst_dense = max(worst_dense, read_zero_fracs(grads))
        apply_grads(optimizer, grads, model)

    # Shallow depths are a different unroll only for an arch that has one.
    shallow = () if args.arch == "plain" else (1, 4)
    if shallow:
        print("— finiteness at shallow depths 1 and 4 —")
    for depth in shallow:
        loss, _, grads, gnorm = compute_grad_step(model, batch, jnp.array(1), depth)
        ok = math.isfinite(float(loss)) and math.isfinite(float(gnorm))
        print(f"  depth {depth}: loss={float(loss):.4f}  grad_norm={float(gnorm):.4f}  {'OK' if ok else '✗'}")
        assert ok
        worst_dense = max(worst_dense, read_zero_fracs(grads))

    print(f"(loss ≈ 2·ln(vocab) = {2 * math.log(VOCAB_SIZE):.2f} at init, both windows summed)")
    # Loud reading, not an assert: the #82 decision rule ("~0 through the smoke
    # and the first base-run stretch") is applied in the finding, and elevated
    # means "file loss-scaling adoption", not "this smoke is broken". 0.05 is
    # only the warn-loudly heuristic.
    if worst_dense > 0.05:
        print(f"⚠️ dense-kernel zero-fraction reached {worst_dense:.4f} — possible f16 "
              f"underflow; per #82, file the loss-scaling adoption issue.")
    else:
        print(f"🧊 dense-kernel zero-fraction max {worst_dense:.4f} — no underflow signal (#82).")
    print(f"✅ GPU smoke passed: {args.arch} f16 path numerically healthy AND fits in 6GB with optimizer state.")


if __name__ == "__main__":
    main()
