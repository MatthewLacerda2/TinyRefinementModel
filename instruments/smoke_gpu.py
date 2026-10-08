"""Real-config GPU smoke, f16 path, for the model a launch would train.

What the CPU suite can't cover: CPU XLA cannot lower the f16-with-f32-accumulation
matmuls (config.py), so the real numerical risk — f16 underflow/overflow — and the
6GB VRAM fit only show up on the GPU. This drives the *production* grad step
(compute_grad_step + apply_grads) at LATENT_DIM/VOCAB_SIZE/MAX_SEQ_LEN, asserts finite
loss + finite, nonzero grads AND f16 activation headroom, then runs a few optimizer
steps.

It traces every block of the stack. Older findings cite it by its former name,
`smoke_refiner_gpu` (#292).

With DATA_ROOT set the smoke reads real tokens and prefers **code**, the distribution
that stresses activations; random tokens are the no-corpus fallback. Beyond non-finite
values, it fails when peak |activation| exceeds (1 - MIN_HEADROOM) x F16_MAX, i.e. when
less than MIN_HEADROOM (currently 0.50) of the f16 range is left unused.
Why both (the overflow is corpus-specific, and finiteness passed a model at 99.4% of
the ceiling): `tests/apparatus/test_smoke_headroom.py` and #235.

Also reads the underflow instrument (#82) on every grad step: per-group zero-gradient
fractions. Embedding rows for absent tokens are legitimately zero; the dense groups
should sit at ~0 — elevated means f16 underflow, and loss scaling is the named fix.

    venv/bin/python -m instruments.smoke_gpu
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

from instruments._common import F16_MAX, param_count
from trm.config import VOCAB_SIZE
from trm.model import build_model
from trm.train.grad_step import compute_grad_step, apply_grads, grad_zero_fractions, dense_zero_frac_max
from trm.settings import CONFIG
from trm.train.optimizers import optimizer_chain
from trm.train.schedules import Schedules

# A dense kernel's zero-gradient fraction above this prints a warning (#82); not a gate.
DENSE_ZERO_WARN = 0.05

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "f16 headroom": ("sampled", "worst |activation| over the probed rows only, against the f16 max"),
    "loss, grad_norm": ("measured", "per probed step; checks finiteness, not quality"),
}


# The margin is what the smoke demands is left unused of F16_MAX.
# The champion sits at 0.6% headroom and is one rounding from #229's whole-window
# NaN. 50% (a factor of two) is the smallest bar that would have caught it while
# leaving ordinary activation growth alone -- prose runs at 47.7, six orders below.
MIN_HEADROOM = 0.50


def residual_blocks(model):
    """Every block whose zero-initialized down_proj the smoke wakes before reading
    zero-fractions: the whole stack, the same blocks the headroom trace reads."""
    return list(model.blocks)


def block_headroom(model, tokens):
    """Peak |activation| in each traced block, and the f16 headroom it leaves.

    Reported per block because #235's growth was not gradual: six blocks behaved
    identically on both corpora and the seventh multiplied by ~794. A single
    end-of-stack number says a run is unsafe; the per-block trace says where.
    A block's peak is the larger of its stream output and its two branch outputs
    (#536): with an f32 stream (#357) the branches are what runs in f16.
    """
    pad_bias = (((tokens != model.pad_token_id).astype(jnp.float32) - 1.0) * 1e9)[:, None, None, :]
    z = model.embed(tokens)
    peaks = []
    for blk in residual_blocks(model):
        z, branches = blk(z, pad_bias, probe=True)
        peaks.append(max(float(jnp.max(jnp.abs(z.astype(jnp.float32)))), float(jnp.max(branches))))
    worst = max(peaks)
    return peaks, worst, 1.0 - worst / F16_MAX


def load_batch(rng):
    """Real tokens when a corpus is reachable, preferring code.

    Random ids are drawn uniformly over the vocabulary, which is nothing like text
    and -- crucially -- nothing like the code that actually drives the stack to
    the f16 ceiling. Falling back to random keeps the no-corpus path (CI, a fresh
    clone) working, but it is the weaker test and says so.
    """
    root = os.environ.get("DATA_ROOT", "")
    if root:
        from trm.config import resolve_root
        from trm.train.validation import read_heldout_rows
        for source in ("codeparrot", "fineweb-edu"):
            path = f"{resolve_root(root)}/pretrain/{source}"
            if not os.path.isdir(path):
                continue
            rows = read_heldout_rows(path, 1, CONFIG.VAL_SKIP_SAMPLES, max_seq_len=CONFIG.MAX_SEQ_LEN,
                                     data_seed=CONFIG.DATA_SEED)
            if rows:
                (row,) = rows
                print(f"📚 real tokens from {source} (the distribution that stresses f16)")
                return jnp.asarray(row[:, :2 * CONFIG.MAX_SEQ_LEN + 1].astype(np.int32))
    print("🎲 random tokens — DATA_ROOT unset, so the corpus-specific overflow "
          "(#235) CANNOT be caught by this run")
    return jnp.asarray(rng.integers(1, VOCAB_SIZE, size=(1, 2 * CONFIG.MAX_SEQ_LEN + 1)).astype(np.int32))


def main():
    import argparse
    argparse.ArgumentParser(description="real-config GPU smoke, f16 path").parse_args()

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    assert jax.default_backend() == "gpu", "smoke must run on GPU (unset JAX_PLATFORMS / FORCE_F32_COMPUTE)"

    model = build_model(CONFIG, nnx.Rngs(42))
    print(f"📐 {param_count(model) / 1e6:.2f}M params")
    # Optimizer state (Adam m+v, MultiSteps grad accumulator) allocated up front, as
    # in training — the peak that matters is grad step + resident optimizer state.
    optimizer = nnx.Optimizer(model, optimizer_chain(CONFIG, Schedules.of(CONFIG).learning_rate), wrt=nnx.Param)

    # Wake the zero-init residual path before measuring zero-fracs: at init,
    # down_proj == 0 blocks all gradient to gate/up_proj, so nearly half of each
    # block group reads exactly zero for structural reasons (pinned in
    # tests/apparatus/test_grad_zero_frac.py) — and MultiSteps lands no real update within
    # this smoke to clear it. Scale 0.02 puts down_proj where early training
    # does, the regime the underflow reading has to certify. Finiteness and
    # VRAM-fit checks are unaffected.
    key = jax.random.PRNGKey(1)
    for blk in residual_blocks(model):
        key, sub = jax.random.split(key)
        kernel = blk.down_proj.kernel[...]
        blk.down_proj.kernel[...] = 0.02 * jax.random.normal(sub, kernel.shape, kernel.dtype)

    rng = np.random.default_rng(0)
    batch = load_batch(rng)

    # Headroom BEFORE the grad steps: this is a property of the weights and the
    # tokens, and reading it first means a doomed run is refused before it spends
    # anything. Measured on one window, the shape the stack actually sees.
    peaks, worst, headroom = block_headroom(model, batch[:, :CONFIG.MAX_SEQ_LEN])
    print("📏 activation peak per block: "
          + "  ".join(f"{v:,.0f}" for v in peaks))
    print(f"   worst {worst:,.1f} of f16 max {F16_MAX:,.0f} → headroom {headroom:.1%}")
    assert headroom >= MIN_HEADROOM, (
        f"activations reach {worst:,.0f}, leaving {headroom:.1%} of f16 headroom "
        f"(need {MIN_HEADROOM:.0%}). This is #235: the residual branches are unbounded, "
        f"and #229's whole-window NaN is what running out looks like. POST_NORM=1 bounds "
        f"the branch outputs.")

    # The grad step with the optimizer resident: the VRAM peak that matters. Loss is
    # expected to stay flat — optimizer_chain is MultiSteps, so no real update lands in a handful
    # of steps; this checks fit + finiteness, not descent.
    def read_zero_fracs(grads):
        zf = {k: float(v) for k, v in grad_zero_fractions(grads).items()}
        dmax = dense_zero_frac_max(zf)
        print("    zero-frac (#82): dense max=" + f"{dmax:.4f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in zf.items()))
        return dmax

    worst_dense = 0.0

    print("— grad steps with optimizer resident —")
    for s in range(1, 4):
        loss, _, grads, gnorm = compute_grad_step(model, batch)
        loss_f, grad_f = float(loss), float(gnorm)
        ok = math.isfinite(loss_f) and math.isfinite(grad_f) and grad_f > 0
        print(f"  step {s}: loss={loss_f:.4f}  grad_norm={grad_f:.4f}  {'OK' if ok else '✗ NON-FINITE/ZERO'}")
        assert ok, f"step {s} non-finite/zero in f16"
        worst_dense = max(worst_dense, read_zero_fracs(grads))
        apply_grads(optimizer, grads, model)

    print(f"(loss ≈ 2·ln(vocab) = {2 * math.log(VOCAB_SIZE):.2f} at init, both windows summed)")
    # Loud reading, not an assert: the #82 decision rule ("~0 through the smoke
    # and the first base-run stretch") is applied in the finding, and elevated
    # means "file loss-scaling adoption", not "this smoke is broken".
    # DENSE_ZERO_WARN is only the warn-loudly heuristic.
    if worst_dense > DENSE_ZERO_WARN:
        print(f"⚠️ dense-kernel zero-fraction reached {worst_dense:.4f} — possible f16 "
              f"underflow; per #82, file the loss-scaling adoption issue.")
    else:
        print(f"🧊 dense-kernel zero-fraction max {worst_dense:.4f} — no underflow signal (#82).")
    print("✅ GPU smoke passed: f16 path numerically healthy AND fits in 6GB with optimizer state.")


if __name__ == "__main__":
    main()
