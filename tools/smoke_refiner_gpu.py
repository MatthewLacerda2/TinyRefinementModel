"""Real-config GPU smoke for Plan A (RefinerForTraining), f16 path.

What the CPU suite can't cover: CPU XLA cannot lower the f16-with-f32-accumulation
matmuls (config.py), so the real numerical risk — f16 underflow/overflow in the deep
unrolled refine loop — and the 6GB VRAM fit only show up on the GPU. This drives the
*production* grad step (compute_grad_step + apply_grads) at LATENT_DIM/VOCAB_SIZE/
MAX_SEQ_LEN on random tokens (text content is irrelevant to numerical health), across
every depth 1..MAX_STEPS_LIMIT, and asserts finite loss + finite, nonzero grads, then
runs a few optimizer steps. Random tokens, no data pipeline, no run dirs.

    venv/bin/python tools/smoke_refiner_gpu.py
"""

import os
# Match the real run's allocator arena (start_training.py) — the smoke must see the
# same VRAM budget training does, not JAX's smaller 0.75 default.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import LATENT_DIM, MAX_SEQ_LEN, MAX_STEPS_LIMIT, VOCAB_SIZE
from grad_step import compute_grad_step, apply_grads
from plan_a_trainer import RefinerForTraining
from trainer import optimizer_chain


def main():
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    assert jax.default_backend() == "gpu", "smoke must run on GPU (unset JAX_PLATFORMS / FORCE_F32_COMPUTE)"

    model = RefinerForTraining(LATENT_DIM, nnx.Rngs(42))
    n = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"📐 refiner: {n / 1e6:.2f}M params")
    # Optimizer state (Adam m+v, MultiSteps grad accumulator) allocated up front, as
    # in training — the peak that matters is grad step + resident optimizer state.
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.integers(1, VOCAB_SIZE, size=(1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32))

    # Worst-case VRAM and the deepest f16 unroll first: if depth-8 fits with the
    # optimizer resident, every shallower depth training samples does too. Loss is
    # expected to stay flat — optimizer_chain is MultiSteps(every_k=128), so no real
    # update lands in a handful of steps; this checks fit + finiteness, not descent.
    print(f"— depth {MAX_STEPS_LIMIT} (worst case) with optimizer resident —")
    for s in range(1, 4):
        loss, _, grads, gnorm = compute_grad_step(model, batch, jnp.array(s), MAX_STEPS_LIMIT)
        l, g = float(loss), float(gnorm)
        ok = math.isfinite(l) and math.isfinite(g) and g > 0
        print(f"  step {s}: loss={l:.4f}  grad_norm={g:.4f}  {'OK' if ok else '✗ NON-FINITE/ZERO'}")
        assert ok, f"depth {MAX_STEPS_LIMIT} step {s} non-finite/zero in f16"
        apply_grads(optimizer, grads, model)

    print("— finiteness at shallow depths 1 and 4 —")
    for depth in (1, 4):
        loss, _, _, gnorm = compute_grad_step(model, batch, jnp.array(1), depth)
        ok = math.isfinite(float(loss)) and math.isfinite(float(gnorm))
        print(f"  depth {depth}: loss={float(loss):.4f}  grad_norm={float(gnorm):.4f}  {'OK' if ok else '✗'}")
        assert ok

    print(f"(loss ≈ 2·ln(vocab) = {2 * math.log(VOCAB_SIZE):.2f} at init, both windows summed)")
    print("✅ GPU smoke passed: f16 refine loop numerically healthy AND fits in 6GB with optimizer state.")


if __name__ == "__main__":
    main()
