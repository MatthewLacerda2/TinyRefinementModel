"""Can the model overfit one batch? The end-to-end smoke for the training path.

Runs the REAL loss and grad step (grad_step.compute_grad_step) on a single
fixed batch with a plain optimizer (no accumulation) and demands that CE
collapses. Wrong labels, off-by-one target shifts, broken gradient flow, and
dead loss components all fail this in minutes instead of an overnight run.

Run it on a free GPU before launching any training run after a model change:
    PYTHONPATH=. python -m instruments.overfit_smoke [--steps 120] [--lr 1e-3]

CI / no-corpus mode (#45): --synthetic overfits a fixed random batch instead of
DATA_ROOT, and --dim/--blocks shrink the model so the whole thing runs in CPU
minutes. Wrong labels, target off-by-ones, broken gradient flow, and dead loss
components fail at tiny scale exactly as they would at full scale:
    FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu PYTHONPATH=. \
        python -m instruments.overfit_smoke --synthetic --dim 60 --blocks 2 --steps 80
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import argparse

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from instruments._common import load_env
from trm.model import build_model
from trm.settings import CONFIG
from trm.train.grad_step import compute_grad_step, apply_grads

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "CE initial -> final": ("measured", "window-2 CE on the one batch being memorized: training CE on seen data, not generalization"),
}

# Where this differs from production's environment, and why (#166).
ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "a tiny correctness smoke; inert anyway under cuda_async, and it asserts learning, not memory"}

load_env()


def main():
    parser = argparse.ArgumentParser(description="single-batch overfit smoke")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic", action="store_true",
                        help="fixed random token batch instead of DATA_ROOT (CI / no-corpus mode)")
    parser.add_argument("--dim", type=int, default=CONFIG.LATENT_DIM,
                        help="model width (shrink for CPU CI; must divide NUM_HEADS)")
    parser.add_argument("--blocks", type=int, default=CONFIG.PLAIN_LAYERS,
                        help="number of blocks (shrink for CPU CI)")
    args = parser.parse_args()

    model = build_model(CONFIG, nnx.Rngs(0), dim=args.dim, num_layers=args.blocks)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr)),
        wrt=nnx.Param,
    )
    if args.synthetic:
        rng = np.random.default_rng(21)
        batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * CONFIG.MAX_SEQ_LEN + 1)), dtype=jnp.int32)
    else:
        from trm.runtime.restore import load_eval_batches
        batch = load_eval_batches(CONFIG, num_rows=1, skip=0)[0]

    ces = []
    for step in range(args.steps):
        loss, out, grads, grad_norm = compute_grad_step(model, batch)
        apply_grads(optimizer, grads, model)
        ce = float(out.diag["token_loss"])
        ces.append(ce)
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} | window-2 CE {ce:.4f} | loss {float(loss):.4f} | grad norm {float(grad_norm):.2f}")

    print("-" * 40)
    initial_ce, final_ce = ces[0], ces[-1]
    threshold = 0.6 * initial_ce
    verdict = "PASS" if final_ce < threshold else "FAIL"
    print(f"{verdict}: CE {initial_ce:.4f} -> {final_ce:.4f} (must drop below {threshold:.4f})")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
