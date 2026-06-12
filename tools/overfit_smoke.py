"""Can the model overfit one batch? The end-to-end smoke for the training path.

Runs the REAL loss and grad step (grad_step.compute_grad_step) on a single
fixed batch with a plain optimizer (no accumulation) and demands that CE
collapses. Wrong labels, off-by-one target shifts, broken gradient flow, and
dead loss components all fail this in minutes instead of an overnight run.

Run it on a free GPU before launching any training run after an architecture
change:
    PYTHONPATH=. python tools/overfit_smoke.py [--steps 120] [--lr 1e-3]
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import argparse

import optax
from flax import nnx
from dotenv import load_dotenv

from config import LATENT_DIM
from model import UniversalReasoner
from grad_step import compute_grad_step, apply_grads

load_dotenv()

from tools.common import load_eval_batches


def main():
    parser = argparse.ArgumentParser(description="single-batch overfit smoke")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2, help="fixed reasoning depth (one compile)")
    args = parser.parse_args()

    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(0))
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr)),
        wrt=nnx.Param,
    )
    batch = load_eval_batches(num_batches=1, skip=0)[0]

    initial_ce = None
    final_ce = None
    for step in range(args.steps):
        loss, out, grads, grad_norm = compute_grad_step(
            model, batch, step, max_steps=args.depth, doc_boundary=False
        )
        apply_grads(optimizer, grads, model)
        ce = float(out.diag["token_loss"])
        if initial_ce is None:
            initial_ce = ce
        final_ce = ce
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} | window-2 CE {ce:.4f} | loss {float(loss):.4f} | grad norm {float(grad_norm):.2f}")

    print("-" * 40)
    threshold = 0.6 * initial_ce
    verdict = "PASS" if final_ce < threshold else "FAIL"
    print(f"{verdict}: CE {initial_ce:.4f} -> {final_ce:.4f} (must drop below {threshold:.4f})")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
