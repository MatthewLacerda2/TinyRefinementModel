"""Golden-run regression: the shipped training path must reproduce a stored loss
trajectory, within the measured cross-machine noise.

Ten real grad steps of `plain`, the arch that ships (#322; until then this pinned the
reasoner, the control arch), at a small config, on a fixed synthetic batch with a fixed
optimizer. The losses are compared against tests/expensive/golden/train_step_losses.json.

WHAT IT CAN AND CANNOT SEE. The file is compared across machines, and float kernels
differ by CPU: the same commit gives numbers one or two ULPs apart. Measured on #330 for
this exact config, one commit across six runs on four CPU models (this box's Ryzen 5
3400G; CI's AMD EPYC 7763, 9V45 and 9V74), the largest relative loss difference over
these 10 steps was NOISE_FLOOR = 1.8e-7 (3.0e-7 over 20). CI runners disagree with each
other, not only with this box. Grad norms were left out on purpose: the optimizer
amplifies the same noise to 4.0e-5 by step 20. RTOL is set 10x above the floor. A later
CI run on an Intel Xeon Platinum 8573C also passed at this tolerance. Measured in CI by
mutation at this tolerance (#330's probe PR #333, never merged), max relative loss move:

  caught      targets shifted by one: 2.0e-1
              attention scores double-scaled (the historical post-mortem): 3.7e-2
              causal mask dropped: 1.2e-2
              attention residual branch scaled by 1+1e-2 / 1+1e-3 / 1+1e-4:
              9.3e-4 / 1.6e-4 / 6.5e-6
  not caught  attention residual branch scaled by 1+1e-5
              every block output scaled by one ULP
              pad mask dropped (the batch has no pads; the leading-pad test in
              tests/core/test_model_invariants.py catches it)

So this catches numeric changes of about 1e-4 relative in one residual branch and up,
not one-ULP drift. Same-process bit-repeatability is tests/core/test_step_determinism.py.

Opt-in locally: RUN_GOLDEN=1 pytest tests/expensive/test_golden_run.py
Re-recording is a deliberate act, done only after an INTENTIONAL math change: delete the
golden file, rerun with RUN_GOLDEN=1, and say in the commit message what changed and why.
A failure is never by itself a reason to re-record.
"""

import json
import os

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from trm.model.plain import PlainTransformer
from trm.train.grad_step import apply_grads, compute_grad_step

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "train_step_losses.json")
CONFIG = {"arch": "plain", "dim": 60, "num_layers": 2, "seed": 9, "steps": 10}
NOISE_FLOOR = 1.8e-7  # max relative loss difference, this box vs CI runs (#330)
# The CPU models that floor was sampled on. Outside them, the floor is an assumption.
FLOOR_CPUS = ("AMD Ryzen 5 3400G", "AMD EPYC 7763", "AMD EPYC 9V45", "AMD EPYC 9V74")
RTOL = 10 * NOISE_FLOOR


def _losses():
    model = PlainTransformer(CONFIG["dim"], nnx.Rngs(CONFIG["seed"]), num_layers=CONFIG["num_layers"])
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)),
        wrt=nnx.Param,
    )
    rng = np.random.default_rng(21)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)

    losses = []
    for step in range(CONFIG["steps"]):
        # depth is a contract argument the plain arch ignores; held fixed so the
        # static argument compiles once.
        loss, _, grads, _ = compute_grad_step(model, batch, step, depth=1)
        apply_grads(optimizer, grads, model)
        losses.append(float(loss))
    return losses


@pytest.mark.skipif(not os.environ.get("RUN_GOLDEN"), reason="opt-in: set RUN_GOLDEN=1")
@pytest.mark.skipif(bool(os.environ.get("RUN_TESTS_ON_GPU")), reason="golden values are recorded under the CPU/f32 test mode")
def test_loss_trajectory_matches_golden():
    losses = _losses()

    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump({"config": CONFIG, "losses": losses}, f, indent=2)
        pytest.skip(f"golden file recorded at {GOLDEN_PATH} — rerun to compare")

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert golden["config"] == CONFIG, "the golden file was recorded under a different config"

    rel = np.abs(np.array(losses) - golden["losses"]) / np.abs(golden["losses"])
    table = "\n".join(f"  step {i + 1:>2}: rel diff {r:.1e}{'  <-- over RTOL' if r > RTOL else ''}"
                      for i, r in enumerate(rel))
    assert rel.max() <= RTOL, (
        f"The loss trajectory moved by up to {rel.max():.1e} relative. Cross-machine noise "
        f"measured for this config is {NOISE_FLOOR:.1e}, sampled on {', '.join(FLOOR_CPUS)}; "
        f"the tolerance is {RTOL:.1e}. On one of those CPUs this is a numeric change in the "
        f"training path, not drift. On a CPU outside that list it MAY be noise: check the "
        f"runner CPU the golden job prints before concluding, and if it is new, measure the "
        f"floor there (#330). If the change was intentional, re-record deliberately (see the "
        f"module docstring).\n{table}"
    )
