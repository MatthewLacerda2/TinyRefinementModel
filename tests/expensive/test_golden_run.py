"""Golden-run regression: a refactor that claims to be math-identical must
reproduce the recorded training trajectory.

Twenty real grad steps of the SHIPPED architecture (`plain`, #322) at a small
config, on a fixed synthetic batch with a fixed optimizer, compared against stored
losses and grad norms. Until #322 this pinned the reasoner, the control arch, so a
numeric change on the plain path moved nothing here.

Why twenty steps and grad norms, not five losses: a change of about one ULP in the
forward (every block output scaled by 1+1e-6) moves the first five losses by one or
two ULPs, which is inside the tolerance. The optimizer amplifies it: by step 12 the
grad norm has moved by more than 1e-6 relative, and by step 20 by ~1e-5. So the
trajectory is long enough for a one-ULP change to come out above the tolerance.

The tolerance exists because float kernels differ by CPU: the same commit gives
numbers a few ULPs apart on different machines. CI is where this guard runs, so the
golden file holds the numbers CI computes; a local run on another CPU can differ by
more than rtol, and when it does, compare your branch against main on the same machine
rather than against the file.

Opt-in locally: RUN_GOLDEN=1 pytest tests/expensive/test_golden_run.py
After an INTENTIONAL math change (new loss term, architecture change), delete
tests/expensive/golden/train_step_losses.json and rerun with RUN_GOLDEN=1 to re-record,
or paste the values the failing CI job prints. Say so in the commit message.
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
CONFIG = {"arch": "plain", "dim": 60, "num_layers": 2, "seed": 9, "steps": 20}
RTOL = 1e-6


def _trajectory():
    model = PlainTransformer(CONFIG["dim"], nnx.Rngs(CONFIG["seed"]), num_layers=CONFIG["num_layers"])
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)),
        wrt=nnx.Param,
    )
    rng = np.random.default_rng(21)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)

    losses, grad_norms = [], []
    for step in range(CONFIG["steps"]):
        # depth is a contract argument the plain arch ignores; held fixed so the
        # static argument compiles once.
        loss, _, grads, grad_norm = compute_grad_step(model, batch, step, depth=1)
        apply_grads(optimizer, grads, model)
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))
    return {"config": CONFIG, "losses": losses, "grad_norms": grad_norms}


@pytest.mark.skipif(not os.environ.get("RUN_GOLDEN"), reason="opt-in: set RUN_GOLDEN=1")
@pytest.mark.skipif(bool(os.environ.get("RUN_TESTS_ON_GPU")), reason="golden values are recorded under the CPU/f32 test mode")
def test_grad_steps_match_golden_trajectory():
    got = _trajectory()

    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(got, f, indent=2)
        pytest.skip(f"golden file recorded at {GOLDEN_PATH} — rerun to compare")

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert golden["config"] == CONFIG, "the golden file was recorded under a different config"
    # The full-precision values, so a failing CI run is also the re-recording.
    recording = json.dumps(got, indent=2)
    for key in ("losses", "grad_norms"):
        np.testing.assert_allclose(
            got[key], golden[key], rtol=RTOL,
            err_msg=f"{key} drifted from the golden record. If the math change was "
                    f"intentional, re-record (delete the golden file and rerun with "
                    f"RUN_GOLDEN=1, or use the values below) and say so in the commit "
                    f"message.\n{recording}",
        )
