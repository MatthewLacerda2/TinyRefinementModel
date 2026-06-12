"""Golden-run regression: a refactor that claims to be math-identical must
reproduce the recorded loss trajectory exactly.

Five real grad steps (varied depths, fixed synthetic batch, fixed optimizer)
compared against stored values. CPU execution is deterministic, so the match
is exact.

Opt-in (several minutes on CPU): RUN_GOLDEN=1 pytest tests/test_golden_run.py
After an INTENTIONAL math change (new loss term, architecture change), delete
tests/golden/train_step_losses.json and rerun with RUN_GOLDEN=1 to re-record —
and say so in the commit message.
"""

import json
import os

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from config import LATENT_DIM, MAX_SEQ_LEN
from grad_step import compute_grad_step, apply_grads
from model import UniversalReasoner

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "train_step_losses.json")
DEPTHS = [1, 2, 3, 2, 1]


@pytest.mark.skipif(not os.environ.get("RUN_GOLDEN"), reason="opt-in: set RUN_GOLDEN=1")
@pytest.mark.skipif(bool(os.environ.get("RUN_TESTS_ON_GPU")), reason="golden values are recorded under the CPU/f32 test mode")
def test_grad_steps_match_golden_trajectory():
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(9), batch_size=1)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)),
        wrt=nnx.Param,
    )
    rng = np.random.default_rng(21)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)

    losses = []
    for step, depth in enumerate(DEPTHS):
        loss, out, grads, grad_norm = compute_grad_step(model, batch, step, max_steps=depth)
        apply_grads(optimizer, grads, model)
        losses.append(float(loss))

    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump({"depths": DEPTHS, "losses": losses}, f, indent=2)
        pytest.skip(f"golden file recorded at {GOLDEN_PATH} — rerun to compare")

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert golden["depths"] == DEPTHS
    np.testing.assert_allclose(
        losses, golden["losses"], rtol=1e-6,
        err_msg="Loss trajectory drifted from the golden record. If the math "
                "change was intentional, delete the golden file, re-record with "
                "RUN_GOLDEN=1, and say so in the commit message.",
    )
