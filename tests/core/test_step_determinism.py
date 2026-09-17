"""Train-step determinism: same seed, same batch, identical loss — twice.

Same-process repeatability, which is what resume-replay rests on. Rebuilds the
model from the same seed and runs the real grad step on the same batch; the losses
must match exactly (CPU execution is deterministic; on GPU this also holds for this
model's ops). This is NOT cross-machine identity: the same commit gives numbers a few
ULPs apart on different CPUs, which is why the golden run compares against a stored
file within a measured tolerance instead of exactly (#322).
"""

import jax.numpy as jnp
import numpy as np

from trm.config import MAX_SEQ_LEN
from trm.train.grad_step import compute_grad_step


def _one_step(make_tiny_model):
    # The arch a run would train, at the shared tiny config (#322).
    model = make_tiny_model(seed=5)
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)
    loss, out, grads, grad_norm = compute_grad_step(model, batch, step=0, depth=1)
    return float(loss), float(grad_norm)


def test_grad_step_is_deterministic(make_tiny_model):
    loss_a, gnorm_a = _one_step(make_tiny_model)
    loss_b, gnorm_b = _one_step(make_tiny_model)
    assert loss_a == loss_b, f"loss differs across identical runs: {loss_a} vs {loss_b}"
    assert gnorm_a == gnorm_b, f"grad norm differs across identical runs: {gnorm_a} vs {gnorm_b}"
