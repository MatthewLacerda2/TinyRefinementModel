"""Train-step determinism: same seed, same batch, identical loss — twice.

This is what makes resume-replay and golden-run comparisons meaningful at all.
Rebuilds the model from the same seed and runs the real grad step on the same
batch; the losses must match exactly (CPU execution is deterministic; on GPU
this also holds for this model's ops).
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from config import LATENT_DIM, MAX_SEQ_LEN
from grad_step import compute_grad_step
from model import UniversalReasoner


def _one_step():
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(5), batch_size=1)
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)
    loss, out, grads, grad_norm = compute_grad_step(model, batch, step=0, max_steps=1)
    return float(loss), float(grad_norm)


def test_grad_step_is_deterministic():
    loss_a, gnorm_a = _one_step()
    loss_b, gnorm_b = _one_step()
    assert loss_a == loss_b, f"loss differs across identical runs: {loss_a} vs {loss_b}"
    assert gnorm_a == gnorm_b, f"grad norm differs across identical runs: {gnorm_a} vs {gnorm_b}"
