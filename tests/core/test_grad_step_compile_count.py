"""The grad step compiles once per run, not once per micro-step (#316).

What moves between micro-steps — the batch, the loss scale (backed off on a
non-finite step, #199), the clip ceiling (the grad guard's threshold) — must be
traced. A static argument's value is part of the jit cache key, so one that moves
compiles an identical program again, each with its own resident executable and CUDA
graphs on a 6GB card; #316 is the incident (a sampled depth did exactly that).

A static argument's effect lives in the jit cache key, not in the model, so a tiny
model shows it. Sequence length is the one knob that cannot shrink: the grad step
splits its batch at the config's MAX_SEQ_LEN.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.model.plain import PlainTransformer
from trm.settings import CONFIG
from trm.train.grad_step import compute_grad_step

MICRO_STEPS = 16
TRACES = []  # one entry per forward traced, appended at trace time only


class _CountingPlain(PlainTransformer):
    """PlainTransformer whose forward records each time jit traces it."""

    def __call__(self, *args, **kwargs):
        TRACES.append(1)
        return super().__call__(*args, **kwargs)


@pytest.fixture
def plain():
    TRACES.clear()
    return _CountingPlain(32, nnx.Rngs(0), CONFIG, vocab_size=37, num_heads=4, num_layers=1,
                          max_seq_len=CONFIG.MAX_SEQ_LEN, pad_token_id=36)


def test_the_grad_step_traces_once_across_micro_steps(plain):
    """The trainer's call, micro-step by micro-step, with every per-step value moving.
    Each trace runs the forward twice, once per window."""
    for step in range(MICRO_STEPS):
        batch = jnp.full((1, 2 * CONFIG.MAX_SEQ_LEN + 1), 1 + step % 30, dtype=jnp.int32)
        compute_grad_step(plain, batch, loss_scale=jnp.float32(2.0 ** (16 - step)),
                          clip_norm=jnp.float32(1.0 + step))
    assert len(TRACES) // 2 == 1
