"""The residual stream's width is a knob, and the model obeys it (#357).

CLAUDE.md's dtype policy: matmuls in f16, accumulators in f32. The residual stream is
the accumulator every block adds into; in f16 a block's O(1) output rounds to nothing
once the stream passes ~4k. `RESIDUAL_DTYPE` sets it, and #357's pair judges the
default. Built with f16 compute explicitly, so the check does not depend on the
suite's FORCE_F32_COMPUTE.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.model.plain import PlainTransformer


@pytest.mark.parametrize("residual", [jnp.float16, jnp.float32])
def test_every_state_of_the_stream_has_the_residual_dtype(residual):
    model = PlainTransformer(32, nnx.Rngs(0), vocab_size=37, num_heads=4, num_layers=2,
                             max_seq_len=16, dtype=jnp.float16, residual_dtype=residual)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 37
    _, _, _, states = model._stream(tokens, keep_states=True)
    assert [s.dtype for s in states] == [jnp.dtype(residual)] * 3
    # The matmuls still run in the compute dtype: the loss head gets f16 either way.
    assert model(tokens, training=True).hidden.dtype == jnp.float16


def test_the_stream_is_never_narrower_than_compute():
    """Under FORCE_F32_COMPUTE (the suite's default) the stream is f32 whatever the
    env says, so the test path stays all-f32."""
    from trm.config import COMPUTE_DTYPE, RESIDUAL_DTYPE
    assert jnp.dtype(RESIDUAL_DTYPE).itemsize >= jnp.dtype(COMPUTE_DTYPE).itemsize
