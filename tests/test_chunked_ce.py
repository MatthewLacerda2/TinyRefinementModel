"""Chunked cross-entropy must be math-identical to the naive full-logit CE — in both
the loss value AND the gradients (a wrong backward would be a silent training bug that
the loss value alone wouldn't catch). Chunking only reorders a float32 sum over
positions, so the tolerance is tight. Includes a non-dividing chunk size, since the
real seq length need not be a multiple of the chunk.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from losses import chunked_cross_entropy


def _naive_ce(hidden, embedding, targets, pad_id):
    """The computation chunked_cross_entropy replaces: full [b, s, vocab] logits."""
    logits = jnp.matmul(hidden, embedding.astype(hidden.dtype).T,
                        preferred_element_type=jnp.float32)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = (targets != pad_id).astype(ce.dtype)
    return jnp.sum(ce * mask) / jnp.sum(mask).clip(min=1.0)


def _fixture(seed=0, b=2, s=40, d=8, vocab=17):
    rng = np.random.default_rng(seed)
    hidden = jnp.asarray(rng.standard_normal((b, s, d)), dtype=jnp.float32)
    embedding = jnp.asarray(rng.standard_normal((vocab, d)), dtype=jnp.float32)
    targets = jnp.asarray(rng.integers(0, vocab, size=(b, s)), dtype=jnp.int32)
    return hidden, embedding, targets


@pytest.mark.parametrize("chunk_size", [16, 13, 40, 64])  # divides, doesn't divide, ==s, >s
def test_value_matches_naive(chunk_size):
    hidden, embedding, targets = _fixture()
    pad_id = 0  # mask out the zero-token positions too, exercising the mask path
    naive = _naive_ce(hidden, embedding, targets, pad_id)
    chunked, _ = chunked_cross_entropy(hidden, embedding, targets, pad_id, chunk_size=chunk_size)
    assert jnp.allclose(naive, chunked, rtol=1e-5, atol=1e-6), (float(naive), float(chunked))


@pytest.mark.parametrize("chunk_size", [16, 13])
def test_gradients_match_naive(chunk_size):
    hidden, embedding, targets = _fixture(seed=1)
    pad_id = 0

    naive_g = jax.grad(lambda h, e: _naive_ce(h, e, targets, pad_id), argnums=(0, 1))(hidden, embedding)
    chunk_g = jax.grad(lambda h, e: chunked_cross_entropy(h, e, targets, pad_id, chunk_size=chunk_size)[0],
                       argnums=(0, 1))(hidden, embedding)

    assert jnp.allclose(naive_g[0], chunk_g[0], rtol=1e-4, atol=1e-6), "grad wrt hidden differs"
    assert jnp.allclose(naive_g[1], chunk_g[1], rtol=1e-4, atol=1e-6), "grad wrt embedding differs"


def test_all_padding_is_safe():
    """A window that is entirely padding must not divide by zero — neither the loss
    nor any of the telemetry stats."""
    hidden, embedding, _ = _fixture(seed=2)
    targets = jnp.zeros((2, 40), dtype=jnp.int32)
    loss, stats = chunked_cross_entropy(hidden, embedding, targets, pad_id=0, chunk_size=16)
    assert jnp.isfinite(loss)
    for name, value in stats.items():
        assert jnp.isfinite(value), name
