"""Logit-scale telemetry (#80): the thermometer for output-distribution drift.

The chunked-CE scan accumulates mean softmax entropy, mean log Z, and max |logit|
over non-pad positions. Three contracts pinned here:

1. The chunked, masked accumulation matches a naive full-logit computation.
2. The stats are measurement-only — no gradient flows through them.
3. The real grad step surfaces them on out.diag, where the metrics logger reads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from config import LATENT_DIM, MAX_SEQ_LEN
from grad_step import compute_grad_step
from losses import chunked_cross_entropy
from model import UniversalReasoner


def _naive_stats(hidden, embedding, targets, pad_id):
    """The unchunked reference: full [b, s, vocab] logits, masked reductions."""
    logits = jnp.matmul(hidden, embedding.astype(hidden.dtype).T,
                        preferred_element_type=jnp.float32)
    mask = targets != pad_id
    logz = jax.nn.logsumexp(logits, axis=-1)
    entropy = logz - jnp.sum(jax.nn.softmax(logits, axis=-1) * logits, axis=-1)
    count = jnp.sum(mask).clip(min=1)
    return {
        'out_entropy': jnp.sum(entropy * mask) / count,
        'logz_mean': jnp.sum(logz * mask) / count,
        'max_abs_logit': jnp.max(jnp.where(mask, jnp.max(jnp.abs(logits), axis=-1), 0.0)),
    }


def _fixture(seed=0, b=2, s=40, d=8, vocab=17):
    rng = np.random.default_rng(seed)
    hidden = jnp.asarray(rng.standard_normal((b, s, d)), dtype=jnp.float32)
    embedding = jnp.asarray(rng.standard_normal((vocab, d)), dtype=jnp.float32)
    targets = jnp.asarray(rng.integers(0, vocab, size=(b, s)), dtype=jnp.int32)
    return hidden, embedding, targets


@pytest.mark.parametrize("chunk_size", [16, 13, 40, 64])  # divides, doesn't divide, ==s, >s
def test_stats_match_naive(chunk_size):
    hidden, embedding, targets = _fixture()
    pad_id = 0  # some targets are 0, so the pad mask genuinely excludes positions
    naive = _naive_stats(hidden, embedding, targets, pad_id)
    _, stats = chunked_cross_entropy(hidden, embedding, targets, pad_id, chunk_size=chunk_size)
    for name in naive:
        assert jnp.allclose(naive[name], stats[name], rtol=1e-5, atol=1e-6), \
            (name, float(naive[name]), float(stats[name]))


def test_stats_carry_no_gradient():
    """Differentiating through a stat must yield exactly zero — the backward drops
    the stats cotangent, so telemetry can never leak into training gradients."""
    hidden, embedding, targets = _fixture(seed=1)

    def stat_sum(h, e):
        _, stats = chunked_cross_entropy(h, e, targets, pad_id=0, chunk_size=16)
        return stats['out_entropy'] + stats['logz_mean'] + stats['max_abs_logit']

    gh, ge = jax.grad(stat_sum, argnums=(0, 1))(hidden, embedding)
    assert not jnp.any(gh) and not jnp.any(ge)


def test_grad_step_surfaces_stats_in_diag():
    """The production grad step must expose the readings where the metrics logger
    looks (out.diag), finite and within the softmax's hard bounds."""
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(5), batch_size=1)
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.integers(1, 5000, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)
    _, out, _, _ = compute_grad_step(model, batch, step=0, max_steps=1)

    vocab = model.embed.embedding[...].shape[0]
    entropy = float(out.diag['out_entropy'])
    logz = float(out.diag['logz_mean'])
    max_abs = float(out.diag['max_abs_logit'])
    assert 0.0 <= entropy <= float(jnp.log(vocab)) + 1e-3
    assert np.isfinite(logz)
    assert np.isfinite(max_abs) and max_abs > 0.0
