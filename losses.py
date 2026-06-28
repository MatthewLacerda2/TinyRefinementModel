"""Memory-lean training losses.

Chunked cross-entropy. The full ``[batch, seq, vocab]`` logit + softmax tensor is the
single biggest activation in the loss/backward — f32, ~1.37 GB at seq 512 / vocab 50304
with two windows live together — and it OOM'd the dim960 base run on the 6 GB card
(#19, #16). We never need all of it at once: each position's cross-entropy is
independent given that position's own full-vocab logits. So we project + score the
sequence one chunk of positions at a time, keeping only ``[batch, chunk, vocab]`` live,
and rematerialize each chunk in the backward (``jax.checkpoint``) so the gradient pass
is bounded the same way.

The per-token loss is identical to the naive full-logit CE — the vocab axis is never
chunked, so each logsumexp is over the full vocabulary. Only the order of the float32
sum over positions changes, so a golden-run re-record is expected (drift ~1e-6).
"""

import jax
import jax.numpy as jnp
import optax


def chunked_cross_entropy(hidden, embedding, targets, pad_id, chunk_size=64):
    """Masked-mean next-token cross-entropy through the tied LM head, computed in
    sequence-axis chunks so the full vocab-wide logits are never materialized.

    Args:
        hidden:    ``[b, s, d]`` final pre-head states (the model returns these during
                   training instead of full logits).
        embedding: ``[vocab, d]`` the tied token embedding; its transpose is the LM head.
                   Passed in (not closed over) so autograd routes the head's gradient
                   back to the embedding exactly as the in-model matmul used to.
        targets:   ``[b, s]`` next-token ids. Positions equal to ``pad_id`` are masked.
        chunk_size: positions scored per chunk. Need not divide ``s`` (last chunk is
                   shorter). Smaller = lower peak, more recompute.

    Returns:
        Scalar ``sum(CE * mask) / sum(mask)`` — identical to the naive computation up to
        float32 summation order.
    """
    b, s, d = hidden.shape
    embed_t = embedding.astype(hidden.dtype).T  # [d, vocab]

    @jax.checkpoint  # recompute each chunk's logits in the backward instead of storing
    def chunk_loss(h_chunk, t_chunk):
        # Full-vocab logits for this slice only. f32 accumulation matches the original
        # head (model.py / plan_a_model.py both use preferred_element_type=float32).
        logits = jnp.matmul(h_chunk, embed_t, preferred_element_type=jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, t_chunk)
        mask = (t_chunk != pad_id).astype(ce.dtype)
        return jnp.sum(ce * mask), jnp.sum(mask)

    # s is static at trace time, so this Python loop unrolls cleanly and only the two
    # running scalars (not any chunk's logits) survive across iterations — that is what
    # bounds the forward peak; the @checkpoint bounds the backward.
    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
    count = jnp.asarray(0.0, dtype=jnp.float32)
    for start in range(0, s, chunk_size):
        end = min(start + chunk_size, s)
        ls, c = chunk_loss(hidden[:, start:end], targets[:, start:end])
        loss_sum = loss_sum + ls
        count = count + c
    return loss_sum / count.clip(min=1.0)
