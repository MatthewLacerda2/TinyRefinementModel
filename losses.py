"""Memory-lean training losses.

Chunked cross-entropy via a custom VJP whose forward AND backward are both ``lax.scan``
loops. The full ``[batch, seq, vocab]`` logit + softmax tensor is the single biggest
activation in the loss/backward — f32, ~1.4 GB at seq 512 / vocab 50304 — and it OOM'd
the dim960 base run on the 6 GB card (#19, #16). Each position's cross-entropy is
independent given that position's own full-vocab logits, so we score the sequence one
chunk of positions at a time, never materializing more than ``[batch, chunk, vocab]``.

Three earlier cuts failed, each teaching the constraint the next had to satisfy:

* Python ``for`` + ``jax.checkpoint`` per chunk — XLA fused the *unrolled* regions and
  raised the remat floor.
* ``lax.scan`` with a rematerialized body — bounded the logits, but scan's reverse mode
  *stacked* the per-step gradient of the shared embedding into ``[n_chunks, vocab, dim]``.
* ``custom_vjp`` with Python-loop forward/backward — fixed the stacking, but the Python
  loops *unrolled* again and XLA re-materialized the full logits under a loose memory
  budget.

This version satisfies all of it: ``lax.scan`` (rolled, XLA cannot unroll it) in both
directions, and a hand-written backward that **accumulates** ``grad_embedding`` in the
scan carry (never stacked) and emits ``grad_hidden`` one chunk at a time. Memory is
bounded by construction — one chunk's logits plus the small fixed gradients —
independent of XLA's rematerialization heuristics or the allocator's budget.

The per-token loss and its gradients are identical to the naive full-logit CE (the vocab
axis is never chunked); only the float32 sum over positions reorders, so a golden-run
re-record is expected. The unit test asserts value AND gradient parity.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax


def _to_chunks(hidden, targets, pad_id, chunk_size):
    """Pad the sequence up to a whole number of chunks and lay it out as
    ``[n_chunks, b, chunk, ...]`` so ``lax.scan`` walks the chunk axis. Pad positions
    carry ``pad_id`` and are masked, so they change nothing numerically."""
    b, s, d = hidden.shape
    n_pad = (-s) % chunk_size
    if n_pad:
        hidden = jnp.pad(hidden, ((0, 0), (0, n_pad), (0, 0)))
        targets = jnp.pad(targets, ((0, 0), (0, n_pad)), constant_values=pad_id)
    n_chunks = (s + n_pad) // chunk_size
    h_steps = hidden.reshape(b, n_chunks, chunk_size, d).swapaxes(0, 1)
    t_steps = targets.reshape(b, n_chunks, chunk_size).swapaxes(0, 1)
    return h_steps, t_steps, n_chunks, n_pad


def _masked_chunked_loss(hidden, embedding, targets, pad_id, chunk_size):
    """Scalar loss, scored chunk-by-chunk via scan so the forward never holds the full
    vocab-wide logits. Shared by the custom_vjp primal and its forward rule."""
    embed_t = embedding.astype(hidden.dtype).T  # [d, vocab]
    h_steps, t_steps, _, _ = _to_chunks(hidden, targets, pad_id, chunk_size)

    def step(carry, xs):
        loss_sum, count = carry
        h, t = xs
        # f32 logits accumulation matches the original head (preferred_element_type).
        logits = jnp.matmul(h, embed_t, preferred_element_type=jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, t)
        mask = (t != pad_id).astype(ce.dtype)
        return (loss_sum + jnp.sum(ce * mask), count + jnp.sum(mask)), None

    init = (jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32))
    (loss_sum, count), _ = jax.lax.scan(step, init, (h_steps, t_steps))
    return loss_sum / count.clip(min=1.0)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def chunked_cross_entropy(hidden, embedding, targets, pad_id, chunk_size=128):
    """Masked-mean next-token cross-entropy through the tied LM head, scored in
    sequence-axis chunks with an explicit, memory-bounded (scan) backward.

    Args:
        hidden:    ``[b, s, d]`` final pre-head states (the model returns these during
                   training instead of full logits).
        embedding: ``[vocab, d]`` the tied token embedding; its transpose is the LM head.
        targets:   ``[b, s]`` next-token ids. Positions equal to ``pad_id`` are masked.
        chunk_size: positions scored per scan step. Smaller = lower peak logits, more
                   steps. With the accumulating backward there is no per-chunk gradient
                   penalty, so smaller is strictly leaner.

    Returns:
        Scalar ``sum(CE * mask) / sum(mask)`` — identical to the naive computation (value
        and gradients) up to float32 summation order.
    """
    return _masked_chunked_loss(hidden, embedding, targets, pad_id, chunk_size)


def _cce_fwd(hidden, embedding, targets, pad_id, chunk_size):
    # Residuals are just the inputs — no logits saved, so the forward stays bounded.
    # targets is integer (non-differentiable) but can't be a nondiff_argnum because it is
    # a tracer; it rides in the residuals and gets a None cotangent in _cce_bwd.
    loss = _masked_chunked_loss(hidden, embedding, targets, pad_id, chunk_size)
    return loss, (hidden, embedding, targets)


def _cce_bwd(pad_id, chunk_size, residuals, g):
    hidden, embedding, targets = residuals
    b, s, d = hidden.shape
    vocab = embedding.shape[0]
    embed_t = embedding.astype(hidden.dtype).T  # [d, vocab]

    # loss = sum(mask * CE) / count, so d loss/d logits = (1/count) * mask * (p - onehot).
    count = jnp.sum(targets != pad_id).astype(jnp.float32).clip(min=1.0)
    scale = g / count

    h_steps, t_steps, n_chunks, n_pad = _to_chunks(hidden, targets, pad_id, chunk_size)

    def step(grad_emb, xs):
        h, t = xs                           # [b, c, d], [b, c]
        logits = jnp.matmul(h, embed_t, preferred_element_type=jnp.float32)  # [b, c, vocab]
        probs = jax.nn.softmax(logits, axis=-1)
        glog = probs - jax.nn.one_hot(t, vocab, dtype=probs.dtype)
        glog = glog * (t != pad_id)[..., None].astype(probs.dtype) * scale   # [b, c, vocab] f32
        # this chunk's grad_hidden: glog @ embedding  ([b,c,vocab]@[vocab,d])
        gh = jnp.matmul(glog.astype(hidden.dtype), embedding.astype(hidden.dtype))
        # accumulate this chunk's embedding-grad contribution into the carry (no stacking)
        ge = jnp.einsum('bcv,bcd->vd', glog, h.astype(glog.dtype)).astype(grad_emb.dtype)
        return grad_emb + ge, gh

    grad_embedding, gh_steps = jax.lax.scan(step, jnp.zeros_like(embedding), (h_steps, t_steps))
    # gh_steps [n_chunks, b, chunk, d] -> [b, n_chunks*chunk, d], trim the padding.
    grad_hidden = gh_steps.swapaxes(0, 1).reshape(b, n_chunks * chunk_size, d)[:, :s]

    # (hidden, embedding, targets) cotangents; targets is integer → no gradient.
    return grad_hidden, grad_embedding, None


chunked_cross_entropy.defvjp(_cce_fwd, _cce_bwd)
