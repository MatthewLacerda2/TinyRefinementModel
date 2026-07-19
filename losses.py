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

A fourth constraint surfaced at dim960 (#128): the grad step scored its two windows
with two SEPARATE chunked-CE calls, and XLA does not reuse buffers across the two
custom_vjp boundaries — the [vocab, dim] f32 gradient plumbing (scan carry ping-pong,
per-chunk GEMM output, final add, plus the f16 embed-cast) existed once PER CALL,
~1.3 GiB of duplicated vocab-sized temporaries in the packed temp arena. The core is
therefore per-row (``chunked_cross_entropy_rows``): callers stack their windows on the
batch axis and score them in ONE scan — one carry, one GEMM chain, one cast — and read
per-window losses from the per-row sums/counts. The scalar API below wraps it.

The per-token loss and its gradients are identical to the naive full-logit CE (the vocab
axis is never chunked); only the float32 sum over positions reorders, so a golden-run
re-record is expected. The unit test asserts value AND gradient parity.

The scan also accumulates logit-scale telemetry (#80) — softmax entropy, log Z,
max |logit|, per row over non-pad positions — since each chunk's full-vocab f32
logits are already in hand here and nowhere else. Measurement only: the stats ride
back as extra outputs whose cotangents the backward ignores, so they can never leak
gradient into training.
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


def _masked_chunked_rows(hidden, embedding, targets, pad_id, chunk_size):
    """Per-row CE sums and telemetry, scored chunk-by-chunk via one scan so the
    forward never holds the full vocab-wide logits. Shared by the custom_vjp primal
    and its forward rule. Returns ``(loss_sums [b], counts [b], stats)`` where stats
    holds per-row ``out_entropy``, ``logz_mean`` (masked means) and ``max_abs_logit``."""
    b = hidden.shape[0]
    embed_t = embedding.astype(hidden.dtype).T  # [d, vocab]
    h_steps, t_steps, _, _ = _to_chunks(hidden, targets, pad_id, chunk_size)

    def step(carry, xs):
        loss_sum, count, ent_sum, logz_sum, max_abs = carry  # all [b]
        h, t = xs
        # f32 logits accumulation matches the original head (preferred_element_type).
        logits = jnp.matmul(h, embed_t, preferred_element_type=jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, t)
        mask = (t != pad_id).astype(ce.dtype)
        # Logit-scale telemetry (#80): a few reductions on logits already in hand.
        # H = log Z − Σ p·logits (softmax entropy), per position.
        logz = jax.nn.logsumexp(logits, axis=-1)
        entropy = logz - jnp.sum(jax.nn.softmax(logits, axis=-1) * logits, axis=-1)
        pos_max_abs = jnp.max(jnp.abs(logits), axis=-1)
        carry = (
            loss_sum + jnp.sum(ce * mask, axis=-1),
            count + jnp.sum(mask, axis=-1),
            ent_sum + jnp.sum(entropy * mask, axis=-1),
            logz_sum + jnp.sum(logz * mask, axis=-1),
            jnp.maximum(max_abs, jnp.max(jnp.where(mask > 0, pos_max_abs, 0.0), axis=-1)),
        )
        return carry, None

    zeros = jnp.zeros((b,), jnp.float32)
    init = (zeros, zeros, zeros, zeros, zeros)
    (loss_sums, counts, ent_sums, logz_sums, max_abs), _ = jax.lax.scan(
        step, init, (h_steps, t_steps))
    denom = counts.clip(min=1.0)
    stats = {
        'out_entropy': ent_sums / denom,
        'logz_mean': logz_sums / denom,
        'max_abs_logit': max_abs,
    }
    return loss_sums, counts, stats


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def chunked_cross_entropy_rows(hidden, embedding, targets, pad_id, chunk_size=128):
    """Per-row masked CE sums through the tied LM head, scored in sequence-axis
    chunks with an explicit, memory-bounded (scan) backward.

    This is the core: ONE scan regardless of how many logical windows the caller
    packed onto the batch axis, so the [vocab, dim] gradient plumbing exists once
    (#128). Callers derive their losses from the sums/counts, e.g.
    ``ce_w = loss_sums[w] / counts[w].clip(min=1)``.

    Args:
        hidden:    ``[b, s, d]`` final pre-head states (the model returns these during
                   training instead of full logits).
        embedding: ``[vocab, d]`` the tied token embedding; its transpose is the LM head.
        targets:   ``[b, s]`` next-token ids. Positions equal to ``pad_id`` are masked.
        chunk_size: positions scored per scan step. Smaller = lower peak logits, more
                   steps. With the accumulating backward there is no per-chunk gradient
                   penalty, so smaller is strictly leaner.

    Returns:
        ``(loss_sums, counts, stats)``:
        loss_sums — ``[b]`` per-row ``sum(CE * mask)`` — identical to the naive
                computation (value and gradients) up to float32 summation order.
        counts — ``[b]`` per-row non-pad position counts (f32). Measurement-grade:
                the backward ignores its cotangent, so divide by it freely.
        stats — per-row logit-scale telemetry over non-pad positions (#80):
                ``out_entropy`` (mean softmax entropy), ``logz_mean`` (mean log Z),
                ``max_abs_logit``. Measurement only — the backward ignores their
                cotangent, so no gradient can flow through them.
    """
    return _masked_chunked_rows(hidden, embedding, targets, pad_id, chunk_size)


def _cce_fwd(hidden, embedding, targets, pad_id, chunk_size):
    # Residuals are just the inputs — no logits saved, so the forward stays bounded.
    # targets is integer (non-differentiable) but can't be a nondiff_argnum because it is
    # a tracer; it rides in the residuals and gets a None cotangent in _cce_bwd.
    out = _masked_chunked_rows(hidden, embedding, targets, pad_id, chunk_size)
    return out, (hidden, embedding, targets)


def _cce_bwd(pad_id, chunk_size, residuals, g):
    hidden, embedding, targets = residuals
    # g mirrors the (loss_sums, counts, stats) output; the counts and stats cotangents
    # are dropped — counts is a denominator-grade measurement, stats are diagnostics,
    # both structurally outside the training gradient.
    g_sums, _, _ = g
    b, s, d = hidden.shape
    vocab = embedding.shape[0]
    embed_t = embedding.astype(hidden.dtype).T  # [d, vocab]

    # loss_sums[i] = sum(mask_i * CE_i), so d loss_sums[i] / d logits_i = mask_i * (p - onehot)
    # scaled by that row's incoming cotangent.
    scale = g_sums[:, None, None]  # [b, 1, 1]

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


chunked_cross_entropy_rows.defvjp(_cce_fwd, _cce_bwd)


def chunked_cross_entropy(hidden, embedding, targets, pad_id, chunk_size=128):
    """Masked-mean CE over ALL rows — the original scalar API, now a thin wrapper
    over the per-row core. ``loss = Σ_rows sum_i / Σ_rows count_i`` with the stats
    aggregated the same way, so single-window callers (tests, tools) see exactly
    the old semantics. Gradients flow through the per-row sums with the correct
    1/total_count scale — identical to the old global-mean backward."""
    loss_sums, counts, stats = chunked_cross_entropy_rows(
        hidden, embedding, targets, pad_id, chunk_size)
    counts = jax.lax.stop_gradient(counts)
    denom_rows = counts.clip(min=1.0)
    total = counts.sum().clip(min=1.0)
    agg = {
        'out_entropy': jnp.sum(stats['out_entropy'] * denom_rows) / total,
        'logz_mean': jnp.sum(stats['logz_mean'] * denom_rows) / total,
        'max_abs_logit': jnp.max(stats['max_abs_logit']),
    }
    return loss_sums.sum() / total, agg
