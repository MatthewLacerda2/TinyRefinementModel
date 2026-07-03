"""Memory-lean causal attention (#66).

Blockwise attention via a custom VJP whose forward AND backward are ``lax.scan``
loops over query blocks — the same construction that fixed the vocab-wide CE
peak in losses.py (#19), applied to the other quadratic tensor. Stock
``jax.nn.dot_product_attention`` materializes [heads, s, s] scores and saves the
probabilities for the backward; at seq 512 / depth 8 those transients dominate
the compiled grad step (the 1.9 GiB peak mem_profile names, and what OOM'd the
dim960 fit-test). Here one query block's [heads, block_q, s] scores exist at a
time, and the backward saves NO probabilities — it recomputes each block from
(q, k, v), flash-attention's memory story in pure XLA (no Ampere kernels, so it
runs on the Turing card and the CPU test lane alike).

Numerics match the stock path: scores accumulate in f32 (``preferred_element_
type``), softmax runs in f32, the causal mask and additive pad bias apply to
scaled scores. Only the block loop reorders float ops, so parity is allclose,
not bit-exact — pinned by tests/test_chunked_attention.py in value and grads.
"""

from functools import partial

import jax
import jax.numpy as jnp

# Queries scored per scan step. 128 keeps the live block at [h, 128, s] —
# a single bounded transient instead of one [h, s, s] per attention call.
DEFAULT_BLOCK_Q = 128


def _block_scores(q_blk, k, row_start, pad_cols):
    """Scaled f32 scores for one query block, causal + pad bias applied.
    q_blk [b, Bq, h, d]; k [b, s, h, d]; pad_cols [b, s] additive f32."""
    d = q_blk.shape[-1]
    s = k.shape[1]
    scores = jnp.einsum('bqhd,bkhd->bhqk', q_blk, k,
                        preferred_element_type=jnp.float32) / jnp.sqrt(d).astype(jnp.float32)
    rows = row_start + jnp.arange(q_blk.shape[1])
    causal = rows[:, None] >= jnp.arange(s)[None, :]                # True = allowed
    bias = jnp.where(causal, 0.0, -1e9)[None, None, :, :] + pad_cols[:, None, None, :]
    return scores + bias


def _pad_to_blocks(q, block_q):
    """Pad the query axis up to a whole number of blocks and reshape to
    [n_blocks, b, Bq, h, d] for lax.scan. Padded rows are discarded by the
    caller's final slice; their all-masked softmax is uniform, not NaN."""
    b, s, h, d = q.shape
    n_pad = (-s) % block_q
    if n_pad:
        q = jnp.pad(q, ((0, 0), (0, n_pad), (0, 0), (0, 0)))
    n_blocks = (s + n_pad) // block_q
    return q.reshape(b, n_blocks, block_q, h, d).swapaxes(0, 1), n_blocks


def _forward(q, k, v, pad_cols, block_q):
    b, s, h, d = q.shape
    q_steps, n_blocks = _pad_to_blocks(q, block_q)
    starts = jnp.arange(n_blocks) * block_q

    def step(_, xs):
        q_blk, row_start = xs
        probs = jax.nn.softmax(_block_scores(q_blk, k, row_start, pad_cols), axis=-1)
        out_blk = jnp.einsum('bhqk,bkhd->bqhd', probs.astype(v.dtype), v)
        return None, out_blk

    _, out_steps = jax.lax.scan(step, None, (q_steps, starts))
    return out_steps.swapaxes(0, 1).reshape(b, n_blocks * block_q, h, d)[:, :s]


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def chunked_causal_attention(q, k, v, pad_cols, block_q=DEFAULT_BLOCK_Q):
    """Causal multi-head attention over [b, s, h, d] inputs, scored one query
    block at a time. ``pad_cols`` is the additive key-padding bias [b, s]
    (0 = attend, -1e9 = masked); it takes no gradient (it comes from a token
    comparison upstream). Drop-in for the stock
    ``jax.nn.dot_product_attention(q, k, v, bias=causal+pad)`` up to float
    summation order."""
    return _forward(q, k, v, pad_cols, block_q)


def _cca_fwd(q, k, v, pad_cols, block_q):
    # Residuals are just the inputs — no [s, s] probabilities are saved; the
    # backward recomputes each block. This is the entire memory win.
    return _forward(q, k, v, pad_cols, block_q), (q, k, v, pad_cols)


def _cca_bwd(block_q, residuals, g):
    q, k, v, pad_cols = residuals
    b, s, h, d = q.shape
    scale = jnp.sqrt(d).astype(jnp.float32)

    q_steps, n_blocks = _pad_to_blocks(q, block_q)
    g_steps, _ = _pad_to_blocks(g, block_q)
    starts = jnp.arange(n_blocks) * block_q

    def step(carry, xs):
        dk_acc, dv_acc = carry
        q_blk, g_blk, row_start = xs
        probs = jax.nn.softmax(_block_scores(q_blk, k, row_start, pad_cols), axis=-1)  # [b,h,Bq,s] f32

        g32 = g_blk.astype(jnp.float32)
        # dV += P^T dO ; dP = dO V^T ; softmax bwd: dS = P * (dP - sum(dP*P))
        dv_blk = jnp.einsum('bhqk,bqhd->bkhd', probs, g32)
        dp = jnp.einsum('bqhd,bkhd->bhqk', g32, v.astype(jnp.float32))
        ds = probs * (dp - jnp.sum(dp * probs, axis=-1, keepdims=True))
        dq_blk = jnp.einsum('bhqk,bkhd->bqhd', ds, k.astype(jnp.float32)) / scale
        dk_blk = jnp.einsum('bhqk,bqhd->bkhd', ds, q_blk.astype(jnp.float32)) / scale
        return (dk_acc + dk_blk, dv_acc + dv_blk), dq_blk

    zeros = jnp.zeros((b, s, h, d), jnp.float32)
    (dk, dv), dq_steps = jax.lax.scan(step, (zeros, zeros), (q_steps, g_steps, starts))
    dq = dq_steps.swapaxes(0, 1).reshape(b, n_blocks * block_q, h, d)[:, :s]

    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype),
            jnp.zeros_like(pad_cols))


chunked_causal_attention.defvjp(_cca_fwd, _cca_bwd)
