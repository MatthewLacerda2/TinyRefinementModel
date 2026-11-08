"""TEMPORARY (#153): is XLA CPU row-count invariant for the ops a decode step runs?

Reports, in its failure message, whether each op gives row p bit for bit the same
when run on 1 (or a few) rows as on the full window. Deleted before the PR leaves draft.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.model.plain import PlainTransformer
from trm.settings import CONFIG

L = CONFIG.MAX_SEQ_LEN


def _model(dim, heads):
    return PlainTransformer(dim, nnx.Rngs(0), CONFIG, vocab_size=37, num_heads=heads,
                            num_layers=2, max_seq_len=L, pad_token_id=36)


def _cmp(full_row, part_row):
    full_row, part_row = np.asarray(full_row), np.asarray(part_row)
    return f"eq={np.array_equal(full_row, part_row)} maxdiff={np.max(np.abs(full_row - part_row)):.3g}"


def test_row_invariance_report():
    report = []
    for dim, heads in ((32, 4), (64, 4), (960, 15)):
        m = _model(dim, heads)
        blk = m.blocks[0]
        x = jax.random.normal(jax.random.PRNGKey(0), (1, L, dim), jnp.float32)
        p = 7
        for name, fn in (("linear_q", blk.attn.q), ("norm1", blk.norm1), ("gate", blk.gate_proj)):
            f = nnx.jit(lambda mod, a: mod(a))
            full = f(fn, x)
            for rows in (1, 2, 8):
                part = f(fn, x[:, p:p + rows])
                report.append(f"d{dim} {name} M={rows}: {_cmp(full[:, p], part[:, 0])}")
        h = jax.random.normal(jax.random.PRNGKey(1), (1, L, heads, dim // heads), jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(2), (1, L, heads, dim // heads), jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (1, L, heads, dim // heads), jnp.float32)
        pos = jnp.arange(L)
        bias = jnp.where(pos[:, None] >= pos[None, :], 0.0, -1e9)[None, None]
        att = jax.jit(lambda q, k, v, b: jax.nn.dot_product_attention(q, k, v, bias=b))
        full = att(h, k, v, bias)
        for rows in (1, 2, 8):
            part = att(h[:, p:p + rows], k, v, bias[:, :, p:p + rows])
            report.append(f"d{dim} attention M={rows}: {_cmp(full[:, p], part[:, 0])}")
        tokens = jnp.full((1, L), 36, jnp.int32).at[0, :12].set(jnp.arange(1, 13))
        fwd = nnx.jit(lambda mod, t: mod(t, training=True).hidden)
        report.append(f"d{dim} forward repeat: {_cmp(fwd(m, tokens), fwd(m, tokens))}")
    raise AssertionError("ROW-INVARIANCE REPORT\n" + "\n".join(report))
