"""TEMPORARY (#153): the measured cached-vs-uncached gap with live MLPs. Deleted before
the PR leaves draft."""

import jax
import jax.numpy as jnp
from flax import nnx

from tests.core import test_kv_cache as kv
from trm.model.plain import PlainTransformer


def test_gap_report():
    report = []
    for dim, heads in ((32, 4), (960, 15)):
        for post_norm in (False, True):
            model = PlainTransformer(dim, nnx.Rngs(0), kv.CONFIG, vocab_size=kv.TOY_VOCAB,
                                     num_heads=heads, num_layers=2, max_seq_len=kv.WINDOW,
                                     pad_token_id=kv.TOY_PAD, post_norm=post_norm)
            for i, blk in enumerate(model.blocks):
                k = blk.down_proj.kernel
                k.value = jax.random.normal(jax.random.PRNGKey(i), k.value.shape) * k.value.shape[0] ** -0.5
            rows = kv._walk(model)
            gaps = [kv._gap(*r) for r in rows]
            argmax = sum(int(jnp.argmax(a)) == int(jnp.argmax(b)) for a, b in rows)
            report.append(f"d{dim} post_norm={post_norm}: first={gaps[0]:.3g} max={max(gaps):.3g} "
                          f"argmax agree {argmax}/{len(rows)}")
        late = max(kv._gap(*r) for r in kv._walk(model, feed_at=lambda pos: pos + 1)[1:])
        report.append(f"d{dim} one-late max gap={late:.3g}")
    raise AssertionError("GAP REPORT\n" + "\n".join(report))
