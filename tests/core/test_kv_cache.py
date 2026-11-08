"""Generation through a KV cache gives the uncached path's logits (#153).

The cache is a refactor of generation: the window once (`prefill`), then one token
per step against every block's cached K and V (`decode_step`), where the uncached
path runs the whole padded window per token. Same model, same arithmetic.

Not the same bits after the first token, and that cannot be fixed from this side:
float matmul is not row-count invariant on XLA's CPU backend (#206 found it for the
head). Measured on CI for #628, one row run alone against the same row inside the
window's [512, dim] GEMM: the q projection differs by up to 2.4e-7 at dim 32 and
3.1e-6 at dim 960, the attention row by up to 3.6e-7, while every reduction (the
norms) is exact. Running fewer rows is the cache's whole saving, so the decode step
cannot match the window's shapes back.

So the gate has three parts:
- **bit for bit** where the shapes do match: the first token's logits, which prefill
  computes at the window's own shapes;
- **within kernel noise** after that (#206's bound, orders of magnitude under the
  val-CE noise floor), with an **exact argmax** at every step;
- **the bound can see a cache bug**: a token fed one position late, the classic one,
  misses it by orders of magnitude.

Measured on CI for #628 over this file's 32-step walk, MLPs live: the first token
exact, the worst later gap 2.4e-6 at dim 32 and 5.3e-6 at dim 960 (both arms of
POST_NORM), every argmax the same; one position late, 1.4 at dim 32 and 0.59 at dim 960.

What the training forward computes is pinned in test_training_forward_pin.py; the
last test here holds that generating leaves the model itself alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm import infer
from trm.config import COMPUTE_DTYPE
from trm.model.plain import PlainTransformer
from trm.settings import CONFIG

TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_LAYERS = 2
TOY_PAD = TOY_VOCAB - 1
# The window generation serves, so prefill and decode run at the shapes they serve.
WINDOW = CONFIG.MAX_SEQ_LEN
# A pad inside the prompt: both paths must mask that key, not only the tail.
PROMPT = [5, 9, TOY_PAD, 14, 3, 21, 8]
STEPS = 32
# temperature 1, no top-k, no top-p: truncation is the identity, so these are raw logits.
RAW = dict(top_k=0, top_p=1.0, temperature=1.0)
SAMPLED = dict(top_k=8, top_p=0.9, temperature=0.7)
# test_infer_head_slice.py's bound for the same kernel-selection noise: f32's is
# measured above; f16 compute (the GPU suite) moves up to one f16 ulp (#452).
RTOL = max(1e-4, 2 * float(jnp.finfo(COMPUTE_DTYPE).eps))
ATOL = 1e-5


def _toy(post_norm=False):
    """A toy whose MLPs are live: `down_proj` starts at zero (Block), which at init
    would leave the MLP half of `Block.decode` outside everything this file checks."""
    model = PlainTransformer(
        TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=WINDOW, pad_token_id=TOY_PAD, post_norm=post_norm)
    for i, blk in enumerate(model.blocks):
        kernel = blk.down_proj.kernel
        fan_in = kernel.value.shape[0]
        kernel.value = jax.random.normal(jax.random.PRNGKey(i), kernel.value.shape,
                                         kernel.value.dtype) * fan_in ** -0.5
    return model


@pytest.fixture(scope="module")
def toy():
    return _toy()


def _window(ids):
    padded = np.full((1, WINDOW), TOY_PAD, np.int32)
    padded[0, :len(ids)] = ids
    return jnp.asarray(padded)


def _uncached(model, ids, sampling):
    """The uncached path: the whole window, read at the last real position."""
    return infer.get_logits_for_token(model, _window(ids), len(ids) - 1, **sampling)


def _gap(cached, uncached):
    """Max |difference| over the entries both rows kept (top-k/top-p set the rest to -inf)."""
    cached, uncached = np.asarray(cached), np.asarray(uncached)
    kept = np.isfinite(cached) & np.isfinite(uncached)
    return float(np.max(np.abs(cached[kept] - uncached[kept]))) if kept.any() else float("nan")


def _assert_agree(step, cached, uncached):
    """Within kernel noise, the same tokens kept by top-k/top-p, the same winner."""
    gap = _gap(cached, uncached)
    assert jnp.allclose(cached, uncached, rtol=RTOL, atol=ATOL), (
        f"step {step}: cached and uncached logits differ by {gap:.3g}, more than kernel "
        f"noise: a real change, not a different reduction order")
    assert jnp.array_equal(jnp.isinf(cached), jnp.isinf(uncached)), (
        f"step {step}: top-k/top-p kept a different set of tokens")
    assert int(jnp.argmax(cached)) == int(jnp.argmax(uncached)), (
        f"step {step}: the winning token changed (gap {gap:.3g})")


def _walk(model, feed_at=lambda pos: pos):
    """Prefill PROMPT, then STEPS greedy tokens through decode, each fed at
    `feed_at(its position)`: every step's (cached, uncached) row, the uncached one
    from the window run on the same prefix."""
    ids = list(PROMPT)
    logits, cache = infer.prefill_for_token(model, _window(ids), len(ids) - 1, **RAW)
    rows = [(logits, _uncached(model, ids, RAW))]
    for _ in range(STEPS):
        # Greedy over real tokens only, so the walk never stops on a pad.
        ids.append(int(np.argmax(np.asarray(logits)[:TOY_PAD])))
        newest = jnp.asarray([[ids[-1]]], dtype=jnp.int32)
        logits, cache = infer.decode_for_token(model, newest, cache, feed_at(len(ids) - 1), **RAW)
        rows.append((logits, _uncached(model, ids, RAW)))
    return rows


@pytest.mark.parametrize("post_norm", [False, True])
def test_cached_decoding_gives_the_uncached_logits(post_norm):
    """The acceptance gate: the first token bit for bit, every later one within noise
    and with the same winner."""
    rows = _walk(_toy(post_norm))
    # Two programs at the same shapes: shown bit-equal on XLA's CPU backend. On the
    # card a GEMM algorithm may be chosen per program, so there the bound applies.
    if jax.default_backend() == "cpu":
        assert jnp.array_equal(*rows[0]), (
            f"prefill runs the window's own computation, yet differs by {_gap(*rows[0]):.3g}")
    for step, (cached, uncached) in enumerate(rows):
        _assert_agree(step, cached, uncached)


def test_the_bound_sees_a_token_fed_one_position_late(toy):
    """A tolerance is a gate only if a cache bug clears it: one position late (every
    key read at the wrong RoPE distance) must miss by far more than the noise."""
    worst = max(_gap(*row) for row in _walk(toy, feed_at=lambda pos: pos + 1)[1:])
    assert worst > 100 * ATOL, f"an off-by-one cache moved the logits by only {worst:.3g}"


def test_the_causal_mask_hides_stale_keys_past_the_position(toy):
    """In generation every key past `pos` is a pad, so the pad bias alone would mask
    it and an off-by-one causal term would go unseen. Here they are real tokens: a
    longer prefix prefilled, then a token decoded over its middle. Only the causal
    term keeps the stale keys out, and the row must be the shorter window's."""
    ids = list(PROMPT)
    _, cache = infer.prefill_for_token(toy, _window(ids + [11, 12, 13, 14]), len(ids) - 1, **RAW)
    newest = jnp.asarray([[17]], dtype=jnp.int32)
    logits, _ = infer.decode_for_token(toy, newest, cache, len(ids), **RAW)
    _assert_agree(0, logits, _uncached(toy, ids + [17], RAW))


def test_generate_text_samples_from_the_uncached_logits(toy, in_vocab_encoder, monkeypatch):
    """The wiring: generate_text takes the cache, and every row it samples from (after
    temperature, top-k and top-p) agrees with the uncached path's for the same tokens."""
    enc = in_vocab_encoder(TOY_VOCAB)
    seen = []
    monkeypatch.setattr(infer, "reject_unsampleable", lambda logits, **_: seen.append(logits))

    def window_ran(*_, **__):
        raise AssertionError("generate_text ran the window per token instead of the cache")

    monkeypatch.setattr(infer, "get_logits_for_token", window_ran)
    prompt = "the cache"
    tokens = infer.generate_text(toy, enc, prompt, max_new_tokens=STEPS, seed=7, quiet=True,
                                 **SAMPLED)
    monkeypatch.undo()

    n = len(enc.encode(prompt))
    assert len(seen) >= 8, "too few steps to say anything; pick another seed"
    for step, cached in enumerate(seen):
        _assert_agree(step, cached, _uncached(toy, tokens[:n + step], SAMPLED))


def _compilations_during(thunk):
    """XLA compilations triggered by `thunk` (as in test_infer_compile_count.py)."""
    import jax._src.compiler as compiler

    count = 0
    original = compiler.compile_or_get_cached

    def counting(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    compiler.compile_or_get_cached = counting
    try:
        thunk()
    finally:
        compiler.compile_or_get_cached = original
    return count


def test_the_next_decode_step_compiles_nothing(toy):
    """#207's rule for the cached path: the position and the token are traced, so the
    second token reuses the first one's executable."""
    ids = list(PROMPT)
    _, cache = infer.prefill_for_token(toy, _window(ids), len(ids) - 1, **SAMPLED)
    first, second = (jnp.asarray([[t]], dtype=jnp.int32) for t in (4, 11))
    logits, cache = infer.decode_for_token(toy, first, cache, len(ids), **SAMPLED)
    logits.block_until_ready()

    def next_token():
        out, _ = infer.decode_for_token(toy, second, cache, len(ids) + 1, **SAMPLED)
        out.block_until_ready()

    assert _compilations_during(next_token) == 0, "a per-token argument is in the jit cache key"


def test_generating_leaves_the_model_and_its_training_forward_alone(in_vocab_encoder):
    """The cache lives in the generation loop, never on the model: the state tree a
    checkpoint saves and restores (paths, shapes, dtypes, values) and the training
    forward's output are the same after a generation as before it."""
    model = _toy()
    tokens = _window(PROMPT)

    def snapshot():
        leaves = jax.tree_util.tree_leaves_with_path(nnx.state(model))
        return ([(jax.tree_util.keystr(path), leaf.shape, leaf.dtype) for path, leaf in leaves],
                [np.asarray(leaf) for _, leaf in leaves])

    def training_forward():
        out = model(tokens, training=True)
        return [np.asarray(x) for x in jax.tree_util.tree_leaves((out.hidden, out.diag))]

    tree_before, values_before = snapshot()
    forward_before = training_forward()
    infer.generate_text(model, in_vocab_encoder(TOY_VOCAB), "abc", max_new_tokens=8,
                        seed=0, quiet=True, **SAMPLED)
    tree_after, values_after = snapshot()

    assert tree_after == tree_before, "generating added or reshaped model state"
    assert all(np.array_equal(a, b) for a, b in zip(values_before, values_after, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(forward_before, training_forward(), strict=True))
