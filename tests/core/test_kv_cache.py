"""Generation through a KV cache (#153): one row per token, same tokens out.

The bar the issue set was bit-identical logits. Measured: the cached path and
the padded-window path agree to ~2e-6 in f32 — the float reduction order over
the masked, zero-weight tail of the window differs, nothing else — and every
greedy token is identical. The tolerance below is 1e-5, ten times the measured
gap; the gap itself is recorded in the PR.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from instruments.arch import build
from trm.config import MAX_SEQ_LEN, PAD_TOKEN_ID


@pytest.fixture(scope="module")
def toy_plain():
    return build("plain", dim=60, num_layers=2, seed=0)


def _window_logits(model, ids):
    padded = np.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, np.int32)
    padded[0, :len(ids)] = ids
    return np.asarray(model(jnp.asarray(padded), training=False, logits_at=len(ids) - 1).logits[0, 0])


def test_prefill_and_decode_match_the_padded_window_and_pick_the_same_tokens(toy_plain):
    rng = np.random.default_rng(0)
    ids = list(rng.integers(1, 50000, size=12))
    row, caches = toy_plain.prefill(jnp.asarray([ids], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(row[0]), _window_logits(toy_plain, ids), atol=1e-5, rtol=0)
    for _ in range(20):
        nxt = int(np.argmax(np.asarray(row[0])))
        assert nxt == int(np.argmax(_window_logits(toy_plain, ids))), "greedy choice must agree"
        ids.append(nxt)
        row, caches = toy_plain.decode_step(jnp.asarray([[nxt]], dtype=jnp.int32), caches, jnp.int32(len(ids) - 1))
        np.testing.assert_allclose(np.asarray(row[0]), _window_logits(toy_plain, ids), atol=1e-5, rtol=0)


def test_generate_text_uses_the_cache_for_the_plain_model_and_reproduces_greedy_text(toy_plain, monkeypatch):
    from trm import infer

    class Enc:
        def encode(self, text):
            return [7, 11, 13, 17, 19]

        def decode(self, ids):
            return ""
    cached = infer.generate_text(toy_plain, Enc(), "x", max_new_tokens=15, temperature=0.0, quiet=True)
    monkeypatch.setattr(type(toy_plain), "prefill", property(lambda self: (_ for _ in ()).throw(AttributeError())))
    assert not hasattr(toy_plain, "prefill")
    window = infer.generate_text(toy_plain, Enc(), "x", max_new_tokens=15, temperature=0.0, quiet=True)
    assert cached == window and len(cached) > 5
    assert "_generate_cached" in inspect.getsource(infer.generate_text)


def test_the_refiner_keeps_the_window_path():
    """The retired arch has no cache; generate_text must not assume one."""
    from trm.model.refiner_lm import RefinerForTraining
    assert not hasattr(RefinerForTraining, "prefill")
