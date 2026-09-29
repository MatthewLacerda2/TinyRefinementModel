"""The deep-narrow shape is selectable by knobs, and today's shape did not move (#433).

#433 adopts SmolLM2-135M's shape: 30 layers x 576, 9 heads sharing 3 K/V heads (GQA).
It is four Config knobs (LATENT_DIM, NUM_HEADS, NUM_KV_HEADS, PLAIN_LAYERS); the
defaults stay 8 x 960 with one K/V head per query head, so every stored checkpoint and
the golden run build the same network from the same init draws.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.model import build_model
from trm.model.plain import PlainTransformer
from trm.settings import CONFIG, Config

TOY_DIM, TOY_VOCAB, TOY_HEADS, TOY_LAYERS = 48, 37, 6, 2
TOY_PAD = TOY_VOCAB - 1


def toy(num_kv_heads=None, seed=0):
    return PlainTransformer(TOY_DIM, nnx.Rngs(seed), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
                            num_layers=TOY_LAYERS, pad_token_id=TOY_PAD, num_kv_heads=num_kv_heads)


def tokens(seed=0):
    return jax.random.randint(jax.random.PRNGKey(seed), (2, 32), 0, TOY_PAD, dtype=jnp.int32)


def params(model):
    return jax.tree.leaves(nnx.state(model, nnx.Param))


def test_one_kv_head_per_query_head_is_the_network_before_the_knob():
    """MHA spelled out is the default, bit for bit: same tree, same init draws."""
    for a, b in zip(params(toy()), params(toy(num_kv_heads=TOY_HEADS)), strict=True):
        assert a.shape == b.shape and jnp.array_equal(a, b)
    assert toy().blocks[0].attn.k.kernel.shape == (TOY_DIM, TOY_DIM)


def test_gqa_projects_keys_and_values_to_fewer_heads():
    attn = toy(num_kv_heads=2).blocks[0].attn
    head_dim = TOY_DIM // TOY_HEADS
    assert attn.q.kernel.shape == (TOY_DIM, TOY_DIM)
    assert attn.k.kernel.shape == attn.v.kernel.shape == (TOY_DIM, 2 * head_dim)


def test_gqa_is_mha_with_each_kv_head_repeated_over_its_group():
    """The grouping convention: query head n reads K/V head n // group. An MHA model
    whose K/V weights repeat each GQA head over its group must predict identically;
    a tiled (n % kv) grouping, or a reshape that splits heads wrongly, would not."""
    kv, group = 2, TOY_HEADS // 2
    head_dim = TOY_DIM // TOY_HEADS
    gqa, mha = toy(num_kv_heads=kv, seed=1), toy(seed=2)
    nnx.update(mha, nnx.filter_state(nnx.state(gqa, nnx.Param), lambda path, _: path[-2] not in ("k", "v")))

    def repeat(x):   # [dim, kv * hd] -> [dim, heads * hd], or [kv * hd] -> [heads * hd]
        split = x.reshape(*x.shape[:-1], kv, head_dim)
        return jnp.repeat(split, group, axis=-2).reshape(*x.shape[:-1], TOY_HEADS * head_dim)

    for g_block, m_block in zip(gqa.blocks, mha.blocks, strict=True):
        for name in ("k", "v"):
            g, m = getattr(g_block.attn, name), getattr(m_block.attn, name)
            m.kernel[...] = repeat(g.kernel[...])
            m.bias[...] = repeat(g.bias[...])
        # A zero-initialized down_proj would hide attention from the output entirely.
        m_block.down_proj.kernel[...] = g_block.down_proj.kernel[...] = jnp.full_like(
            g_block.down_proj.kernel[...], 0.02)
    np.testing.assert_allclose(gqa(tokens()).logits, mha(tokens()).logits, rtol=1e-5, atol=1e-5)


def test_gqa_stays_causal():
    model, t = toy(num_kv_heads=3), tokens()
    for block in model.blocks:
        block.down_proj.kernel[...] = jnp.full_like(block.down_proj.kernel[...], 0.02)
    a = model(t).logits
    b = model(t.at[:, 10].set((t[:, 10] + 1) % TOY_PAD)).logits
    assert jnp.array_equal(a[:, :10], b[:, :10]), "a later token moved an earlier prediction"
    assert not jnp.array_equal(a[:, 10:], b[:, 10:])


def test_unset_kv_heads_follow_the_head_count():
    """Changing NUM_HEADS alone stays MHA; a KV count that does not divide it refuses."""
    assert Config.from_env({"NUM_HEADS": "9"}).NUM_KV_HEADS == 9
    assert Config.from_env({"NUM_HEADS": "9", "NUM_KV_HEADS": "3"}).NUM_KV_HEADS == 3
    with pytest.raises(SystemExit, match="NUM_KV_HEADS"):
        Config.from_env({"NUM_HEADS": "9", "NUM_KV_HEADS": "2"})


def n_params(config):
    shapes = nnx.eval_shape(lambda: build_model(config, nnx.Rngs(0)))
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(nnx.state(shapes, nnx.Param)))


def test_smollm2_shape_is_selectable_through_the_config_and_is_its_size():
    """30 x 576 with GQA 9/3 builds from knobs alone, at SmolLM2-135M's size, and the
    8/3 FFN rounding gives its 1536 without a knob of its own."""
    smol = Config.from_env({"LATENT_DIM": "576", "NUM_HEADS": "9", "NUM_KV_HEADS": "3", "PLAIN_LAYERS": "30"})
    model = nnx.eval_shape(lambda: build_model(smol, nnx.Rngs(0)))
    assert len(model.blocks) == 30
    assert model.blocks[0].attn.k.kernel.shape == (576, 3 * 64)
    assert model.blocks[0].gate_proj.kernel.shape == (576, 1536)
    assert 134e6 < n_params(smol) < 136e6
    # Today's 8 x 960 for comparison: the same budget, spent on width.
    assert 136e6 < n_params(Config()) < 138e6
