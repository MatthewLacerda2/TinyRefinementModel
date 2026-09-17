"""The pre-run GPU smoke gates on f16 headroom, not just finiteness (#235).

The smoke exists to catch f16 overflow before a run starts. It ran clean through an
entire 10-day run while the model trained itself to **0.6%** of the f16 ceiling,
for two reasons it used to state as a feature:

1. it used random tokens, on the stated grounds that "text content is irrelevant to
   numerical health" — but the overflow is corpus-specific (65,120 on code, 47.7 on
   prose), and uniform random ids look like prose
2. it asserted only that loss and gradients were finite — and a model at 99.4% of the
   ceiling is finite

A gate that only fires after you have already gone over is a post-mortem. These guard
the arithmetic of the replacement; that it fires on the real champion and stays quiet
on prose is recorded in the PR.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN
from instruments import smoke_refiner_gpu as smoke

TOY_DIM = 32
TOY_VOCAB = 37
TOY_PAD = TOY_VOCAB - 1
TOY_LAYERS = 3


def _toy(arch, embed_scale=1.0):
    """A three-block model of `arch`: plain's whole stack, or the refiner's encoder."""
    if arch == "plain":
        from trm.model.plain import PlainTransformer
        model = PlainTransformer(TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=4,
                                 num_layers=TOY_LAYERS, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)
    else:
        from trm.model.refiner_lm import RefinerForTraining
        model = RefinerForTraining(TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=4,
                                   encoder_layers=TOY_LAYERS, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)
    embed, _ = smoke.traced_stack(model, arch)
    embed.embedding[...] = embed.embedding[...] * embed_scale
    return model


@pytest.fixture(scope="module", params=["plain", "refiner"])
def arch(request):
    return request.param


@pytest.fixture(scope="module")
def toy(arch):
    return _toy(arch)


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, :8].set(jnp.arange(1, 9, dtype=jnp.int32))


def test_headroom_is_reported_per_block_not_just_at_the_end(toy, tokens, arch):
    """#235's growth was not gradual — six blocks matched on both corpora and the
    seventh multiplied by ~794. An end-of-stack scalar says a run is unsafe; the
    per-block trace says where to look. `plain`, the live arch, traces every block."""
    peaks, worst, _ = smoke.block_headroom(toy, tokens, arch)
    assert len(peaks) == TOY_LAYERS, "one peak per traced block"
    assert worst == max(peaks)


def test_headroom_is_the_fraction_of_f16_left_unused(toy, tokens, arch):
    peaks, worst, headroom = smoke.block_headroom(toy, tokens, arch)
    assert headroom == pytest.approx(1.0 - worst / smoke.F16_MAX)
    assert 0.0 < headroom <= 1.0


def test_a_healthy_model_has_headroom_to_spare(toy, tokens, arch):
    """The counter-test. A gate that failed everything would satisfy the test below
    and block every run."""
    _, _, headroom = smoke.block_headroom(toy, tokens, arch)
    assert headroom >= smoke.MIN_HEADROOM


def test_a_model_pushed_toward_the_ceiling_loses_headroom(toy, tokens, arch):
    """Scaling the embedding drives activations up the way code-shaped input does on
    the real model. The gate must track that, or it measures nothing."""
    hot = _toy(arch, embed_scale=5000.0)

    _, calm_worst, calm = smoke.block_headroom(toy, tokens, arch)
    _, hot_worst, hot_hr = smoke.block_headroom(hot, tokens, arch)

    assert hot_worst > calm_worst
    assert hot_hr < calm, "headroom must fall as activations rise"


def test_the_smoke_wakes_every_residual_block_it_reads(toy, arch):
    """Zero-fractions are read only after each zero-init down_proj is woken; a block
    missing from this list would report structural zeros as f16 underflow."""
    blocks = smoke.residual_blocks(toy, arch)
    assert len(blocks) == TOY_LAYERS + (arch == "refiner")
    assert all(hasattr(b, "down_proj") for b in blocks)


def test_an_arch_without_a_trace_is_refused_by_name():
    with pytest.raises(SystemExit, match="'reasoner'"):
        smoke.refuse_untraced("reasoner")


def test_the_threshold_would_have_caught_the_champion():
    """The bar is set against a measured number, not a round one: the champion
    leaves 0.9% headroom on code and 99.9% on prose. Any threshold between those
    separates them; 50% is a factor of two, which tolerates ordinary growth."""
    champion_code_headroom = 1.0 - 64896.0 / smoke.F16_MAX
    champion_prose_headroom = 1.0 - 35.9 / smoke.F16_MAX

    assert champion_code_headroom < smoke.MIN_HEADROOM, "the gate must reject this"
    assert champion_prose_headroom >= smoke.MIN_HEADROOM, "and must not reject this"


def test_random_tokens_are_the_documented_fallback_not_the_default(monkeypatch, capsys):
    """Without a corpus the smoke still runs, but it must SAY that the thing it was
    rewritten to catch cannot be caught — a silent weaker test is how this gate
    passed for ten days."""
    monkeypatch.delenv("DATA_ROOT", raising=False)
    batch = smoke.load_batch(np.random.default_rng(0))

    assert batch.shape == (1, 2 * MAX_SEQ_LEN + 1)
    assert "CANNOT be caught" in capsys.readouterr().out
