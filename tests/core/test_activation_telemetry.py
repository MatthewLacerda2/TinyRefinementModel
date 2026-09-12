"""Peak activation is logged during training, and flagged before it overflows (#235).

The 4B run finished with the encoder's output at **65,120** against an f16 ceiling of
**65,504** — 0.6% of headroom. That is what #229's whole-window NaN was, and it was
found two weeks after the run ended, by hand, because nothing watched it.

Two halves:

- the model reports `act_max` on `diag`, the channel that already reaches metrics.csv
- `instruments/invariants` flags it past **half** the ceiling — a MARGIN, not a
  pass/fail. A check that only fires once you have already overflowed is a
  post-mortem (CLAUDE.md rule 1).
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from instruments.invariants import F16_MAX, row_violations
from trm.config import MAX_SEQ_LEN
from trm.model.plain import PlainTransformer

TOY_VOCAB, TOY_PAD = 37, 36


@pytest.fixture(scope="module")
def toy():
    return PlainTransformer(32, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=4,
                            num_layers=3, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)


@pytest.fixture(scope="module")
def tokens():
    t = jnp.full((1, MAX_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return t.at[0, :6].set(jnp.arange(1, 7, dtype=jnp.int32))


@pytest.mark.parametrize("training", [True, False])
def test_act_max_is_reported_on_both_paths(toy, tokens, training):
    """Training is the path that matters — a run must be able to see this while it
    happens — but inference reporting it too means an instrument gets the same
    number without a second code path to drift."""
    diag = toy(tokens, depth=1, training=training).diag
    assert "act_max" in diag
    assert float(diag["act_max"]) > 0.0


def test_it_is_the_peak_not_the_average(toy, tokens):
    """A mean would understate exactly the quantity being guarded: one block out of
    seven multiplied by ~794 in #235, and an average over blocks would have hidden
    it."""
    scaled = PlainTransformer(32, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=4,
                              num_layers=3, max_seq_len=MAX_SEQ_LEN, pad_token_id=TOY_PAD)
    scaled.embed.embedding.value = scaled.embed.embedding.value * 100.0

    calm = float(toy(tokens, depth=1, training=True).diag["act_max"])
    hot = float(scaled(tokens, depth=1, training=True).diag["act_max"])
    assert hot > calm * 10, "act_max must track the peak it is supposed to report"


def test_the_metrics_logger_knows_the_key():
    """A model reporting a diagnostic nothing writes down is a diagnostic that does
    not exist. This is the seam #105 made arch-optional, and the seam where a new
    metric silently goes nowhere."""
    from trm.runtime.metrics import MetricsLogger

    logger = MetricsLogger.__new__(MetricsLogger)
    MetricsLogger.__init__(logger, history_file="/dev/null")
    assert "act_max" in logger.diag_keys
    assert "act_max" in logger.fields


# --- the margin, not the cliff -----------------------------------------------

def test_an_ordinary_activation_is_clean():
    assert not row_violations({"step": 1, "act_max": 100.0})


def test_it_fires_with_headroom_left_rather_than_after_overflow():
    """The whole point. At 40,000 there is still a run to save; at 65,120 — where the
    4B champion actually finished — it is already one rounding from NaN."""
    assert row_violations({"step": 1, "act_max": 40_000.0})
    assert row_violations({"step": 1, "act_max": 65_120.0})
    assert 40_000.0 < F16_MAX, "the fixture must sit BELOW the ceiling, or it proves nothing"


def test_the_champions_final_value_would_have_been_flagged():
    """Stated as the regression it is: this run completed without a single warning."""
    assert row_violations({"step": 30_464, "act_max": 65_120.0})
