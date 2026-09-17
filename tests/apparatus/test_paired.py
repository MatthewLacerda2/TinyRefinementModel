"""The paired-comparison instrument computes what it claims (#230).

This exists because the statistic was got wrong by hand. Comparing one set of
weights at two depths on the same tokens is a PAIRED test, and the noise floor from
#17 -- seed-to-seed variance across training runs, sigma ~ 0.03 nats -- is the wrong
bar for it by about two orders of magnitude. It measures a different source of
variation: training two models, not scoring one model twice. Reaching for it made a
real effect look like noise for a day.

So the arithmetic is pinned against distributions whose answers are known by
construction, and the contrast requirement is pinned too, because a pooled number
with nothing to compare it against is what produced every wrong turn.
"""

import numpy as np
import pytest

from instruments.paired import Paired, _Accumulator, compare


def _acc(values):
    a = _Accumulator()
    a.add(np.asarray(values, dtype=np.float64))
    return a


def test_the_mean_and_error_match_the_textbook_formulas():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    got = _acc(values).result("x")
    assert got.mean == pytest.approx(3.0)
    assert got.se == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))
    assert got.n == 5


def test_streaming_in_chunks_equals_measuring_all_at_once():
    """The accumulator exists so a corpus never has to fit in memory. If chunking
    changed the answer, that saving would have cost correctness."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.01, 1.0, size=5000)

    whole = _acc(data).result("x")
    streamed = _Accumulator()
    for chunk in np.array_split(data, 37):
        streamed.add(chunk)
    part = streamed.result("x")

    assert part.mean == pytest.approx(whole.mean, rel=1e-12)
    assert part.se == pytest.approx(whole.se, rel=1e-12)


def test_a_real_effect_clears_two_sigma_and_noise_does_not():
    """Both directions of the instrument's actual job, on data built to have the
    answer. A tool that called everything significant would pass a one-sided test."""
    rng = np.random.default_rng(1)
    real = _acc(rng.normal(0.05, 1.0, size=40000)).result("real")
    noise = _acc(rng.normal(0.0, 1.0, size=40000)).result("noise")

    assert abs(real.t) > 2.0 and real.verdict == "treatment better"
    assert abs(noise.t) < 2.0 and noise.verdict == "indistinguishable"


def test_the_sign_convention_is_treatment_positive():
    """Stated in the dataclass and pinned here, because a sign convention nobody
    wrote down is one somebody will misread."""
    assert _acc([0.5, 0.5, 0.5, 0.5]).result("x").verdict == "treatment better"
    assert _acc([-0.5, -0.5, -0.5, -0.5]).result("x").verdict == "control better"


def test_non_finite_positions_are_counted_and_dropped_not_averaged():
    """#229 returns all-NaN logits on ~10% of code documents and #233 poisons a
    whole window. Folding either into a mean is how a bad number becomes a finding."""
    got = _acc([1.0, 2.0, np.nan, 3.0, np.inf]).result("x")
    assert got.nonfinite == 2
    assert got.n == 3
    assert got.mean == pytest.approx(2.0)


def test_too_few_points_reports_no_error_rather_than_inventing_one():
    got = _acc([1.0]).result("x")
    assert got.n == 1 and got.se == 0.0
    assert got.verdict == "indistinguishable", "one point cannot distinguish anything"


def test_a_single_corpus_is_refused():
    """The contrast is structural. Every real conclusion in the depth work came from
    comparing code against prose; every wrong turn came from a pooled number."""
    with pytest.raises(ValueError, match="at least two corpora"):
        compare(object(), {"only-one": []}, treatment_depth=8, control_depth=1,
                pad_token_id=0)


def test_the_summary_line_carries_the_error_not_just_the_estimate():
    """A point estimate printed without its error is how 0.0008 becomes a finding.
    Doctrine rule 1: report the spread, not just the point estimate."""
    line = str(Paired("codeparrot", 0.000828, 0.000777, 12261, 0))
    assert "se" in line and "0.000777" in line
    assert "indistinguishable" in line


def test_zero_variance_is_maximal_signal_not_absent_signal():
    """A difference identical at every position is perfectly consistent, so it is
    the MOST distinguishable result possible. The first version of `t` returned 0.0
    whenever se == 0, which reported the cleanest imaginable effect as noise. Caught
    by the sign-convention test above."""
    clean = _acc([0.5, 0.5, 0.5, 0.5]).result("x")
    assert clean.se == 0.0
    assert clean.t == float("inf")
    assert clean.verdict == "treatment better"

    nothing = _acc([0.0, 0.0, 0.0]).result("x")
    assert nothing.t == 0.0 and nothing.verdict == "indistinguishable"


def test_a_plain_model_is_refused_before_anything_loads(monkeypatch):
    """Plain ignores depth, so both arms would score one forward pass and every
    difference would be an exact zero (#317). Refused by name, before a restore."""
    import instruments.paired as paired
    monkeypatch.setattr("trm.config.MODEL_ARCH", "plain")
    monkeypatch.setattr("trm.runtime.restore.restore_model",
                        lambda *a, **k: pytest.fail("restored a model it should have refused"))
    with pytest.raises(SystemExit, match="MODEL_ARCH='plain'"):
        paired._main(["--checkpoint", "nowhere"])
