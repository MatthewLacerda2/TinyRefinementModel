"""The referee must reproduce history before it is allowed to judge anything new.

Three completed, pre-registered experiments are checked in as spec files with the
per-seed numbers their findings published (#40 build 1). This suite asserts that
`instruments/verdict.py` recomputes their sigma figures and lands on their
recorded verdicts. That is the whole warrant for trusting it on a fresh run: not
that the arithmetic looks right, but that it agrees with every judgement the
project has already made and written down.

If a spec here ever disagrees with its finding, one of the two is wrong — do not
"fix" the spec to match. Read both.
"""

import math
import pathlib

import pytest

from instruments.verdict import (
    INCONCLUSIVE,
    KEEP,
    KILL,
    Summary,
    evaluate,
    load_recorded_results,
    load_spec,
    mean_sigma,
    pooled_sigma,
    separation,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = {
    "86": REPO_ROOT / "experiments/depth/specs/086-sinusoidal-time-signal.toml",
    "77": REPO_ROOT / "experiments/depth/specs/077-per-pass-supervision.toml",
    "79": REPO_ROOT / "experiments/scratchpad/specs/079-dense-supervision-without-slots.toml",
}


# --- the convention, pinned against published numbers ------------------------

def test_sigma_is_the_sample_sigma_the_findings_used():
    """statistics.stdev (ddof=1), not the population sigma. Checked against a
    figure the #77 finding published: the 6-seed control's 0.254."""
    control_6seed = [0.958, 0.981, 0.977, 0.352, 0.972, 0.984]
    mean, sigma = mean_sigma(control_6seed)
    assert mean == pytest.approx(0.871, abs=0.001)
    assert sigma == pytest.approx(0.254, abs=0.001)


def test_pooled_sigma_reproduces_the_86_finding():
    """#86 leg B published sigma_pooled = 0.147 at d12. Recompute it from the
    per-seed numbers in the same finding."""
    table = [0.2050, 0.2042, 0.2052]
    sinusoidal = [0.7672, 0.4082, 0.7693]
    assert pooled_sigma(sinusoidal, table) == pytest.approx(0.147, abs=0.001)
    assert separation(sinusoidal, table) == pytest.approx(3.0, abs=0.05)


def test_separation_reproduces_the_77_finding():
    """#77 published -3.46 sigma for islands vs the 3-seed control, and +0.64
    sigma for the supervised arm at 6 seeds. Both fall out of the same call."""
    control_3 = [0.958, 0.981, 0.977]
    islands = [0.712, 0.724, 0.485]
    assert separation(islands, control_3) == pytest.approx(-3.46, abs=0.02)

    control_6 = [0.958, 0.981, 0.977, 0.352, 0.972, 0.984]
    supervised = [0.986, 0.989, 0.982, 0.988, 0.983, 0.988]
    assert separation(supervised, control_6) == pytest.approx(0.64, abs=0.02)


def test_separation_reproduces_the_79_finding():
    """#79 published serial - densedepth = +0.803, about 15.6x sigma_pooled."""
    serial = [0.9988, 0.9790, 0.9988]
    densedepth = [0.2703, 0.1323, 0.1655]
    assert separation(serial, densedepth) == pytest.approx(15.6, abs=0.2)


# --- degenerate inputs, which a referee meets eventually ----------------------

def test_zero_spread_does_not_produce_a_nan_verdict():
    """Toy tasks do produce perfectly repeatable arms. A real gap with zero
    pooled sigma is decided, not undefined; an exact tie is zero, not NaN."""
    assert separation([1.0, 1.0], [0.5, 0.5]) == math.inf
    assert separation([0.5, 0.5], [1.0, 1.0]) == -math.inf
    assert separation([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_a_single_seed_has_no_spread():
    assert mean_sigma([0.9]) == (0.9, 0.0)


def test_summary_arms_read_back_as_given():
    s = Summary(mean=0.8144, sigma=0.0281, n=3)
    assert mean_sigma(s) == (0.8144, 0.0281)
    with pytest.raises(ValueError):
        Summary(mean=0.5, sigma=-1.0, n=3)
    with pytest.raises(ValueError):
        Summary(mean=0.5, sigma=0.1, n=0)


# --- the specs themselves -----------------------------------------------------

@pytest.mark.parametrize("issue", sorted(SPECS))
def test_every_spec_loads_and_validates(issue):
    spec = load_spec(SPECS[issue])
    assert spec.id == issue
    assert spec.hypothesis.strip(), "a spec with no hypothesis pre-registers nothing"
    assert spec.criteria, "a spec needs at least one criterion"


@pytest.mark.parametrize(
    "issue,expected",
    [("86", KEEP), ("77", INCONCLUSIVE), ("79", KILL)],
)
def test_the_referee_reproduces_the_recorded_verdict(issue, expected):
    """The load-bearing test of #40 build 3: the module must reach the same
    conclusion the humans did, from the same numbers, via the criteria written
    down before the runs."""
    path = SPECS[issue]
    spec = load_spec(path)
    results = load_recorded_results(path)
    verdict = evaluate(spec, results)
    assert verdict.outcome == expected, f"#{issue}: {verdict.describe()}"


@pytest.mark.parametrize("issue", sorted(SPECS))
def test_the_spec_agrees_with_its_own_recorded_verdict_field(issue):
    """Each spec states the verdict its finding reached. Recompute and compare —
    this is what catches a spec transcribed wrongly from its finding."""
    path = SPECS[issue]
    spec = load_spec(path)
    assert spec.recorded, "a retrofit spec must state the verdict its finding reached"
    verdict = evaluate(spec, load_recorded_results(path))
    assert verdict.outcome == spec.recorded, verdict.describe()


# --- the logic itself ---------------------------------------------------------

def test_kill_wins_over_keep():
    """A spec that satisfies both branches is a badly-drafted spec; the safe
    reading of 'the kill condition fired' is that the idea is dead."""
    from instruments.verdict import Criterion, Spec

    spec = Spec(
        id="x", title="t", hypothesis="h",
        criteria={
            "k": Criterion("k", "within", "a", "b", 2.0),
            "d": Criterion("d", "beats", "a", "b", 0.0),  # trivially true
        },
        keep_if=("k",), kill_if=("d",),
    )
    v = evaluate(spec, {"p": {"a": [1.0, 1.0], "b": [1.0, 1.0]}})
    assert v.outcome == KILL


def test_require_any_versus_all():
    from instruments.verdict import Criterion, Spec

    results = {
        "p1": {"t": [1.0, 1.0], "c": [1.0, 1.0]},   # parity
        "p2": {"t": [5.0, 5.1], "c": [1.0, 1.0]},   # big win
    }
    for require, expected in (("all", INCONCLUSIVE), ("any", KEEP)):
        spec = Spec(
            id="x", title="t", hypothesis="h",
            criteria={"c1": Criterion("c1", "within", "t", "c", 2.0,
                                      points=("p1", "p2"), require=require)},
            keep_if=("c1",), kill_if=(),
        )
        assert evaluate(spec, results).outcome == expected


def test_a_spec_that_pre_registers_nothing_is_refused():
    from instruments.verdict import Criterion, Spec

    with pytest.raises(ValueError, match="pre-registers nothing"):
        Spec(id="x", title="t", hypothesis="h",
             criteria={"c": Criterion("c", "within", "a", "b", 2.0)},
             keep_if=(), kill_if=())


def test_criteria_must_name_real_arms_and_points():
    from instruments.verdict import Criterion, Spec

    spec = Spec(id="x", title="t", hypothesis="h",
                criteria={"c": Criterion("c", "within", "missing", "b", 2.0)},
                keep_if=("c",), kill_if=())
    with pytest.raises(KeyError, match="missing"):
        evaluate(spec, {"p": {"a": [1.0], "b": [1.0]}})


def test_unknown_rule_is_refused_at_construction():
    from instruments.verdict import Criterion

    with pytest.raises(ValueError, match="unknown rule"):
        Criterion("c", "exceeds", "a", "b", 2.0)
