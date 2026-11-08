"""Two registered looks at a parity pair (#604): the bars match the published
tables, a look judges only its own seeds at its own bar, and spending keeps the
overall error at alpha.

Single-look specs are not here: tests/apparatus/test_verdict_pinned.py shows each
one is judged bit for bit as before.
"""

import random
import textwrap

import pytest

from instruments.spec_file import SpecFile, StagesTable
from instruments.verdict import (
    ALPHA,
    INCONCLUSIVE,
    KEEP,
    KILL,
    SPENDING,
    Criterion,
    Spec,
    Summary,
    evaluate,
    load_spec,
    look_bounds,
    looks,
)

# --- the bars, against published references ------------------------------------


@pytest.mark.parametrize("spending, expected", [
    # Lan-DeMets O'Brien-Fleming-type, two equally spaced looks, one-sided 0.025:
    # z 2.963 then 1.969 (Jennison & Turnbull 2000, ch. 7; gsDesign's sfLDOF).
    ("obrien-fleming", (2.963, 1.969)),
    # Lan-DeMets Pocock-type, same design: z 2.157 then 2.201 (gsDesign's sfLDPocock).
    ("pocock", (2.157, 2.201)),
])
def test_two_look_bounds_match_the_published_tables(spending, expected):
    assert look_bounds(spending, 0.5, 0.025) == pytest.approx(expected, abs=1e-3)


def test_obrien_fleming_spends_almost_nothing_at_half_the_information():
    """Lan & DeMets (1983): the O'Brien-Fleming-type function has spent 0.0056 of a
    one-sided 0.05 at t = 0.5, and all of it at t = 1."""
    spend = SPENDING["obrien-fleming"]
    assert spend(0.5, 0.05) == pytest.approx(0.00557, abs=1e-5)
    for name, spend in SPENDING.items():
        assert spend(1.0, ALPHA) == pytest.approx(ALPHA, abs=1e-12), name
        assert 0 < spend(0.25, ALPHA) < spend(0.5, ALPHA) < ALPHA, name


def test_the_spec_format_offers_exactly_the_spending_functions_the_referee_has():
    assert set(StagesTable.model_fields["spending"].annotation.__args__) == set(SPENDING)


def test_looks_turn_seed_counts_into_nominal_bars():
    first, final = looks(StagesTable(seeds=(4, 8)))
    assert (first.index, first.seeds, final.index, final.seeds) == (1, 4, 2, 8)
    assert first.alpha == pytest.approx(0.00557, abs=1e-5)
    assert first.alpha < final.alpha < ALPHA      # spending keeps the final bar near alpha


# --- the spec format -------------------------------------------------------------

STAGED = textwrap.dedent("""
    [experiment]
    id = "x"
    title = "t"
    hypothesis = "h; prediction: parity"
    [execution]
    command = ["python", "-m", "harness"]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    [stages]
    seeds = [4, 8]
    [criteria.same]
    rule = "equivalent"
    treatment = "t"
    control = "c"
    sigmas = 0.0
    margin = 0.01
    [verdict]
    keep_if = ["same"]
""")


def test_a_staged_spec_reads_with_its_defaults(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text(STAGED)
    spec = load_spec(path)
    assert spec.stages == StagesTable(seeds=(4, 8), spending="pocock", futility=True)
    assert spec.criteria["same"].margin == 0.01


@pytest.mark.parametrize("edit, message", [
    (("seeds = [4, 8]", "seeds = [4, 8]\nlooks = 2"), r"\[stages\.looks\] Extra inputs"),
    (("seeds = [4, 8]", "seeds = [4, 6, 8]"), r"\[stages\.seeds\]"),
    (("seeds = [4, 8]", "seeds = [2, 8]"), r"at least 3 seeds a side and grow"),
    (("seeds = [4, 8]", "seeds = [8, 8]"), r"at least 3 seeds a side and grow"),
    (("[stages]\n", '[stages]\nspending = "haybittle"\n'), r"\[stages\.spending\]"),
])
def test_a_malformed_stages_table_fails_at_the_boundary(tmp_path, edit, message):
    path = tmp_path / "s.toml"
    path.write_text(STAGED.replace(*edit))
    with pytest.raises(ValueError, match=message):
        SpecFile.load(path)


@pytest.mark.parametrize("edit, message", [
    (("margin = 0.01\n", ""), r"must register an absolute `margin`"),
    (('rule = "equivalent"', 'rule = "beats"'), r"margin is the parity margin"),
    (('sigmas = 0.0\nmargin', 'sigmas = 2.0\nmargin'), r"write sigmas = 0"),
    (("seeds = [0, 1, 2, 3, 4, 5, 6, 7]", "seeds = [0, 1, 2]"), r"\[execution\] registers 3"),
])
def test_a_staged_spec_that_breaks_the_design_is_refused(tmp_path, edit, message):
    path = tmp_path / "s.toml"
    path.write_text(STAGED.replace(*edit))
    with pytest.raises(ValueError, match=message):
        load_spec(path)


def test_two_looks_need_a_parity_keep():
    beats = {"win": Criterion("win", "beats", "t", "c", 2.0)}
    with pytest.raises(ValueError, match="spends alpha on equivalent only"):
        Spec(id="x", title="t", hypothesis="h", criteria=beats, keep_if=("win",), kill_if=(),
             stages=StagesTable(seeds=(3, 6)))


# --- judging at a look ----------------------------------------------------------

MARGIN = 1.0


def staged(**stages) -> Spec:
    criteria = {"same": Criterion("same", "equivalent", "t", "c", 0.0, margin=MARGIN),
                "above_floor": Criterion("above_floor", "beats", "c", "floor", 2.0),
                "worse": Criterion("worse", "loses", "t", "c", 3.0)}
    return Spec(id="x", title="t", hypothesis="h", criteria=criteria, keep_if=("same", "above_floor"),
                kill_if=("worse",), stages=StagesTable(seeds=(4, 8), **stages))


def arms(treatment, control):
    return {"run": {"t": treatment, "c": control, "floor": Summary(mean=-10.0, sigma=0.0, n=1)}}


TIGHT = [0.0, 0.01, -0.01, 0.005, 0.3, -0.3, 0.2, -0.2]   # first four nearly identical


def test_look_1_keeps_when_parity_clears_look_1s_bar():
    verdict = evaluate(staged(), arms(TIGHT, [v + 0.001 for v in TIGHT]), look=1)
    assert (verdict.outcome, verdict.settled, verdict.look.index) == (KEEP, True, 1)
    point = verdict.criteria[0].points[0]
    assert point.seeds == (4, 4) and point.p <= verdict.look.alpha
    assert "look 1 of 2: 4 seeds a side" in verdict.describe()


def test_look_1_continues_when_parity_misses_its_bar_but_could_still_be_shown():
    """p under ALPHA but over look 1's spent bar: the fixed design would have kept,
    the staged one runs on."""
    control = [0.0, 0.3, -0.3, 0.1, 0.0, 0.3, -0.3, 0.1]
    treatment = [v + 0.5 for v in control]
    verdict = evaluate(staged(), arms(treatment, control), look=1)
    point = verdict.criteria[0].points[0]
    assert verdict.look.alpha < point.p <= ALPHA
    assert (verdict.outcome, verdict.settled) == (INCONCLUSIVE, False)
    assert "run look 2's seeds" in verdict.reason
    final = evaluate(staged(), arms(treatment, control))
    assert (final.look.index, final.outcome, final.settled) == (2, KEEP, True)


def test_look_1_stops_a_pair_shown_beyond_its_margin():
    control = [0.0, 0.1, -0.1, 0.05, 0.0, 0.1, -0.1, 0.05]
    treatment = [v + 2.0 for v in control]                  # 2x the margin, tight seeds
    verdict = evaluate(staged(), arms(treatment, control), look=1)
    assert (verdict.outcome, verdict.settled) == (INCONCLUSIVE, True)
    assert verdict.reason.startswith("futile at look 1: same")
    without = evaluate(staged(futility=False), arms(treatment, control), look=1)
    assert (without.outcome, without.settled) == (INCONCLUSIVE, False)


def test_a_kill_branch_still_fires_at_look_1():
    control = [0.0, 0.1, -0.1, 0.05, 0.0, 0.1, -0.1, 0.05]
    treatment = [v - 2.0 for v in control]
    verdict = evaluate(staged(), arms(treatment, control), look=1)
    assert (verdict.outcome, verdict.settled) == (KILL, True)


def test_the_default_replays_the_design_look_1_first():
    """A look-1 stop stands even when look 2's seeds were run anyway."""
    assert evaluate(staged(), arms(TIGHT[:5], TIGHT[:5])).look.index == 1
    assert evaluate(staged(), arms(TIGHT, TIGHT)).look.index == 1          # settled at look 1
    loose = [0.0, 0.3, -0.3, 0.1, 0.0, 0.3, -0.3, 0.1]
    unsettled = arms([v + 0.5 for v in loose], loose)
    assert evaluate(staged(), {"run": {k: v[:5] if isinstance(v, list) else v
                                       for k, v in unsettled["run"].items()}}).settled is False
    assert evaluate(staged(), unsettled).look.index == 2
    with pytest.raises(ValueError, match="look 1 judges 4 seeds a side, and an arm has 3"):
        evaluate(staged(), arms(TIGHT[:3], TIGHT[:3]))
    with pytest.raises(ValueError, match="look 2 judges 8 seeds a side"):
        evaluate(staged(), arms(TIGHT[:5], TIGHT[:5]), look=2)


def test_a_single_look_spec_has_no_look_to_choose():
    spec = Spec(id="x", title="t", hypothesis="h",
                criteria={"win": Criterion("win", "beats", "t", "c", 2.0)}, keep_if=("win",), kill_if=())
    assert evaluate(spec, {"p": {"t": [3.0, 3.1, 2.9], "c": [0.0, 0.1, -0.1]}}).look is None
    with pytest.raises(ValueError, match="one look"):
        evaluate(spec, {"p": {"t": [3.0, 3.1, 2.9], "c": [0.0, 0.1, -0.1]}}, look=1)


# --- smoke 2 of #604: a synthetic null keeps its error at alpha -------------------

PAIRS = 8000


@pytest.mark.parametrize("spending", sorted(SPENDING))
def test_two_looks_keep_the_error_at_alpha_on_a_synthetic_null(spending):
    """Arms one margin apart (the edge of the null: parity is false) and sigma 1,
    at a margin of 2 sigma. Declaring parity at either look must stay at ALPHA
    within simulation error. Judging both looks at an unspent 0.05 reads ~0.079
    here (measured in review of #631), which this bound refuses."""
    spec = Spec(id="null", title="t", hypothesis="h",
                criteria={"same": Criterion("same", "equivalent", "t", "c", 0.0, margin=2.0)},
                keep_if=("same",), kill_if=(), stages=StagesTable(seeds=(4, 8), spending=spending))
    rng = random.Random(604)
    kept = 0
    for _ in range(PAIRS):
        results = {"p": {"t": [rng.gauss(2.0, 1.0) for _ in range(8)], "c": [rng.gauss(0.0, 1.0) for _ in range(8)]}}
        kept += evaluate(spec, results).outcome == KEEP
    error = kept / PAIRS
    se = (ALPHA * (1 - ALPHA) / PAIRS) ** 0.5
    assert error <= ALPHA + 3 * se, f"declared parity on {error:.4f} of null pairs"
