"""The runner's plumbing, checked against a harness that costs milliseconds.

`instruments/experiment.py` launches real training runs, so testing it against a
real harness would cost minutes per assertion. Instead these tests point it at a
stub that emits the RESULT protocol instantly. That is not a shortcut around the
hard part — the hard part *is* the plumbing: which command each arm gets, what
resume skips, how legs keep two measurements of "d8" from overwriting each
other. The one thing a stub cannot prove — that the protocol works against a
genuine JAX harness — is checked in tests/expensive/test_runner_end_to_end.py.
"""

import json
import pathlib
import sys
import textwrap

import pytest

from instruments import experiment
from instruments.results import PREFIX, emit, parse
from instruments.verdict import KILL, load_spec

# A harness in eight lines: echo a number that depends on the arm flag, so the
# tests can pre-register a verdict and watch the runner reach it.
STUB_HARNESS = textwrap.dedent("""
    import argparse, sys
    sys.path.insert(0, "REPO_ROOT")
    from instruments.results import emit
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--good", action="store_true")
    ap.add_argument("--depths", default="8")
    args = ap.parse_args()
    for d in args.depths.split(","):
        base = 0.90 if args.good else 0.30
        emit(f"d{d}", acc=base + 0.01 * args.seed, ce=1.0)
""")

REPO_ROOT = pathlib.Path(experiment.__file__).resolve().parents[1]


def write_stub(tmp_path):
    path = tmp_path / "stub_harness.py"
    path.write_text(STUB_HARNESS.replace("REPO_ROOT", str(REPO_ROOT)))
    return path


def write_spec(tmp_path, execution_block, *, criteria=None):
    spec = tmp_path / "spec.toml"
    spec.write_text(textwrap.dedent(f"""
        [experiment]
        id = "stub"
        title = "a stub experiment"
        hypothesis = "the good flag helps"

        [protocol]
        metric_key = "acc"

        {execution_block}

        [arms.control]
        role = "control"
        flags = []

        [arms.treated]
        role = "treatment"
        flags = ["--good"]

        {criteria or '''
        [criteria.treated_wins]
        rule = "beats"
        treatment = "treated"
        control = "control"
        sigmas = 2.0
        '''}

        [verdict]
        keep_if = ["treated_wins"]
        kill_if = []
    """))
    return spec


def single_leg(stub):
    return textwrap.dedent(f"""
        [execution]
        command = ["{sys.executable}", "{stub}"]
        seeds = [0, 1, 2]
        seed_flag = "--seed"
    """)


# --- the RESULT protocol ------------------------------------------------------

def test_emit_and_parse_round_trip(capsys):
    emit("d8", acc=0.5, ce=1.25)
    rows = parse(capsys.readouterr().out)
    assert rows == [{"point": "d8", "acc": 0.5, "ce": 1.25}]


def test_non_result_output_is_ignored():
    assert parse("a human table\n" + PREFIX + '{"point": "x", "acc": 1.0}\nmore prose\n') == [
        {"point": "x", "acc": 1.0}]


def test_a_malformed_result_line_is_an_error_not_a_skip():
    """Dropping it silently would turn a broken harness into a sweep that
    measured fewer seeds than it claims."""
    with pytest.raises(ValueError, match="malformed"):
        parse(PREFIX + "{not json}")
    with pytest.raises(ValueError, match="no 'point'"):
        parse(PREFIX + '{"acc": 1.0}')


# --- building command lines ---------------------------------------------------

def test_each_arm_gets_the_base_command_plus_its_own_flags(tmp_path):
    stub = write_stub(tmp_path)
    ex = experiment.load_execution(load_spec(write_spec(tmp_path, single_leg(stub))))
    leg = ex.legs[0]
    assert ex.argv(leg, "control", 1) == (sys.executable, str(stub), "--seed", "1")
    assert ex.argv(leg, "treated", 1) == (sys.executable, str(stub), "--good", "--seed", "1")


def test_the_fingerprint_changes_with_the_command(tmp_path):
    """Resume reuses a journal row only if it came from the same command — a
    sweep whose flags changed underneath it must re-run, not resume."""
    stub = write_stub(tmp_path)
    ex = experiment.load_execution(load_spec(write_spec(tmp_path, single_leg(stub))))
    leg = ex.legs[0]
    assert ex.fingerprint(leg, "control", 0) != ex.fingerprint(leg, "treated", 0)
    assert ex.fingerprint(leg, "control", 0) != ex.fingerprint(leg, "control", 1)
    assert ex.fingerprint(leg, "control", 0) == ex.fingerprint(leg, "control", 0)


def test_a_spec_with_no_execution_section_is_judge_only(tmp_path):
    spec_path = tmp_path / "s.toml"
    spec_path.write_text(textwrap.dedent("""
        [experiment]
        id = "x"
        title = "t"
        hypothesis = "h"
        [criteria.c]
        rule = "within"
        treatment = "a"
        control = "b"
        sigmas = 2.0
        [arms.a]
        [arms.b]
        [verdict]
        keep_if = ["c"]
    """))
    with pytest.raises(ValueError, match="can be judged but not run"):
        experiment.load_execution(load_spec(spec_path))


def test_criteria_naming_an_undefined_arm_are_refused_before_the_sweep(tmp_path):
    """Two hours into a sweep is the wrong time to discover the spec compares an
    arm nobody configured."""
    stub = write_stub(tmp_path)
    bad = """
        [criteria.c]
        rule = "beats"
        treatment = "ghost"
        control = "control"
        sigmas = 2.0
    """
    spec = write_spec(tmp_path, single_leg(stub), criteria=bad)
    spec.write_text(spec.read_text().replace('keep_if = ["treated_wins"]', 'keep_if = ["c"]'))
    with pytest.raises(ValueError, match="ghost"):
        experiment.load_execution(load_spec(spec))


def test_a_spec_without_metric_key_is_refused(tmp_path):
    stub = write_stub(tmp_path)
    spec = write_spec(tmp_path, single_leg(stub))
    spec.write_text(spec.read_text().replace('metric_key = "acc"', ""))
    with pytest.raises(ValueError, match="metric_key"):
        experiment.load_execution(load_spec(spec))


# --- legs ---------------------------------------------------------------------

MULTI_LEG = """
[execution]
command = ["{python}", "{stub}"]
seed_flag = "--seed"
seeds = [0, 1]

[execution.legs.trained]
flags = ["--depths", "8"]

[execution.legs.extended]
flags = ["--depths", "8"]
seeds = [0, 1, 2]
arms = ["treated"]
"""


def test_legs_carry_their_own_seeds_and_arms(tmp_path):
    """#77 extended one arm from three seeds to six under a fresh
    pre-registration; a runner that cannot express that cannot run #77."""
    stub = write_stub(tmp_path)
    ex = experiment.load_execution(load_spec(write_spec(
        tmp_path, MULTI_LEG.format(python=sys.executable, stub=stub))))
    planned = [(leg.name, arm, seed) for leg, arm, seed in ex.runs()]
    assert ("trained", "control", 0) in planned
    assert ("extended", "control", 0) not in planned, "leg restricted itself to one arm"
    assert ("extended", "treated", 2) in planned, "leg's own seed list wins"
    assert len(planned) == 4 + 3


def test_leg_names_namespace_the_points(tmp_path):
    """#86's two legs both measure depth 8, under different eval conditions.
    Without namespacing, one silently overwrites the other."""
    stub = write_stub(tmp_path)
    ex = experiment.load_execution(load_spec(write_spec(
        tmp_path, MULTI_LEG.format(python=sys.executable, stub=stub))))
    trained, extended = ex.legs
    assert ex.point(trained, "d8") == "trained/d8"
    assert ex.point(extended, "d8") == "extended/d8"


def test_an_unnamed_leg_leaves_point_names_alone(tmp_path):
    stub = write_stub(tmp_path)
    ex = experiment.load_execution(load_spec(write_spec(tmp_path, single_leg(stub))))
    assert ex.point(ex.legs[0], "d8") == "d8"


def test_a_leg_naming_an_undefined_arm_is_refused(tmp_path):
    stub = write_stub(tmp_path)
    block = single_leg(stub) + '\n[execution.legs.only]\narms = ["nonexistent"]\n'
    with pytest.raises(ValueError, match="nonexistent"):
        experiment.load_execution(load_spec(write_spec(tmp_path, block)))


# --- collecting ---------------------------------------------------------------

def test_collect_groups_by_point_and_orders_by_seed():
    rows = [
        {"arm": "a", "seed": 2, "point": "d8", "acc": 0.3},
        {"arm": "a", "seed": 0, "point": "d8", "acc": 0.1},
        {"arm": "a", "seed": 1, "point": "d8", "acc": 0.2},
    ]
    assert experiment.collect(rows, "acc") == {"d8": {"a": [0.1, 0.2, 0.3]}}


def test_collect_names_the_missing_metric(tmp_path):
    rows = [{"arm": "a", "seed": 0, "point": "d8", "ce": 1.0}]
    with pytest.raises(KeyError, match="acc"):
        experiment.collect(rows, "acc")


def test_leg_namespaced_points_survive_a_toml_round_trip(tmp_path):
    """A '/' is not a bare TOML key; an unquoted one would make the spec
    unparseable after its own runner wrote it."""
    results = {"extended/d8": {"treated": [0.9, 0.91]}}
    path = tmp_path / "s.toml"
    path.write_text('[experiment]\nid="x"\ntitle="t"\nhypothesis="h"\n'
                    '[criteria.c]\nrule="beats"\ntreatment="treated"\ncontrol="treated"\nsigmas=0.0\n'
                    '[verdict]\nkeep_if=["c"]\n')
    experiment.record_results(path, results, "acc")
    from instruments.verdict import load_recorded_results
    assert load_recorded_results(path) == {"extended/d8": {"treated": [0.9, 0.91]}}


# --- the sweep, end to end against the stub -----------------------------------

def test_the_runner_sweeps_records_and_reaches_the_pre_registered_verdict(tmp_path, monkeypatch):
    stub = write_stub(tmp_path)
    spec_path = write_spec(tmp_path, single_leg(stub))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")

    assert experiment.main([str(spec_path), "--no-gate"]) == 0

    # The verdict landed in the spec file, parseable and correct.
    from instruments.verdict import evaluate, load_recorded_results
    spec = load_spec(spec_path)
    results = load_recorded_results(spec_path)
    assert results["d8"]["control"] == [0.30, 0.31, 0.32]
    assert results["d8"]["treated"] == [0.90, 0.91, 0.92]
    assert evaluate(spec, results).outcome == "KEEP"

    draft = tmp_path / "runs" / "stub" / "finding-draft.md"
    assert "Verdict: KEEP" in draft.read_text()
    assert "_To be written by a human._" in draft.read_text(), "the prose stays a human's job"


def test_recorded_results_are_not_silently_overwritten(tmp_path, monkeypatch):
    stub = write_stub(tmp_path)
    spec_path = write_spec(tmp_path, single_leg(stub))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    experiment.main([str(spec_path), "--no-gate"])

    before = spec_path.read_text()
    assert experiment.record_results(spec_path, {"d8": {"control": [0.0]}}, "acc") is False
    assert spec_path.read_text() == before, "a recorded number is not the runner's to rewrite"
    assert experiment.record_results(spec_path, {"d8": {"control": [0.0]}}, "acc", force=True)


def test_resume_skips_what_the_journal_already_has(tmp_path, monkeypatch):
    """A sweep that dies at seed 4 of 6 must not restart from zero."""
    stub = write_stub(tmp_path)
    spec_path = write_spec(tmp_path, single_leg(stub))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")

    spec = load_spec(spec_path)
    ex = experiment.load_execution(spec)
    experiment.sweep(spec, ex)

    launched = []
    monkeypatch.setattr(experiment, "run_one",
                        lambda *a, **k: launched.append(a) or [])
    rows = experiment.sweep(spec, ex)
    assert launched == [], "everything was already in the journal"
    assert len(rows) == 6

    experiment.sweep(spec, ex, resume=False)
    assert len(launched) == 6, "--no-resume re-runs the lot"


def test_a_harness_that_emits_nothing_is_an_error(tmp_path, monkeypatch):
    """Silence is not a result. Without this the sweep would 'succeed' with an
    empty results table and the referee would judge nothing at all."""
    silent = tmp_path / "silent.py"
    silent.write_text("import sys; print('trained, honest')\n")
    spec_path = write_spec(tmp_path, textwrap.dedent(f"""
        [execution]
        command = ["{sys.executable}", "{silent}"]
        seeds = [0]
        seed_flag = "--seed"
    """))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    spec = load_spec(spec_path)
    with pytest.raises(RuntimeError, match="no RESULT line"):
        experiment.sweep(spec, experiment.load_execution(spec))


def test_a_failing_run_stops_the_sweep(tmp_path, monkeypatch):
    crash = tmp_path / "crash.py"
    crash.write_text("import sys; sys.exit(3)\n")
    spec_path = write_spec(tmp_path, textwrap.dedent(f"""
        [execution]
        command = ["{sys.executable}", "{crash}"]
        seeds = [0]
        seed_flag = "--seed"
    """))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    spec = load_spec(spec_path)
    with pytest.raises(RuntimeError, match="exited 3"):
        experiment.sweep(spec, experiment.load_execution(spec))


def test_dry_run_prints_the_plan_and_launches_nothing(tmp_path, monkeypatch, capsys):
    stub = write_stub(tmp_path)
    spec_path = write_spec(tmp_path, single_leg(stub))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(experiment, "run_one", lambda *a, **k: pytest.fail("launched"))

    assert experiment.main([str(spec_path), "--dry-run"]) == 0
    assert capsys.readouterr().out.count("--seed") == 6


def test_the_journal_is_json_per_line(tmp_path, monkeypatch):
    """The journal is the crash-recovery record and the audit trail; it has to
    stay readable by something other than this module."""
    stub = write_stub(tmp_path)
    spec_path = write_spec(tmp_path, single_leg(stub))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    spec = load_spec(spec_path)
    experiment.sweep(spec, experiment.load_execution(spec))

    rows = [json.loads(line) for line in
            (tmp_path / "runs" / "stub" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {"arm", "seed", "point", "acc", "fingerprint", "elapsed_s"} <= set(rows[0])


def test_a_kill_verdict_is_reported_as_such(tmp_path, monkeypatch):
    """The runner must be as willing to report a kill as a keep — the whole
    point of handing the verdict to code."""
    stub = write_stub(tmp_path)
    kill_criteria = """
        [criteria.treated_wins]
        rule = "beats"
        treatment = "treated"
        control = "control"
        sigmas = 2.0

        [criteria.treated_loses]
        rule = "loses"
        treatment = "control"
        control = "treated"
        sigmas = 2.0
    """
    spec_path = write_spec(tmp_path, single_leg(stub), criteria=kill_criteria)
    spec_path.write_text(spec_path.read_text().replace(
        'kill_if = []', 'kill_if = ["treated_loses"]'))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")
    experiment.main([str(spec_path), "--no-gate"])

    from instruments.verdict import evaluate, load_recorded_results
    spec = load_spec(spec_path)
    verdict = evaluate(spec, load_recorded_results(spec_path))
    assert verdict.outcome == KILL
