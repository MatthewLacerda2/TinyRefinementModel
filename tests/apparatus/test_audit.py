"""The audit's rules, each against a spec built to trip exactly it (#304).

Every fixture is a small spec in a tmp repo layout — experiments/<line>/specs/,
docs/findings/, a ROADMAP with a graveyard — so a rule is proven on the shape of
failure it names rather than on whatever the real specs happen to contain today.
The one rule that reads git (criteria-predate-results) gets a throwaway repository.
"""

import os
import pathlib
import subprocess
import textwrap

import pytest

from instruments import audit
from instruments.audit import GRANDFATHERED, GREEN, NA, RED

SPEC = """\
[experiment]
id = "{id}"
status = "{status}"
title = "a toy"
hypothesis = "{hypothesis}"
issue = 999

[protocol]
metric_key = "{metric_key}"
{protocol_extra}

[execution]
command = ["python", "-m", "x"]
seeds = [0, 1, 2]

[arms.control]
role = "control"
flags = []

[arms.treated]
role = "treatment"
flags = ["--x"]
{arms_extra}

[criteria.parity]
rule = "within"
treatment = "treated"
control = "control"
sigmas = 2.0
points = ["p"]

[criteria.loses]
rule = "loses"
treatment = "treated"
control = "control"
sigmas = 2.0
points = ["p"]

[verdict]
keep_if = ["parity"]
kill_if = ["loses"]
{recorded}

{extra}
"""

RESULTS = """
[results.p]
control = {control}
treated = {treated}
{results_extra}
"""


def spec_text(*, control=(0.90, 0.91, 0.92), treated=(0.92, 0.93, 0.91), recorded=None,
              hypothesis="Prediction registered: parity.", status="open", metric_key="acc",
              protocol_extra="", arms_extra="", results_extra="", extra="", results=True, id="001-toy"):
    text = SPEC.format(id=id, status=status, hypothesis=hypothesis, metric_key=metric_key,
                       protocol_extra=protocol_extra, arms_extra=arms_extra, extra=extra,
                       recorded=f'recorded = "{recorded}"' if recorded else "")
    if results:
        text += RESULTS.format(control=list(control), treated=list(treated), results_extra=results_extra)
    return text


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "experiments" / "toy" / "specs").mkdir(parents=True)
    (tmp_path / "docs" / "findings").mkdir(parents=True)
    (tmp_path / "docs" / "ROADMAP.md").write_text("# Roadmap\n\n## Graveyard\n- nothing yet\n")
    return tmp_path


def write_spec(repo, text, name="001-toy.toml"):
    path = repo / "experiments" / "toy" / "specs" / name
    path.write_text(text)
    return path


def load(repo, text, name="001-toy.toml"):
    return audit.load(write_spec(repo, text, name), repo)


def states(repo, a):
    return {s.rule: s for s in audit.audit_spec(a, audit.all_citations(repo), audit.graveyard_text(repo))}


# --- a sound spec ------------------------------------------------------------------

def test_a_sound_spec_goes_red_nowhere(repo):
    (repo / "docs" / "findings" / "2026-09-14-toy.md").write_text(
        "# Toy\n\nSpec: experiments/toy/specs/001-toy.toml\n")
    a = load(repo, spec_text(recorded="KEEP"))
    result = states(repo, a)
    assert not [s for s in result.values() if s.state == RED], result
    assert result["verdict-current"].state == GREEN
    assert result["finding-cites-spec"].state == GREEN
    assert result["criteria-predate-results"].state == NA  # tmp dir, not a repository
    assert result["run-ended-by-budget"].state == NA


def test_a_spec_with_no_measured_results_is_pending_not_red(repo):
    write_spec(repo, spec_text(results=False))
    blocks, red = audit.run(["experiments/toy/specs/001-toy.toml"], [], repo)
    assert not red
    assert "[pending]" in blocks[0]


# --- floor-both-arms -----------------------------------------------------------------

FLOOR_ARM = """
[arms.chance]
role = "floor"
constant = true
"""


def test_parity_keep_with_both_arms_at_the_floor_is_red(repo):
    a = load(repo, spec_text(control=(0.200, 0.201, 0.199), treated=(0.201, 0.200, 0.202),
                             arms_extra=FLOOR_ARM, results_extra="chance = { mean = 0.2, sigma = 0.001, n = 3 }"))
    assert a.verdict.outcome == "KEEP"  # the referee is fooled; the audit is the guard
    s = audit.floor_both_arms(a)
    assert s.state == RED and "both arms at the floor" in s.reason


def test_parity_keep_well_above_the_floor_is_green(repo):
    a = load(repo, spec_text(arms_extra=FLOOR_ARM, results_extra="chance = { mean = 0.2, sigma = 0.001, n = 3 }"))
    assert audit.floor_both_arms(a).state == GREEN


def test_no_floor_declared_is_not_applicable(repo):
    assert audit.floor_both_arms(load(repo, spec_text())).state == NA


# --- seeds-complete -------------------------------------------------------------------

def test_a_missing_seed_is_red_until_marked_void(repo):
    short = load(repo, spec_text(control=(0.90, 0.91)))
    s = audit.seeds_complete(short)
    assert s.state == RED and "control at p: 2 of 3" in s.reason

    voided = load(repo, spec_text(control=(0.90, 0.91), extra=textwrap.dedent("""
        [void.p.control]
        seeds = [2]
        reason = "seed 2 OOMed at step 40; re-registered as 002"
    """)))
    assert audit.seeds_complete(voided).state == GREEN


def test_a_void_without_a_reason_does_not_count(repo):
    a = load(repo, spec_text(control=(0.90, 0.91), extra="[void.p.control]\nseeds = [2]\nreason = \"\"\n"))
    assert audit.seeds_complete(a).state == RED


# --- sigma-plausible ---------------------------------------------------------------------

def test_identical_seeds_give_zero_sigma_and_go_red(repo):
    a = load(repo, spec_text(control=(0.9, 0.9, 0.9), treated=(0.9, 0.9, 0.9)))
    s = audit.sigma_plausible(a)
    assert s.state == RED and "sigma_pooled is 0" in s.reason


def test_noise_ten_times_the_gap_on_a_carrying_criterion_is_noted_not_red(repo):
    """A true null lands here one time in ten by chance; a red would fail honest
    parity results, so the audit says what the KEEP means instead."""
    a = load(repo, spec_text(control=(0.80, 0.90, 1.00), treated=(0.801, 0.901, 1.001)))
    assert a.verdict.outcome == "KEEP"
    s = audit.sigma_plausible(a)
    assert s.state == GREEN and "no difference the measurement could see" in s.reason
    assert audit.sigma_plausible(load(repo, spec_text())).reason == ""  # a resolved gap gets no note


MARGIN_SPEC = """\
[experiment]
id = "002-margin"
status = "recorded"
title = "a toy with an absolute margin"
hypothesis = "Prediction registered: the treatment wins."
issue = 999

[protocol]
cap = 67.108864
metric_key = "tokens_to_target_M"

[execution]
command = ["python", "-m", "x"]
seeds = [0, 1, 2]

[arms.control]
role = "control"
flags = []

[arms.treated]
role = "treatment"
flags = ["--x"]

[criteria.wins]
rule = "beats"
treatment = "control"
control = "treated"
sigmas = 2.0
{margin}
points = ["p"]

[criteria.mirror]
rule = "loses"
treatment = "control"
control = "treated"
sigmas = 2.0
points = ["p"]

[verdict]
keep_if = ["wins"]
kill_if = ["mirror"]
recorded = "KEEP"

[results.p]
control = [44.5645, 44.5645, 44.5645]
treated = [20.9715, 20.9715, 20.9715]
"""


def test_zero_sigma_is_noted_not_red_when_a_registered_margin_carried_the_verdict(repo):
    """A quantized metric (tokens to target resolves no finer than the probe interval)
    repeats to the last digit, so sigma is 0 — but `require = all` made the gap clear
    min_delta too, so the verdict never rested on the undefined sigma."""
    a = load(repo, MARGIN_SPEC.format(margin="min_delta = 6.7"))
    assert a.verdict.outcome == "KEEP"
    s = audit.sigma_plausible(a)
    assert s.state == GREEN and "rests on the registered margin" in s.reason


def test_zero_sigma_with_no_margin_on_the_carrying_criterion_is_still_red(repo):
    a = load(repo, MARGIN_SPEC.format(margin=""))
    s = audit.sigma_plausible(a)
    assert s.state == RED and "sigma_pooled is 0" in s.reason


def test_zero_sigma_with_a_margin_the_gap_does_not_clear_is_still_red(repo):
    a = load(repo, MARGIN_SPEC.format(margin="min_delta = 90.0"))
    s = audit.sigma_plausible(a)
    assert s.state == RED and "sigma_pooled is 0" in s.reason


def test_zero_sigma_on_a_criterion_that_decided_nothing_is_noted(repo):
    """The unfired kill bar of a KEEP: an undefined sigma there did not corrupt this
    verdict, but it would have fired on any gap at all, and the audit says so."""
    s = audit.sigma_plausible(load(repo, MARGIN_SPEC.format(margin="min_delta = 6.7")))
    assert s.state == GREEN and "would have fired on any gap at all" in s.reason


# --- control-reached-target ------------------------------------------------------------------

def test_a_control_seed_at_the_cap_is_red_until_voided(repo):
    tokens = dict(metric_key="tokens_to_target_M", control=(44.6, 67.1, 45.0), treated=(21.0, 22.0, 21.5))
    no_cap = load(repo, spec_text(**tokens))
    assert audit.control_reached_target(no_cap).state == RED  # the cap must be declared to be checked

    capped = load(repo, spec_text(protocol_extra="cap = 67.1", **tokens))
    s = audit.control_reached_target(capped)
    assert s.state == RED and "never reached the target" in s.reason

    voided = load(repo, spec_text(protocol_extra="cap = 67.1", extra=textwrap.dedent("""
        [void.p.control]
        seeds = [1]
        reason = "never reached 5.85 inside the cap; re-registered"
    """), **tokens))
    assert audit.control_reached_target(voided).state == GREEN


def test_an_accuracy_metric_has_no_target_to_reach(repo):
    assert audit.control_reached_target(load(repo, spec_text())).state == NA


# --- prediction-registered / verdict-current ---------------------------------------------

def test_a_hypothesis_without_a_prediction_is_red_outside_git(repo):
    a = load(repo, spec_text(hypothesis="post-norm bounds the branch output."))
    assert audit.prediction_registered(a).state == RED


def test_a_stored_verdict_the_referee_does_not_reproduce_is_red(repo):
    a = load(repo, spec_text(recorded="KILL"))  # the numbers read KEEP
    s = audit.verdict_current(a)
    assert s.state == RED and "records KILL" in s.reason and "reads KEEP" in s.reason


def test_no_stored_verdict_is_not_applicable_and_names_what_the_referee_reads(repo):
    s = audit.verdict_current(load(repo, spec_text()))
    assert s.state == NA and "KEEP" in s.reason


# --- finding-cites-spec, both directions -------------------------------------------------

def test_a_keep_nobody_wrote_down_is_red(repo):
    result = states(repo, load(repo, spec_text(recorded="KEEP")))
    assert result["finding-cites-spec"].state == RED


def test_a_graveyard_tombstone_naming_the_issue_counts(repo):
    (repo / "docs" / "ROADMAP.md").write_text("## Graveyard\n- **toy** — KILLED (#999).\n")
    result = states(repo, load(repo, spec_text(recorded="KILL", control=(0.9, 0.91, 0.92), treated=(0.5, 0.51, 0.52))))
    assert result["finding-cites-spec"].state == GREEN


def test_an_inconclusive_spec_has_nothing_to_cite(repo):
    a = load(repo, spec_text(recorded="INCONCLUSIVE", control=(0.90, 0.91, 0.92), treated=(0.95, 0.96, 0.97)))
    assert audit.finding_cites_spec(a, {}, "").state == NA


def test_a_finding_after_the_cutoff_names_its_spec_or_is_observational(repo):
    findings = repo / "docs" / "findings"
    write_spec(repo, spec_text())
    bare = findings / "2026-09-13-bare.md"
    bare.write_text("# A claim\n\nDate: 2026-09-13\n")
    assert audit.audit_finding(bare, repo).state == RED

    cites = findings / "2026-09-13-cites.md"
    cites.write_text("# A claim\n\nSpec: experiments/toy/specs/001-toy.toml\n")
    assert audit.audit_finding(cites, repo).state == GREEN

    dangling = findings / "2026-09-13-dangling.md"
    dangling.write_text("Spec: experiments/toy/specs/404.toml\n")
    assert audit.audit_finding(dangling, repo).state == RED

    observational = findings / "2026-09-13-obs.md"
    observational.write_text("Evidence: observational — found while running something else\n")
    assert audit.audit_finding(observational, repo).state == GREEN

    old = findings / "2026-09-11-old.md"
    old.write_text("# predates the format\n")
    assert audit.audit_finding(old, repo).state == GRANDFATHERED


def test_a_spec_line_may_continue_over_indented_lines():
    text = textwrap.dedent("""\
        Spec: experiments/depth/specs/242-a.toml,
              experiments/depth/specs/238-b.toml,
              experiments/depth/specs/246-c.toml
        Measured with: `instruments/paired.py`
    """)
    assert audit.spec_citations(text) == ["experiments/depth/specs/242-a.toml",
                                          "experiments/depth/specs/238-b.toml",
                                          "experiments/depth/specs/246-c.toml"]


# --- run-ended-by-budget --------------------------------------------------------------------

def test_a_run_log_must_end_at_a_budget_stop(repo):
    (repo / "runs" / "run_a").mkdir(parents=True)
    (repo / "runs" / "run_a" / "train.log").write_text("step 1\nstep 2\n2026-09-15 BUDGET_COMPLETE: budget reached\n")
    (repo / "runs" / "run_b").mkdir()
    (repo / "runs" / "run_b" / "train.log").write_text("step 1\n2026-09-15 KILLED_DIVERGENCE: CE non-finite\n")

    good = load(repo, spec_text(extra='[runs.control]\nlogs = ["runs/run_a/train.log"]\n'))
    assert audit.run_ended_by_budget(good).state == GREEN

    bad = load(repo, spec_text(extra='[runs.treated]\ndirs = ["runs/run_b"]\n'))
    s = audit.run_ended_by_budget(bad)
    assert s.state == RED and "KILLED_DIVERGENCE" in s.reason

    absent = load(repo, spec_text(extra='[runs.control]\nlogs = ["runs/elsewhere/train.log"]\n'))
    assert audit.run_ended_by_budget(absent).state == NA
    assert audit.run_ended_by_budget(load(repo, spec_text())).state == NA


# --- criteria-predate-results, against a real git history ----------------------------------

def git(repo, *args, date="2026-10-01T12:00:00"):
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def git_repo(repo):
    git(repo, "init", "-q")
    return repo


def test_criteria_committed_before_results_is_green(git_repo):
    path = write_spec(git_repo, spec_text(results=False))
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "pre-register")
    path.write_text(spec_text())
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "results")
    s = audit.criteria_predate_results(audit.load(path, git_repo))
    assert s.state == GREEN and "precede results" in s.reason


def test_criteria_and_results_in_one_commit_after_the_cutoff_is_red(git_repo):
    path = write_spec(git_repo, spec_text())
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "everything at once")
    s = audit.criteria_predate_results(audit.load(path, git_repo))
    assert s.state == RED and "landed together" in s.reason


def test_one_commit_before_the_cutoff_or_a_retrofit_is_grandfathered_with_the_date(git_repo):
    retro = write_spec(git_repo, spec_text(status="retrofit"), "002-retro.toml")
    old = write_spec(git_repo, spec_text(), "003-old.toml")
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "squashed", date="2026-09-12T12:00:00")
    s = audit.criteria_predate_results(audit.load(retro, git_repo))
    assert s.state == GRANDFATHERED and "retrofit" in s.reason
    s = audit.criteria_predate_results(audit.load(old, git_repo))
    assert s.state == GRANDFATHERED and "2026-09-12" in s.reason and "squash" in s.reason


def test_results_committed_before_criteria_is_red(git_repo):
    path = git_repo / "experiments" / "toy" / "specs" / "004-late.toml"
    path.write_text(textwrap.dedent("""\
        [experiment]
        id = "004"
        title = "t"
        hypothesis = "h"
        [protocol]
        metric_key = "acc"
        [results.p]
        control = [0.9, 0.91, 0.92]
        treated = [0.91, 0.92, 0.90]
    """))
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "numbers first")
    path.write_text(spec_text())
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "criteria after")
    s = audit.criteria_predate_results(audit.load(path, git_repo))
    assert s.state == RED and "before criteria" in s.reason


def test_an_uncommitted_spec_is_red_and_predates_nothing(git_repo):
    path = write_spec(git_repo, spec_text())
    git(git_repo, "add", "docs"), git(git_repo, "commit", "-qm", "unrelated")
    assert audit.criteria_predate_results(audit.load(path, git_repo)).state == RED


def test_a_spec_committed_before_the_doctrine_is_grandfathered_on_prediction(git_repo):
    path = write_spec(git_repo, spec_text(hypothesis="the mechanism helps; nothing registered about how much"))
    git(git_repo, "add", "."), git(git_repo, "commit", "-qm", "old", date="2026-09-01T12:00:00")
    s = audit.prediction_registered(audit.load(path, git_repo))
    assert s.state == GRANDFATHERED and "2026-09-01" in s.reason


# --- scope: the diff -> spec map ----------------------------------------------------------

def test_the_scope_map(repo):
    specs = repo / "experiments" / "toy" / "specs"
    for name in ("001-toy.toml", "002-toy.toml"):
        write_spec(repo, spec_text(), name)
    (repo / "experiments" / "toy" / "harness.py").write_text("")
    (repo / "docs" / "findings" / "2026-09-14-x.md").write_text("Spec: experiments/toy/specs/002-toy.toml\n")
    (repo / "experiments" / "other" / "specs").mkdir(parents=True)

    assert audit.affected(["experiments/toy/specs/001-toy.toml"], repo).specs == {"experiments/toy/specs/001-toy.toml"}
    assert audit.affected(["experiments/toy/harness.py"], repo).specs == {
        "experiments/toy/specs/001-toy.toml", "experiments/toy/specs/002-toy.toml"}
    scope = audit.affected(["docs/findings/2026-09-14-x.md"], repo)
    assert scope.specs == {"experiments/toy/specs/002-toy.toml"}
    assert scope.findings == {"docs/findings/2026-09-14-x.md"}
    assert audit.affected(["instruments/verdict.py"], repo).everything
    assert audit.affected(["experiments/other/harness.py"], repo).specs == set()  # a line with no specs
    nothing = audit.affected(["trm/model/plain.py", "README.md", "experiments/toy/specs/gone.toml"], repo)
    assert not nothing.specs and not nothing.findings and not nothing.everything
    assert specs.exists()


def test_the_rule_names_are_the_ones_the_issue_registers(repo):
    result = states(repo, load(repo, spec_text()))
    assert list(result) == ["floor-both-arms", "control-reached-target", "seeds-complete", "sigma-plausible",
                            "prediction-registered", "criteria-predate-results", "verdict-current",
                            "finding-cites-spec", "run-ended-by-budget"]


def test_every_real_spec_loads_without_crashing():
    """The audit must open every spec the repo has. The verdicts themselves are CI's
    business (the audit job); this only proves nothing here crashes the loader.

    It used to be named for a "pending" check it never made (#325). Pending is pinned
    on a synthetic spec in test_a_spec_with_no_measured_results_is_pending_not_red;
    pinning it on a real one would break the day that experiment records results."""
    real = pathlib.Path(audit.REPO)
    rels = audit.all_specs(real)
    assert rels, "no specs found — has the layout moved?"
    for rel in rels:
        audit.load(real / rel, real)
