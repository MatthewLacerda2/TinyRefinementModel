"""The validity audit: can a recorded verdict be trusted? (#304)

    python -m instruments.audit                # the specs a change vs main can reach
    python -m instruments.audit --all          # every spec, every finding
    python -m instruments.audit --spec PATH    # one spec
    python -m instruments.audit --since SHA    # what a push's commit range touched (CI on main)

`instruments/verdict.py` decides KEEP / KILL / INCONCLUSIVE and nothing here
changes that. This asks the question that comes before it: was the experiment
that produced the verdict sound? CLAUDE.md's rules 1-5 are each a way a verdict
can be right by arithmetic and wrong as evidence — both arms at the floor (#246
build 1), a single seed, a criterion written after the number, a finding with no
spec behind it. Each of those is a named rule below, and each prints one line of
reason when it fires.

Scope is decided by the same diff `tests/affected.py` uses, never by whoever is
running it: a changed spec is audited, a changed harness audits its whole line, a
changed finding audits the specs it cites, and a change to the referee audits
everything. Nothing relevant changed -> nothing runs, green.

A spec that predates a rule is reported as grandfathered, with the date, and is
never rewritten: rewriting a pre-registration to satisfy a check written later
would be inventing provenance, which is the exact thing the audit exists to catch.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import subprocess
import tomllib
from dataclasses import dataclass, field

from instruments.changed import changed_paths
from instruments.verdict import (
    KEEP, KILL, Spec, Summary, Verdict, evaluate, load_recorded_results, load_spec,
    mean_sigma, pooled_sigma,
)
from trm.runtime.supervisor import BUDGET_COMPLETE, KILLED_DIVERGENCE, WALLCLOCK_COMPLETE

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC_GLOB = "experiments/*/specs/*.toml"
FINDINGS_DIR = pathlib.Path("docs/findings")
ROADMAP = pathlib.Path("docs/ROADMAP.md")
# Files whose change means every verdict in the repo may read differently.
REFEREE_FILES = {"instruments/verdict.py", "instruments/audit.py", "instruments/results.py",
                 "instruments/changed.py"}

# The doctrine commit (baa5c89): measure the floor, register the prediction, and a
# finding names its spec. Specs and findings from before it are grandfathered.
DOCTRINE_CUTOFF = datetime.date(2026, 9, 12)
# This audit. From here on, git must show the criteria commit preceding the
# results commit; every earlier spec was squash-merged and the order is gone.
AUDIT_CUTOFF = datetime.date(2026, 9, 15)

GREEN, RED, NA, GRANDFATHERED = "green", "red", "n/a", "grandfathered"

# What a training log's tail must say for a run to count (trm/runtime/supervisor.py).
RUN_ENDED_WELL = (BUDGET_COMPLETE, WALLCLOCK_COMPLETE)
RUN_ENDED_BADLY = (KILLED_DIVERGENCE, "Non-finite loss/grad")
LOG_TAIL_LINES = 40
# sigma-plausible: a noise floor this many times the gap it was asked to resolve
# means the criterion decided nothing the measurement could see.
SIGMA_DWARFS_GAP = 10.0


@dataclass(frozen=True)
class Status:
    rule: str
    state: str
    reason: str = ""


@dataclass(frozen=True)
class Commit:
    sha: str
    date: datetime.date
    text: str


# --- one spec, loaded once, with everything a rule may ask -------------------

@dataclass
class Audited:
    path: pathlib.Path
    rel: str                    # how the spec is named in output and in citations
    repo: pathlib.Path
    raw: dict
    spec: Spec
    results: dict
    verdict: Verdict | None
    verdict_error: str | None   # why the referee could not judge, if it could not
    _history: list[Commit] | None = field(default=None, repr=False)
    _history_loaded: bool = field(default=False, repr=False)

    @property
    def constants(self) -> set[str]:
        return {n for n, b in self.raw.get("arms", {}).items() if b.get("constant")}

    @property
    def measured(self) -> bool:
        """Has anything actually been run? Constant arms are declared, not measured."""
        return any(arm not in self.constants
                   for arms in self.results.values() for arm in arms)

    @property
    def outcome(self) -> str | None:
        """The verdict on record: the stored one, else what the referee reads now."""
        if self.spec.recorded:
            return self.spec.recorded
        return self.verdict.outcome if self.verdict else None

    def points_of(self, criterion) -> tuple[str, ...]:
        return criterion.points or tuple(sorted(self.results))

    def carrying_criteria(self):
        """The criteria the verdict rests on: every keep criterion for a KEEP, the
        kill criteria that fired for a KILL. An INCONCLUSIVE rests on nothing."""
        if self.verdict is None:
            return []
        held = {r.criterion.name for r in self.verdict.criteria if r.holds}
        if self.outcome == KEEP:
            return [self.spec.criteria[n] for n in self.spec.keep_if]
        if self.outcome == KILL:
            return [self.spec.criteria[n] for n in self.spec.kill_if if n in held]
        return []

    def role(self, arm: str) -> str:
        return self.raw.get("arms", {}).get(arm, {}).get("role", "")

    def floors(self) -> dict[str, Summary]:
        """{point: floor} from constant floor arms declared in [results]."""
        floor_arms = {n for n in self.constants if self.role(n) == "floor"}
        return {point: arms[name] for point, arms in self.results.items()
                for name in floor_arms if name in arms}

    @property
    def cap(self) -> float | None:
        cap = self.raw.get("protocol", {}).get("cap")
        return float(cap) if cap is not None else None

    def voided(self, point: str, arm: str) -> list[int]:
        """Seeds declared void at a point, with a reason: [void.<point>.<arm>]."""
        entry = self.raw.get("void", {}).get(point, {}).get(arm, {})
        return list(entry.get("seeds", ())) if str(entry.get("reason", "")).strip() else []

    def registered_seeds(self, point: str) -> int | None:
        """How many seeds the spec registered for a point, or None if it never said."""
        execution = self.raw.get("execution")
        if execution:
            default = execution.get("seeds", ())
            leg = point.split("/", 1)[0] if "/" in point else ""
            legs = execution.get("legs", {})
            if leg in legs:
                return len(legs[leg].get("seeds", default))
            return len(default) if default else None
        seeds = self.raw.get("protocol", {}).get("seeds")
        return len(seeds) if isinstance(seeds, list) else None

    def history(self) -> list[Commit] | None:
        """Every committed version of the spec, oldest first. [] if never committed,
        None if git cannot answer at all."""
        if not self._history_loaded:
            self._history = spec_history(self.repo, self.rel)
            self._history_loaded = True
        return self._history

    @property
    def first_commit(self) -> Commit | None:
        history = self.history()
        return history[0] if history else None


def load(path: pathlib.Path, repo: pathlib.Path = REPO) -> Audited:
    path, repo = path.resolve(), repo.resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    spec = load_spec(path)
    results = load_recorded_results(path)
    verdict, error = None, None
    try:
        verdict = evaluate(spec, results)
    except (KeyError, ValueError) as exc:
        error = str(exc)
    rel = path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)
    return Audited(path, rel, repo, raw, spec, results, verdict, error)


def spec_history(repo: pathlib.Path, rel: str) -> list[Commit] | None:
    """The spec's committed versions, oldest first. None when git cannot answer
    (not a repository); [] when the file has never been committed."""
    try:
        log = subprocess.run(["git", "log", "--format=%H %as", "--reverse", "--", rel],
                             cwd=repo, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    commits = []
    for line in log.splitlines():
        sha, date = line.split()
        text = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout
        commits.append(Commit(sha[:7], datetime.date.fromisoformat(date), text))
    return commits


def _parses(text: str) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _has_criteria(text: str) -> bool:
    return bool(_parses(text).get("criteria"))


def _has_measured_results(text: str, constants: set[str]) -> bool:
    return any(arm not in constants for arms in _parses(text).get("results", {}).values() for arm in arms)


# --- the rules ------------------------------------------------------------------
# Each takes the loaded spec and returns one Status. Printed in this order.

def _unjudgeable(a: Audited, rule: str) -> Status | None:
    """The numeric rules read the same tables the referee does; if it could not,
    they cannot either, and verdict-current is where that is reported."""
    if a.verdict_error:
        return Status(rule, NA, "the referee cannot read the recorded results; see verdict-current")
    return None


def _at_floor(arm, floor: Summary) -> bool:
    """An arm is at the floor when it clears it by less than two of its own sigmas."""
    mean, sigma = mean_sigma(arm)
    return mean - floor.mean <= 2 * max(sigma, floor.sigma)


def _at_cap(arm, cap: float) -> bool:
    mean, sigma = mean_sigma(arm)
    return cap - mean <= 2 * sigma


def floor_both_arms(a: Audited) -> Status:
    """No criterion that carried the verdict passed with both arms at the floor or
    cap (rule 1, #246 build 1). Two failures are trivially within 2 sigma."""
    rule = "floor-both-arms"
    if blocked := _unjudgeable(a, rule):
        return blocked
    floors, cap = a.floors(), a.cap
    if not floors and cap is None:
        return Status(rule, NA, "no floor arm or [protocol] cap declared; floor_note carries the exemption")
    carrying = a.carrying_criteria()
    if not carrying:
        return Status(rule, GREEN, f"verdict {a.outcome} rests on no criterion")
    for c in carrying:
        for point in a.points_of(c):
            arms = a.results[point]
            pair = (arms[c.treatment], arms[c.control])
            if point in floors and all(_at_floor(x, floors[point]) for x in pair):
                return Status(rule, RED, f"{c.name} passed at {point} with both arms at the floor "
                                         f"{floors[point].mean:g} — a shared failure read as a result")
            if cap is not None and all(_at_cap(x, cap) for x in pair):
                return Status(rule, RED, f"{c.name} passed at {point} with both arms at the cap {cap:g}")
    return Status(rule, GREEN)


def control_reached_target(a: Audited) -> Status:
    """For a tokens-to-target metric, every control seed reached the target, or the
    seed is marked void with a reason. A control at the cap is not a control."""
    rule = "control-reached-target"
    if blocked := _unjudgeable(a, rule):
        return blocked
    metric_key = str(a.raw.get("protocol", {}).get("metric_key", ""))
    if "tokens_to" not in metric_key and a.cap is None:
        return Status(rule, NA, "not a tokens-to-target metric")
    if a.cap is None:
        return Status(rule, RED, f"{metric_key} is a tokens-to-target metric with no [protocol] cap — "
                                 f"declare the cap so a control seed that never reached the target is visible")
    controls = {n for n in a.raw.get("arms", {}) if a.role(n) == "control"}
    for c in a.spec.criteria.values():
        for point in a.points_of(c):
            for arm in (c.treatment, c.control):
                values = a.results[point][arm]
                if arm not in controls or isinstance(values, Summary):
                    continue
                unreached = [v for v in values if v >= a.cap]
                if len(unreached) > len(a.voided(point, arm)):
                    return Status(rule, RED, f"control {arm} at {point}: {len(unreached)} seed(s) never reached "
                                             f"the target (at the cap {a.cap:g}) and are not marked void")
    return Status(rule, GREEN)


def seeds_complete(a: Audited) -> Status:
    """Every arm a criterion compares has every registered seed, or the missing ones
    are marked void (`[void.<point>.<arm>] seeds = [...], reason = "..."`)."""
    rule = "seeds-complete"
    if blocked := _unjudgeable(a, rule):
        return blocked
    short, unregistered = [], False
    for c in a.spec.criteria.values():
        for point in a.points_of(c):
            registered = a.registered_seeds(point)
            if registered is None:
                unregistered = True
                continue
            for arm in (c.treatment, c.control):
                if arm in a.constants:
                    continue
                values = a.results[point][arm]
                have = values.n if isinstance(values, Summary) else len(values)
                if have + len(a.voided(point, arm)) < registered:
                    short.append(f"{arm} at {point}: {have} of {registered} seeds")
    if short:
        return Status(rule, RED, "; ".join(short) + " — run the missing seeds, or mark them void with a reason")
    if unregistered:
        return Status(rule, NA, "the spec registers no seed list, so there is nothing to count against")
    return Status(rule, GREEN)


def sigma_plausible(a: Audited) -> Status:
    """sigma_pooled is a real noise floor: not zero anywhere a criterion applies.

    Zero is red: one seed, or seeds that repeat to the last digit, and a separation
    in sigmas is undefined (verdict.py maps it to +/-inf, which decides a `beats` on
    any gap at all). A sigma an order of magnitude ABOVE the gap is noted, not red:
    it is what a true null looks like about one time in ten (three seeds put
    |delta| under sigma/10 by chance), so a red there would fail a third of honest
    parity results. The note says what such a KEEP means — no difference the
    measurement could see, which is not the same as no difference.
    """
    rule = "sigma-plausible"
    if blocked := _unjudgeable(a, rule):
        return blocked
    framed = [c for c in a.spec.criteria.values() if c.sigmas > 0]
    if not framed:
        return Status(rule, NA, "criteria use absolute margins (sigmas = 0), not a sigma bar")
    carrying = {c.name for c in a.carrying_criteria()}
    notes = []
    for c in framed:
        for point in a.points_of(c):
            arms = a.results[point]
            sigma = pooled_sigma(arms[c.treatment], arms[c.control])
            if sigma == 0.0:
                return Status(rule, RED, f"sigma_pooled is 0 at {point} ({c.name}): one seed, or identical "
                                         f"seeds — a separation in sigmas is undefined here")
            gap = abs(mean_sigma(arms[c.treatment])[0] - mean_sigma(arms[c.control])[0])
            if c.name in carrying and sigma > SIGMA_DWARFS_GAP * gap:
                notes.append(f"{c.name} at {point}: sigma_pooled {sigma:.3g} is over {SIGMA_DWARFS_GAP:g}x the "
                             f"gap {gap:.3g} — parity here is 'no difference the measurement could see'")
    return Status(rule, GREEN, "; ".join(notes))


def prediction_registered(a: Audited) -> Status:
    """The hypothesis carries a registered prediction (rule 3): the criterion tests
    the idea, the prediction tests the person holding it."""
    rule = "prediction-registered"
    if re.search(r"predict", a.spec.hypothesis, re.IGNORECASE):
        return Status(rule, GREEN)
    first = a.first_commit
    if first and first.date < DOCTRINE_CUTOFF:
        return Status(rule, GRANDFATHERED, f"first committed {first.date}, before the doctrine asked for "
                                           f"a prediction ({DOCTRINE_CUTOFF})")
    return Status(rule, RED, "the hypothesis registers no prediction — write down what you expect "
                             "before the run, beside the criteria")


def criteria_predate_results(a: Audited) -> Status:
    """git shows the commit that introduced [criteria] preceding the one that
    introduced measured [results]. The file cannot prove its own order; git can."""
    rule = "criteria-predate-results"
    history = a.history()
    if history is None:
        return Status(rule, NA, "not under git")
    if not history:
        return Status(rule, RED, "the spec has never been committed — commit the criteria before running")
    constants = a.constants
    criteria = next((c for c in history if _has_criteria(c.text)), None)
    results = next((c for c in history if _has_measured_results(c.text, constants)), None)
    if criteria is None:
        return Status(rule, RED, "no committed version carries [criteria]")
    if results is None:
        return Status(rule, GREEN, f"criteria committed in {criteria.sha} ({criteria.date}); "
                                   f"results not yet committed")
    if criteria.sha == results.sha:
        if a.raw.get("experiment", {}).get("status") == "retrofit":
            return Status(rule, GRANDFATHERED, f"retrofit: both halves transcribed from the finding in "
                                               f"{criteria.sha} ({criteria.date})")
        if criteria.date < AUDIT_CUTOFF:
            return Status(rule, GRANDFATHERED, f"criteria and results share commit {criteria.sha} "
                                               f"({criteria.date}), before this audit ({AUDIT_CUTOFF}): a squash "
                                               f"merge folded the pre-registration into its results commit")
        return Status(rule, RED, f"criteria and results landed together in {criteria.sha} ({criteria.date}) — "
                                 f"git holds no evidence the criteria came first. Pre-register the spec in its "
                                 f"own PR (as 026-stage2 did), or merge without squashing")
    if history.index(criteria) < history.index(results):
        return Status(rule, GREEN, f"criteria {criteria.sha} ({criteria.date}) precede results "
                                   f"{results.sha} ({results.date})")
    return Status(rule, RED, f"results {results.sha} ({results.date}) were committed before criteria "
                             f"{criteria.sha} ({criteria.date})")


def verdict_current(a: Audited) -> Status:
    """Re-running the referee on the recorded numbers reproduces the stored verdict."""
    rule = "verdict-current"
    if a.verdict_error:
        return Status(rule, RED, f"the referee cannot judge the recorded numbers: {a.verdict_error}")
    if not a.spec.recorded:
        return Status(rule, NA, f"no verdict stored; the referee reads {a.verdict.outcome}")
    if a.spec.recorded == a.verdict.outcome:
        return Status(rule, GREEN)
    return Status(rule, RED, f"the spec records {a.spec.recorded} but the referee reads "
                             f"{a.verdict.outcome} from the same numbers")


def finding_cites_spec(a: Audited, citations: dict[str, list[str]], graveyard: str) -> Status:
    """A KEEP or KILL is repo knowledge (rule 5): a findings entry cites the spec,
    or a graveyard tombstone names it (by path or by issue number)."""
    rule = "finding-cites-spec"
    if a.outcome not in (KEEP, KILL):
        return Status(rule, NA, f"verdict {a.outcome or 'undecidable'} — nothing claimed, nothing to cite")
    if a.rel in citations:
        return Status(rule, GREEN, f"cited by {', '.join(citations[a.rel])}")
    finding = a.raw.get("experiment", {}).get("finding")
    if finding and (a.repo / finding).exists():
        return Status(rule, GREEN, f"links its own finding, {finding}")
    issue = a.raw.get("experiment", {}).get("issue")
    if a.rel in graveyard or (issue and re.search(rf"(?<![\d#])#{int(issue)}(?!\d)", graveyard)):
        return Status(rule, GREEN, "named in the ROADMAP graveyard")
    return Status(rule, RED, f"{a.outcome} with no findings entry citing this spec (a `Spec:` line) and no "
                             f"graveyard tombstone naming it — a verdict nobody wrote down is not knowledge")


def run_ended_by_budget(a: Audited) -> Status:
    """A recorded run ended at its budget, not in non-finite skips or a divergence
    kill. Applies only when the spec names its logs: `[runs.<arm>] logs = [...]` or
    `dirs = [...]` (a run dir contributes its train.log and supervisor log)."""
    rule = "run-ended-by-budget"
    runs = a.raw.get("runs")
    if not runs:
        return Status(rule, NA, "the spec records no run log or run dir")
    logs, missing, bad = [], [], []
    for arm, body in runs.items():
        paths = [a.repo / p for p in body.get("logs", ())]
        for d in body.get("dirs", ()):
            run_dir = a.repo / d
            paths += [run_dir / "train.log", run_dir.parent / f"{run_dir.name}.supervisor.log"]
        for path in paths:
            if not path.exists():
                missing.append(path.name)
                continue
            logs.append(path)
            tail = "\n".join(path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])
            ended_badly = next((m for m in RUN_ENDED_BADLY if m in tail), None)
            if ended_badly:
                bad.append(f"{arm}: {path.name} ends in {ended_badly}")
            elif not any(m in tail for m in RUN_ENDED_WELL):
                bad.append(f"{arm}: {path.name} shows no budget stop in its last {LOG_TAIL_LINES} lines")
    if bad:
        return Status(rule, RED, "; ".join(bad) + " — a run that did not end at its budget is flagged, not averaged in")
    if not logs:
        return Status(rule, NA, f"no recorded log is present on this machine ({', '.join(missing)})")
    return Status(rule, GREEN, f"{len(logs)} log(s) end at a budget stop")


def audit_spec(a: Audited, citations: dict[str, list[str]], graveyard: str) -> list[Status]:
    return [
        floor_both_arms(a),
        control_reached_target(a),
        seeds_complete(a),
        sigma_plausible(a),
        prediction_registered(a),
        criteria_predate_results(a),
        verdict_current(a),
        finding_cites_spec(a, citations, graveyard),
        run_ended_by_budget(a),
    ]


# --- findings: the other direction of finding-cites-spec ------------------------

DATED = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-")
SPEC_LINE = re.compile(r"^Spec:\s*(.*)$")
OBSERVATIONAL = re.compile(r"^Evidence:\s*observational\b\s*[—-]\s*\S+", re.M)


def spec_citations(text: str) -> list[str]:
    """The paths a finding's `Spec:` line names. The line continues onto indented
    lines while it ends with a comma — the 2026-09-12 entry cites three specs that way."""
    cited, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = SPEC_LINE.match(lines[i])
        i += 1
        if not m:
            continue
        chunk = m.group(1)
        while chunk.rstrip().endswith(",") and i < len(lines) and lines[i][:1].isspace():
            chunk += lines[i]
            i += 1
        cited += [c.strip() for c in chunk.split(",") if c.strip()]
    return cited


def finding_date(path: pathlib.Path) -> datetime.date | None:
    m = DATED.match(path.name)
    return datetime.date(*(int(g) for g in m.groups())) if m else None


def audit_finding(path: pathlib.Path, repo: pathlib.Path = REPO) -> Status:
    """A dated finding names its spec, or says it is observational and why."""
    rule = "finding-cites-spec"
    date = finding_date(path)
    if date is None:
        return Status(rule, NA, "not a dated entry")
    if date < DOCTRINE_CUTOFF:
        return Status(rule, GRANDFATHERED, f"dated {date}, before findings carried provenance ({DOCTRINE_CUTOFF})")
    text = path.read_text()
    cited = spec_citations(text)
    if cited:
        missing = [c for c in cited if not (repo / c).exists()]
        if missing:
            return Status(rule, RED, f"cites specs that do not exist: {missing} — a citation that cannot be "
                                     f"followed is worse than none")
        return Status(rule, GREEN, f"cites {len(cited)} spec(s)")
    if OBSERVATIONAL.search(text):
        return Status(rule, GREEN, "observational, with the reason stated")
    return Status(rule, RED, "carries neither a `Spec:` line nor `Evidence: observational — <why>`; "
                             "say which pre-registered spec judged it, or that no control applies")


def all_findings(repo: pathlib.Path = REPO) -> list[str]:
    return sorted(p.relative_to(repo).as_posix() for p in (repo / FINDINGS_DIR).glob("*.md"))


def all_citations(repo: pathlib.Path = REPO) -> dict[str, list[str]]:
    """{spec path: [findings that cite it]} over every finding in the repo."""
    cited: dict[str, list[str]] = {}
    for rel in all_findings(repo):
        for spec in spec_citations((repo / rel).read_text()):
            cited.setdefault(spec, []).append(pathlib.PurePosixPath(rel).name)
    return cited


def graveyard_text(repo: pathlib.Path = REPO) -> str:
    text = (repo / ROADMAP).read_text() if (repo / ROADMAP).exists() else ""
    marker = "## Graveyard"
    return text[text.index(marker):] if marker in text else ""


# --- scope: which specs can a change reach ----------------------------------------

@dataclass
class Scope:
    specs: set[str] = field(default_factory=set)      # spec paths, repo-relative
    findings: set[str] = field(default_factory=set)   # finding paths, repo-relative
    everything: bool = False
    why: list[str] = field(default_factory=list)


def all_specs(repo: pathlib.Path = REPO) -> list[str]:
    return sorted(p.relative_to(repo).as_posix() for p in repo.glob(SPEC_GLOB))


def affected(changed: list[str], repo: pathlib.Path = REPO) -> Scope:
    """Map changed paths to the specs (and findings) they can reach. Deleted files
    reach nothing: there is no spec left to audit."""
    scope = Scope()
    for rel in changed:
        p = pathlib.PurePosixPath(rel)
        parts = p.parts
        if rel in REFEREE_FILES:
            scope.everything = True
            scope.why.append(f"{rel} -> every spec and finding")
        elif parts[:1] == ("experiments",) and len(parts) == 4 and parts[2] == "specs" and p.suffix == ".toml":
            if (repo / rel).exists():
                scope.specs.add(rel)
                scope.why.append(f"{rel} -> itself")
        elif parts[:1] == ("experiments",) and len(parts) >= 3 and p.suffix == ".py":
            line = sorted(q.relative_to(repo).as_posix()
                          for q in (repo / "experiments" / parts[1] / "specs").glob("*.toml"))
            scope.specs.update(line)
            scope.why.append(f"{rel} -> every spec in experiments/{parts[1]} ({len(line)})")
        elif parts[:2] == ("docs", "findings") and p.suffix == ".md" and (repo / rel).exists():
            cited = [c for c in spec_citations((repo / rel).read_text()) if (repo / c).exists()]
            scope.findings.add(rel)
            scope.specs.update(cited)
            scope.why.append(f"{rel} -> the {len(cited)} spec(s) it cites")
    return scope


# --- output -------------------------------------------------------------------------

def render_spec(a: Audited, statuses: list[Status]) -> str:
    verdict = a.spec.recorded or (f"{a.verdict.outcome} (referee)" if a.verdict else "undecidable")
    worst = RED if any(s.state == RED for s in statuses) else GREEN
    lines = [f"{a.rel}  [{verdict}]  {worst.upper()}"]
    for s in statuses:
        lines.append(f"  {s.rule:<26} {s.state:<14} {s.reason}".rstrip())
    return "\n".join(lines)


def render_pending(a: Audited) -> str:
    status = a.raw.get("experiment", {}).get("status", "?")
    return f"{a.rel}  [pending]  no measured results yet (status = {status}); nothing to audit"


def run(spec_paths: list[str], finding_paths: list[str], repo: pathlib.Path = REPO) -> tuple[list[str], bool]:
    """Audit the given specs and findings. Returns the report blocks and whether any rule went red."""
    citations, graveyard = all_citations(repo), graveyard_text(repo)
    blocks, red = [], False
    for rel in spec_paths:
        a = load(repo / rel, repo)
        if not a.measured:
            blocks.append(render_pending(a))
            continue
        statuses = audit_spec(a, citations, graveyard)
        red |= any(s.state == RED for s in statuses)
        blocks.append(render_spec(a, statuses))
    findings, grandfathered = [], 0
    for rel in finding_paths:
        s = audit_finding(repo / rel, repo)
        red |= s.state == RED
        if s.state == GRANDFATHERED:
            grandfathered += 1  # one line for all of them: the date is the whole reason
            continue
        findings.append(f"{rel}  {s.rule} {s.state}  {s.reason}".rstrip())
    if grandfathered:
        findings.append(f"{grandfathered} finding(s) dated before {DOCTRINE_CUTOFF}  finding-cites-spec grandfathered")
    if findings:
        blocks.append("\n".join(findings))
    return blocks, red


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="every spec and every finding")
    ap.add_argument("--spec", type=pathlib.Path, action="append", default=[], help="one spec (repeatable)")
    ap.add_argument("--base", default="main", help="branch to diff against for scoping (default: main)")
    ap.add_argument("--since", default=None, help="audit what the commits since this SHA touched (a push)")
    args = ap.parse_args(argv)

    repo = REPO
    if args.spec:
        specs, findings = [str(p.resolve()) for p in args.spec], []
    elif args.all:
        specs, findings = all_specs(repo), all_findings(repo)
    else:
        try:
            changed = changed_paths(repo, base=args.base, since=args.since)
            scope = affected(changed, repo)
        except subprocess.CalledProcessError as exc:
            print(f"cannot resolve the diff ({(exc.stderr or '').strip() or exc}); auditing everything instead")
            changed, scope = [], Scope(everything=True)
        if scope.everything:
            specs, findings = all_specs(repo), all_findings(repo)
        else:
            specs, findings = sorted(scope.specs), sorted(scope.findings)
        if not specs and not findings:
            print("nothing an audit can reach changed")
            return 0
        print(f"{len(changed)} changed path(s) -> {len(specs)} spec(s), {len(findings)} finding(s)")
        for why in scope.why:
            print(f"  {why}")
        print()

    blocks, red = run(specs, findings, repo)
    print("\n\n".join(blocks))
    print()
    print("AUDIT RED — a verdict above cannot be trusted as recorded" if red else "audit green")
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
