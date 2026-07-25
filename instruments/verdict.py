"""The referee: recorded numbers plus a pre-registered spec, in — verdict out.

Every finding in this repo has recomputed sigma-pooled and the 2-sigma bar by
hand, in prose. That works right up until it doesn't: the arithmetic is easy to
get subtly wrong, the convention is easy to drift, and a threshold written in
prose needs a reader to apply it. A threshold written in a spec file can be
applied by code — and code cannot move a goalpost after seeing the number, which
is the whole point of rule 3.

**The convention, pinned.** Sample standard deviation (ddof=1, `statistics.stdev`),
and sigma-pooled as the RMS of the two arms' sample sigmas:

    sigma_pooled = sqrt((s_treatment^2 + s_control^2) / 2)

This is not a choice made here; it is the one every recorded finding already
used, and `test_verdict.py` proves it by reproducing their published sigma
figures from their published per-seed numbers. If this module ever disagrees
with a finding, one of the two is wrong and that is worth knowing.

A verdict is deliberately three-valued. KEEP and KILL are the pre-registered
branches; INCONCLUSIVE is what you get when neither fired, and it is a real
answer — #77 landed there twice and the honest record says so.
"""

from __future__ import annotations

import math
import statistics
import tomllib
from dataclasses import dataclass, field
from typing import Any, Sequence

# Comparison rules a criterion may use. Each takes the signed separation in
# sigmas (treatment minus control) and the bar, and says whether it holds.
RULES = {
    # Parity / no-harm: the arms are indistinguishable at this resolution.
    "within": lambda sigmas, bar: abs(sigmas) <= bar,
    # A win: treatment is ahead by at least the bar.
    "beats": lambda sigmas, bar: sigmas >= bar,
    # A loss big enough to trigger a kill branch.
    "loses": lambda sigmas, bar: sigmas <= -bar,
}

KEEP, KILL, INCONCLUSIVE = "KEEP", "KILL", "INCONCLUSIVE"


@dataclass(frozen=True)
class Summary:
    """An arm reported as mean + sigma rather than per-seed values.

    Needed because that is how some results genuinely exist: several findings
    published per-point means and sigma_pooled without the underlying seeds. The
    alternative — reconstructing per-seed numbers that reproduce the published
    mean and sigma — would be inventing measurements that were never taken, so
    the format carries the coarser truth instead of a prettier fiction.
    """

    mean: float
    sigma: float
    n: int

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")


Arm = Sequence[float] | Summary


def mean_sigma(arm: Arm) -> tuple[float, float]:
    """Mean and SAMPLE sigma of an arm, given either per-seed values or a Summary.

    A single seed has no spread — sigma 0, and the caller is responsible for
    knowing that one seed is not a result (rule 1)."""
    if isinstance(arm, Summary):
        return arm.mean, arm.sigma
    values = list(arm)
    if not values:
        raise ValueError("no values to summarize")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def pooled_sigma(treatment: Arm, control: Arm) -> float:
    """RMS of the two arms' sample sigmas — the repo's noise-floor convention."""
    _, st = mean_sigma(treatment)
    _, sc = mean_sigma(control)
    return math.sqrt((st**2 + sc**2) / 2)


def separation(treatment: Arm, control: Arm) -> float:
    """Signed separation in sigmas: positive means the treatment is ahead.

    Zero pooled sigma is not an infinite result. It means both arms were
    perfectly repeatable at this precision, which happens on toy tasks; treat a
    real gap as unbounded-but-decided and an exact tie as zero, rather than
    returning a NaN that quietly poisons a comparison.
    """
    mt, _ = mean_sigma(treatment)
    mc, _ = mean_sigma(control)
    delta = mt - mc
    sigma = pooled_sigma(treatment, control)
    if sigma == 0.0:
        return 0.0 if delta == 0.0 else math.copysign(math.inf, delta)
    return delta / sigma


@dataclass(frozen=True)
class Criterion:
    """One pre-registered comparison. `points` names the sweep points it covers;
    `require` says whether every point must satisfy it or merely one."""

    name: str
    rule: str
    treatment: str
    control: str
    sigmas: float
    points: tuple[str, ...] = ()
    require: str = "all"  # all | any

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"{self.name}: unknown rule {self.rule!r}; expected one of {sorted(RULES)}")
        if self.require not in ("all", "any"):
            raise ValueError(f"{self.name}: require must be 'all' or 'any', got {self.require!r}")


@dataclass(frozen=True)
class Spec:
    id: str
    title: str
    hypothesis: str
    criteria: dict[str, Criterion]
    keep_if: tuple[str, ...]
    kill_if: tuple[str, ...]
    readouts: tuple[str, ...] = ()
    # For a completed experiment: the verdict its finding reached. Present so a
    # transcription error shows up as a test failure rather than as a spec that
    # quietly disagrees with the record it was copied from.
    recorded: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        unknown = [n for n in (*self.keep_if, *self.kill_if) if n not in self.criteria]
        if unknown:
            raise ValueError(f"keep_if/kill_if name criteria that do not exist: {unknown}")
        if not self.keep_if and not self.kill_if:
            raise ValueError("a spec with no keep_if and no kill_if pre-registers nothing")


@dataclass(frozen=True)
class PointResult:
    point: str
    sigmas: float
    holds: bool


@dataclass(frozen=True)
class CriterionResult:
    criterion: Criterion
    points: tuple[PointResult, ...]
    holds: bool

    def describe(self) -> str:
        bar = f"{self.criterion.rule} {self.criterion.sigmas}σ"
        detail = ", ".join(f"{p.point}: {p.sigmas:+.2f}σ" for p in self.points)
        return f"{self.criterion.name} [{bar}, {self.criterion.require}] → {'met' if self.holds else 'NOT met'} ({detail})"


@dataclass(frozen=True)
class Verdict:
    outcome: str
    criteria: tuple[CriterionResult, ...]
    reason: str

    def describe(self) -> str:
        lines = [f"verdict: {self.outcome} — {self.reason}"]
        lines += [f"  {c.describe()}" for c in self.criteria]
        return "\n".join(lines)


def _criterion_result(criterion: Criterion, results: dict[str, dict[str, Arm]]) -> CriterionResult:
    points = criterion.points or tuple(sorted(results))
    evaluated = []
    for point in points:
        if point not in results:
            raise KeyError(f"{criterion.name}: no results recorded for point {point!r}")
        arms = results[point]
        for arm in (criterion.treatment, criterion.control):
            if arm not in arms:
                raise KeyError(f"{criterion.name}: point {point!r} has no arm {arm!r}")
        sig = separation(arms[criterion.treatment], arms[criterion.control])
        evaluated.append(PointResult(point, sig, RULES[criterion.rule](sig, criterion.sigmas)))
    holds = (all if criterion.require == "all" else any)(p.holds for p in evaluated)
    return CriterionResult(criterion, tuple(evaluated), holds)


def evaluate(spec: Spec, results: dict[str, dict[str, Arm]]) -> Verdict:
    """Apply a spec's pre-registered criteria to recorded per-seed numbers.

    `results` is {sweep point: {arm: [value per seed]}}.

    Kill is checked first and wins ties: a spec that manages to satisfy its keep
    bar while also tripping a kill bar is a spec whose author did not think the
    branches through, and the safe reading of "this fired the kill condition" is
    that the idea is dead.
    """
    evaluated = {name: _criterion_result(c, results) for name, c in spec.criteria.items()}

    fired = [n for n in spec.kill_if if evaluated[n].holds]
    if fired:
        return Verdict(KILL, tuple(evaluated.values()), f"kill criteria fired: {', '.join(fired)}")

    if spec.keep_if:
        unmet = [n for n in spec.keep_if if not evaluated[n].holds]
        if not unmet:
            return Verdict(KEEP, tuple(evaluated.values()), f"all keep criteria met: {', '.join(spec.keep_if)}")
        return Verdict(INCONCLUSIVE, tuple(evaluated.values()), f"keep criteria not met: {', '.join(unmet)}")

    return Verdict(INCONCLUSIVE, tuple(evaluated.values()), "no kill criterion fired and none was pre-registered to keep")


def load_spec(path) -> Spec:
    """Read a spec TOML. Every key is validated here rather than at use time —
    a malformed spec must fail before a run burns GPU hours, not after."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    for section in ("experiment", "criteria", "verdict"):
        if section not in raw:
            raise ValueError(f"{path}: missing [{section}] section")

    exp = raw["experiment"]
    criteria = {}
    for name, body in raw["criteria"].items():
        criteria[name] = Criterion(
            name=name,
            rule=body["rule"],
            treatment=body["treatment"],
            control=body["control"],
            sigmas=float(body["sigmas"]),
            points=tuple(body.get("points", ())),
            require=body.get("require", "all"),
        )

    return Spec(
        id=str(exp["id"]),
        title=exp["title"],
        hypothesis=exp["hypothesis"],
        criteria=criteria,
        keep_if=tuple(raw["verdict"].get("keep_if", ())),
        kill_if=tuple(raw["verdict"].get("kill_if", ())),
        readouts=tuple(raw.get("readouts", {}).get("names", ())),
        recorded=raw["verdict"].get("recorded"),
        meta={k: v for k, v in raw.items() if k not in ("experiment", "criteria", "verdict")},
    )


def load_recorded_results(path) -> dict[str, dict[str, Arm]]:
    """The `[results.<point>]` tables of a completed spec: {point: {arm: ...}}.

    An arm is either a list of per-seed values, or an inline table
    `{mean = ..., sigma = ..., n = ...}` for results that only exist in
    summarized form (see Summary).

    Results live in the same file as the criteria, which is only honest because
    git records the order — a spec is committed with its criteria BEFORE its run,
    and the results commit comes after. For the specs retrofitted from findings
    already published, both arrive together and the finding is the source; those
    are marked `status = "retrofit"`.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    def parse(arm):
        if isinstance(arm, dict):
            return Summary(mean=float(arm["mean"]), sigma=float(arm["sigma"]), n=int(arm["n"]))
        return [float(v) for v in arm]

    return {point: {name: parse(arm) for name, arm in arms.items()}
            for point, arms in raw.get("results", {}).items()}
