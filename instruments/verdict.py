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

**Beside sigma, a t-test.** sigma_pooled is the seeds' spread, not the error of
the means, so a gap of k sigmas with n seeds a side is a t of k*sqrt(n/2): the 2-sigma
bar is one-sided p 0.035 at three seeds and 0.09 at two (#599). Each point reports
the one-sided p of the claim its rule makes, so that is visible. `within` reads
parity off a gap that was merely not shown; `equivalent` asks the gap to be shown
inside the margin (a TOST), which is what a parity claim means.

A verdict is deliberately three-valued. KEEP and KILL are the pre-registered
branches; INCONCLUSIVE is what you get when neither fired, and it is a real
answer — #77 landed there twice and the honest record says so.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from instruments.spec_file import SpecFile, SummaryTable

# One-sided significance for the t-test each point reports (#599). For `equivalent`
# it is a TOST: the gap's 90% confidence interval lies inside +/- the margin.
ALPHA = 0.05

# Comparison rules a criterion may use. Each takes the signed separation in
# sigmas (treatment minus control), the bar, and the point's p (None when sigma is
# 0 or an arm has one seed), and says whether it holds.
RULES = {
    # No difference seen: the gap is inside the bar, whether or not the seeds could
    # have shown one. Kept as registered; a new parity claim uses `equivalent`.
    "within": lambda sigmas, bar, p: abs(sigmas) <= bar,
    # Parity shown: the gap is inside +/- bar sigmas at confidence 1 - ALPHA.
    "equivalent": lambda sigmas, bar, p: p is not None and p <= ALPHA,
    # A win: treatment is ahead by at least the bar.
    "beats": lambda sigmas, bar, p: sigmas >= bar,
    # A loss big enough to trigger a kill branch.
    "loses": lambda sigmas, bar, p: sigmas <= -bar,
}
PARITY_RULES = ("within", "equivalent")

# The continued fraction behind the t distribution's tail; the audit's CI job has
# no scipy, so the referee carries its own.
_BETA_TERMS = 200
_BETA_EPS = 1e-12
_BETA_TINY = 1e-300

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


def seeds(arm: Arm) -> int:
    return arm.n if isinstance(arm, Summary) else len(arm)


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


def _nonzero(x: float) -> float:
    return x if abs(x) > _BETA_TINY else _BETA_TINY


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta I_x(a, b), by Lentz's continued fraction
    (Numerical Recipes, section 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x)) / a
    c, d = 1.0, 1.0 / _nonzero(1.0 - (a + b) * x / (a + 1))
    f = d
    for m in range(1, _BETA_TERMS):
        step = 1.0
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 / _nonzero(1.0 + num * d)
            c = _nonzero(1.0 + num / c)
            step = c * d
            f *= step
        if abs(step - 1.0) < _BETA_EPS:
            break
    return front * f


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def p_value(rule: str, sigmas: float, bar: float, n_treatment: int, n_control: int) -> float | None:
    """One-sided p of the claim a rule makes, from a two-sample t on the separation:
    t = sigmas / sqrt(1/n_t + 1/n_c) with n_t + n_c - 2 degrees of freedom (Student's
    pooled t when the arms have equal seeds, since sigma_pooled is their RMS).
    `beats` claims the treatment ahead, `loses` behind, and a parity rule claims the
    gap inside +/- bar: the larger p of the TOST's two one-sided tests.
    None when there is no spread to test against (sigma 0, or too few seeds)."""
    df = n_treatment + n_control - 2
    if df < 1 or not math.isfinite(sigmas):
        return None
    scale = math.sqrt(1 / n_treatment + 1 / n_control)
    if rule == "beats":
        return t_sf(sigmas / scale, df)
    if rule == "loses":
        return t_sf(-sigmas / scale, df)
    return max(t_sf((sigmas + bar) / scale, df), t_sf((bar - sigmas) / scale, df))


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
    # An absolute margin in the metric's own units, on top of the sigma bar (#26):
    # "beats" then needs delta >= min_delta as well, "loses" delta <= -min_delta.
    # A 1.3x claim is a ~30% difference on a tokens axis; with tight seeds a 3%
    # difference clears 2 sigma and would otherwise count as the win.
    min_delta: float | None = None

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"{self.name}: unknown rule {self.rule!r}; expected one of {sorted(RULES)}")
        if self.min_delta is not None and self.rule in PARITY_RULES:
            raise ValueError(f"{self.name}: min_delta makes no sense for a parity rule")
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
    # The whole file, typed: arms, protocol, execution, results. None for a Spec built
    # in code rather than read from a file.
    file: SpecFile | None = None

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
    delta: float = 0.0  # treatment mean minus control mean, in the metric's units
    p: float | None = None  # one-sided p of the rule's claim (p_value); None at sigma 0
    seeds: tuple[int, int] = (0, 0)  # (treatment, control)

    def describe(self) -> str:
        p = f", p {self.p:.3f}" if self.p is not None else ""
        return f"{self.point}: {self.sigmas:+.2f}σ (Δ {self.delta:+.4g}{p})"


@dataclass(frozen=True)
class CriterionResult:
    criterion: Criterion
    points: tuple[PointResult, ...]
    holds: bool

    def describe(self) -> str:
        bar = f"{self.criterion.rule} {self.criterion.sigmas}σ"
        if self.criterion.min_delta is not None:
            bar += f" and |Δ| ≥ {self.criterion.min_delta:g}"
        detail = ", ".join(p.describe() for p in self.points)
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
        treatment, control = arms[criterion.treatment], arms[criterion.control]
        sig = separation(treatment, control)
        delta = mean_sigma(treatment)[0] - mean_sigma(control)[0]
        n = (seeds(treatment), seeds(control))
        p = p_value(criterion.rule, sig, criterion.sigmas, *n) if pooled_sigma(treatment, control) > 0 else None
        holds = RULES[criterion.rule](sig, criterion.sigmas, p)
        if criterion.min_delta is not None:
            holds = holds and {"beats": delta >= criterion.min_delta,
                               "loses": delta <= -criterion.min_delta}[criterion.rule]
        evaluated.append(PointResult(point, sig, holds, delta, p, n))
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
    """Read a spec TOML through its typed tables (instruments/spec_file.py): a
    malformed spec must fail before a run burns GPU hours, not after."""
    file = SpecFile.load(path)
    criteria = {name: Criterion(name=name, rule=c.rule, treatment=c.treatment, control=c.control,
                                sigmas=c.sigmas, points=c.points, require=c.require, min_delta=c.min_delta)
                for name, c in file.criteria.items()}
    return Spec(
        id=file.experiment.id,
        title=file.experiment.title,
        hypothesis=file.experiment.hypothesis,
        criteria=criteria,
        keep_if=file.verdict.keep_if,
        kill_if=file.verdict.kill_if,
        readouts=file.readouts.names,
        recorded=file.verdict.recorded,
        file=file,
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
    def parse(arm):
        if isinstance(arm, SummaryTable):
            return Summary(mean=arm.mean, sigma=arm.sigma, n=arm.n)
        return list(arm)

    return {point: {name: parse(arm) for name, arm in arms.items()}
            for point, arms in SpecFile.load(path).results.items()}
