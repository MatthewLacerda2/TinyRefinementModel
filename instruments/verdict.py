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

**Two looks, when a spec registers them** (`[stages]`, #604). A parity pair may
judge at its first few seeds and stop if that settles it. Each look then has its
own bar, from a Lan-DeMets alpha-spending function (O'Brien-Fleming by default), so
looking twice keeps the overall error at ALPHA. Without `[stages]`, a spec is
judged once, exactly as before (tests/apparatus/test_verdict_pinned.py).

A verdict is deliberately three-valued. KEEP and KILL are the pre-registered
branches; INCONCLUSIVE is what you get when neither fired, and it is a real
answer — #77 landed there twice and the honest record says so.
"""

from __future__ import annotations

import functools
import math
import statistics
from dataclasses import dataclass, replace
from typing import Sequence

from instruments.spec_file import SpecFile, StagesTable, SummaryTable

# One-sided significance for the t-test each point reports (#599). For `equivalent`
# it is a TOST: the gap's 90% confidence interval lies inside +/- the margin.
ALPHA = 0.05

# Comparison rules a criterion may use. Each takes the signed separation in
# sigmas (treatment minus control), the bar, the point's p (None when sigma is 0
# or an arm has one seed) and the look's alpha (ALPHA at a single look), and says
# whether it holds.
RULES = {
    # No difference seen: the gap is inside the bar, whether or not the seeds could
    # have shown one. Kept as registered; a new parity claim uses `equivalent`.
    "within": lambda sigmas, bar, p, alpha: abs(sigmas) <= bar,
    # Parity shown: the gap is inside the margin (+/- bar sigmas, or the registered
    # absolute `margin`) at confidence 1 - alpha.
    "equivalent": lambda sigmas, bar, p, alpha: p is not None and p <= alpha,
    # A win: treatment is ahead by at least the bar.
    "beats": lambda sigmas, bar, p, alpha: sigmas >= bar,
    # A loss big enough to trigger a kill branch.
    "loses": lambda sigmas, bar, p, alpha: sigmas <= -bar,
}
# The rules a `[stages]` look spends alpha on (#604). `beats` and `loses` are sigma
# bars, not alpha-calibrated, and keep their fixed design: at three seeds a side a
# `beats` pair already stops at its first look (the issue's simulation).
STAGED_RULES = ("equivalent",)
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
    (Numerical Recipes, section 6.4). The fraction converges below x = (a+1)/(a+b+2);
    above it, one reflection, never a second (rounding at the threshold would bounce)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_fraction(b, a, 1.0 - x)
    return _beta_fraction(a, b, x)


def _beta_fraction(a: float, b: float, x: float) -> float:
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


def _t_frame(treatment: Arm, control: Arm) -> tuple[float, float, int] | None:
    """(delta, its standard error, degrees of freedom) for Student's two-sample t:
    the arms' variances pooled by degrees of freedom, n_t + n_c - 2 of them. With
    equal seeds that pooled sigma is sigma_pooled; with unequal ones it weights the
    larger arm, as the RMS does not. None when there is no spread to test against
    (sigma 0, or too few seeds)."""
    (mt, st), (mc, sc) = mean_sigma(treatment), mean_sigma(control)
    nt, nc = seeds(treatment), seeds(control)
    df = nt + nc - 2
    if df < 1:
        return None
    spread = math.sqrt(((nt - 1) * st**2 + (nc - 1) * sc**2) / df)
    if spread == 0.0:
        return None
    return mt - mc, spread * math.sqrt(1 / nt + 1 / nc), df


def p_value(rule: str, treatment: Arm, control: Arm, bar: float,
            margin: float | None = None) -> float | None:
    """One-sided p of the claim a rule makes, by Student's two-sample t (_t_frame).
    `beats` claims the treatment ahead, `loses` behind, and a parity rule the gap
    inside +/- bar * sigma_pooled (the referee's convention), or inside +/- `margin`
    when one is registered in the metric's units: the larger p of the TOST's two
    one-sided tests. None when there is no spread to test against."""
    frame = _t_frame(treatment, control)
    if frame is None:
        return None
    delta, se, df = frame
    if rule == "beats":
        return t_sf(delta / se, df)
    if rule == "loses":
        return t_sf(-delta / se, df)
    if margin is None:
        margin = bar * pooled_sigma(treatment, control)
    return max(t_sf((delta + margin) / se, df), t_sf((margin - delta) / se, df))


def beyond_margin_p(treatment: Arm, control: Arm, margin: float) -> float | None:
    """One-sided p that the gap lies outside +/- margin, on the side it fell: the
    futility test of a parity claim (#604). Small means parity is already ruled out."""
    frame = _t_frame(treatment, control)
    if frame is None:
        return None
    delta, se, df = frame
    return t_sf((abs(delta) - margin) / se, df)


# --- two looks: alpha spending (#604) ------------------------------------------
#
# Lan & DeMets (1983) spend the overall one-sided alpha as a function of the
# information fraction t (here look 1's seeds over look 2's). With two looks:
#   look 1's bar c1 has P(Z1 >= c1) = spend(t1);
#   look 2's bar c2 has P(Z1 < c1, Z2 >= c2) = alpha - spend(t1), corr(Z1, Z2) = sqrt(t1).
# Each bar is handed to the t-test as its nominal one-sided p, 1 - Phi(c), the
# usual way a group-sequential bound meets a small-sample test.
_NORMAL = statistics.NormalDist()
SPENDING = {
    # O'Brien-Fleming-type: almost nothing early, so look 2's bar stays near alpha.
    "obrien-fleming": lambda t, alpha: 2 * (1 - _NORMAL.cdf(_NORMAL.inv_cdf(1 - alpha / 2) / math.sqrt(t))),
    # Pocock-type: spends alpha evenly in t.
    "pocock": lambda t, alpha: alpha * math.log(1 + (math.e - 1) * t),
}
# Simpson's rule over Z1 for look 2's crossing probability: the normal density
# below -8.5 is under 1e-16, and 4,000 steps put the error far below the 1e-4 a
# bar is quoted to.
_Z_FLOOR = -8.5
_SIMPSON_STEPS = 4000
_BISECTIONS = 60


def _crossed_at_look_2(c1: float, c2: float, rho: float) -> float:
    """P(Z1 < c1, Z2 >= c2) for standard normals with correlation rho."""
    tail = math.sqrt(1 - rho * rho)

    def density(z):
        return _NORMAL.pdf(z) * (1 - _NORMAL.cdf((c2 - rho * z) / tail))

    h = (c1 - _Z_FLOOR) / _SIMPSON_STEPS
    total = density(_Z_FLOOR) + density(c1)
    total += 4 * sum(density(_Z_FLOOR + (2 * i - 1) * h) for i in range(1, _SIMPSON_STEPS // 2 + 1))
    total += 2 * sum(density(_Z_FLOOR + 2 * i * h) for i in range(1, _SIMPSON_STEPS // 2))
    return total * h / 3


@functools.lru_cache(maxsize=None)
def look_bounds(spending: str, fraction: float, alpha: float = ALPHA) -> tuple[float, float]:
    """The two looks' z bars, for a look 1 at `fraction` of look 2's information."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"look 1 must come before look 2, got an information fraction of {fraction}")
    first = SPENDING[spending](fraction, alpha)
    c1 = _NORMAL.inv_cdf(1 - first)
    rho, rest = math.sqrt(fraction), alpha - first
    lo, hi = 0.0, c1 + 10.0
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _crossed_at_look_2(c1, mid, rho) > rest else (lo, mid)
    return c1, (lo + hi) / 2


@dataclass(frozen=True)
class Look:
    """One registered look: its number, seeds per arm, and its nominal one-sided bar."""
    index: int
    of: int
    seeds: int
    alpha: float


def looks(stages: StagesTable) -> tuple[Look, ...]:
    first, final = stages.seeds
    bounds = look_bounds(stages.spending, first / final)
    return tuple(Look(i + 1, len(bounds), n, 1 - _NORMAL.cdf(c))
                 for i, (n, c) in enumerate(zip(stages.seeds, bounds, strict=True)))


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
    # `equivalent` only: the parity margin in the metric's own units (e.g. 0.01 nats),
    # fixed before the run as equivalence testing does, in place of sigmas *
    # sigma_pooled, which moves with the seeds' spread (#604).
    margin: float | None = None

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"{self.name}: unknown rule {self.rule!r}; expected one of {sorted(RULES)}")
        if self.min_delta is not None and self.rule in PARITY_RULES:
            raise ValueError(f"{self.name}: min_delta makes no sense for a parity rule")
        if self.margin is not None:
            if self.rule != "equivalent":
                raise ValueError(f"{self.name}: margin is the parity margin of an `equivalent` criterion")
            if self.margin <= 0 or self.sigmas != 0:
                raise ValueError(f"{self.name}: an absolute margin must be positive and replaces the sigma "
                                 f"bar; write sigmas = 0 (got margin {self.margin}, sigmas {self.sigmas})")
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
    # Two registered looks (#604); None is a single look at the full seed count.
    stages: StagesTable | None = None

    def __post_init__(self):
        unknown = [n for n in (*self.keep_if, *self.kill_if) if n not in self.criteria]
        if unknown:
            raise ValueError(f"keep_if/kill_if name criteria that do not exist: {unknown}")
        if not self.keep_if and not self.kill_if:
            raise ValueError("a spec with no keep_if and no kill_if pre-registers nothing")
        if self.stages is not None:
            self._check_stages(self.stages)

    def _check_stages(self, stages: StagesTable) -> None:
        """Two looks are for a parity claim with an absolute margin (the owner's
        decisions on #604), and the seeds they name are the seeds the spec runs."""
        if not any(self.criteria[n].rule in STAGED_RULES for n in self.keep_if):
            raise ValueError(f"[stages] spends alpha on {', '.join(STAGED_RULES)} only, and no keep_if "
                             f"criterion uses it; a `beats` pair keeps its single look")
        sigma_margins = [c.name for c in self.criteria.values() if c.rule in STAGED_RULES and c.margin is None]
        if sigma_margins:
            raise ValueError(f"[stages]: {', '.join(sigma_margins)} must register an absolute `margin` in the "
                             f"metric's units, fixed before the run")
        execution = self.file.execution if self.file else None
        if execution and execution.seeds and len(execution.seeds) != stages.seeds[-1]:
            raise ValueError(f"[stages] ends at {stages.seeds[-1]} seeds a side but [execution] registers "
                             f"{len(execution.seeds)}")


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
        if self.criterion.margin is not None:
            bar = f"{self.criterion.rule} ±{self.criterion.margin:g}"
        if self.criterion.min_delta is not None:
            bar += f" and |Δ| ≥ {self.criterion.min_delta:g}"
        detail = ", ".join(p.describe() for p in self.points)
        return f"{self.criterion.name} [{bar}, {self.criterion.require}] → {'met' if self.holds else 'NOT met'} ({detail})"


@dataclass(frozen=True)
class Verdict:
    outcome: str
    criteria: tuple[CriterionResult, ...]
    reason: str
    # The look this verdict was judged at; None for a spec with no [stages].
    look: Look | None = None
    # False only at a look 1 that decided nothing: run look 2's seeds and judge again.
    settled: bool = True

    def describe(self) -> str:
        lines = [f"verdict: {self.outcome} — {self.reason}"]
        if self.look is not None:
            lines.append(f"  look {self.look.index} of {self.look.of}: {self.look.seeds} seeds a side, "
                         f"{', '.join(STAGED_RULES)} at one-sided p ≤ {self.look.alpha:.4f}"
                         f"{'' if self.settled else ' — not settled, run the next look'}")
        lines += [f"  {c.describe()}" for c in self.criteria]
        return "\n".join(lines)


def _criterion_result(criterion: Criterion, results: dict[str, dict[str, Arm]],
                      alpha: float = ALPHA) -> CriterionResult:
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
        p = p_value(criterion.rule, treatment, control, criterion.sigmas, criterion.margin)
        holds = RULES[criterion.rule](sig, criterion.sigmas, p, alpha)
        if criterion.min_delta is not None:
            holds = holds and {"beats": delta >= criterion.min_delta,
                               "loses": delta <= -criterion.min_delta}[criterion.rule]
        evaluated.append(PointResult(point, sig, holds, delta, p, n))
    holds = (all if criterion.require == "all" else any)(p.holds for p in evaluated)
    return CriterionResult(criterion, tuple(evaluated), holds)


def evaluate(spec: Spec, results: dict[str, dict[str, Arm]], look: int | None = None) -> Verdict:
    """Apply a spec's pre-registered criteria to recorded per-seed numbers.

    `results` is {sweep point: {arm: [value per seed]}}.

    Kill is checked first and wins ties: a spec that manages to satisfy its keep
    bar while also tripping a kill bar is a spec whose author did not think the
    branches through, and the safe reading of "this fired the kill condition" is
    that the idea is dead.

    A spec with [stages] is judged at `look` (1 or 2), or by default as the design
    runs: look 1, then look 2 only if look 1 did not settle and the seeds reach it;
    see _judge_at_look. Without [stages] there is one look, and
    `look` must be left out.
    """
    if spec.stages is not None:
        if look is not None:
            return _judge_at_look(spec, results, look)
        verdict = _judge_at_look(spec, results, 1)
        reaches_look_2 = _seeds_a_side(spec, results) >= spec.stages.seeds[-1]
        return verdict if verdict.settled or not reaches_look_2 else _judge_at_look(spec, results, 2)
    if look is not None:
        raise ValueError(f"{spec.id}: registers no [stages], so it has one look and none to choose")
    return _judge(spec, results, ALPHA)


def _judge(spec: Spec, results: dict[str, dict[str, Arm]], alpha: float) -> Verdict:
    evaluated = {name: _criterion_result(c, results, alpha) for name, c in spec.criteria.items()}

    fired = [n for n in spec.kill_if if evaluated[n].holds]
    if fired:
        return Verdict(KILL, tuple(evaluated.values()), f"kill criteria fired: {', '.join(fired)}")

    if spec.keep_if:
        unmet = [n for n in spec.keep_if if not evaluated[n].holds]
        if not unmet:
            return Verdict(KEEP, tuple(evaluated.values()), f"all keep criteria met: {', '.join(spec.keep_if)}")
        return Verdict(INCONCLUSIVE, tuple(evaluated.values()), f"keep criteria not met: {', '.join(unmet)}")

    return Verdict(INCONCLUSIVE, tuple(evaluated.values()), "no kill criterion fired and none was pre-registered to keep")


def _seeds_a_side(spec: Spec, results: dict[str, dict[str, Arm]]) -> int:
    """The fewest per-seed values any compared arm holds. Summary arms (a declared
    floor) are constants, not seeds, and do not count; a missing arm is left for
    _criterion_result to name."""
    arms = [results[point][name] for c in spec.criteria.values() for point in (c.points or sorted(results))
            for name in (c.treatment, c.control) if name in results.get(point, {})]
    counts = [len(arm) for arm in arms if not isinstance(arm, Summary)]
    if not counts:
        raise ValueError(f"{spec.id}: [stages] judges per-seed values, and no compared arm has any")
    return min(counts)


def _first_seeds(results: dict[str, dict[str, Arm]], n: int) -> dict[str, dict[str, Arm]]:
    return {point: {name: arm if isinstance(arm, Summary) else list(arm)[:n] for name, arm in arms.items()}
            for point, arms in results.items()}


def _futile(spec: Spec, results: dict[str, dict[str, Arm]], alpha: float) -> list[str]:
    """The keep criteria whose parity the data already rule out: the gap shown
    beyond the margin at one-sided p <= alpha, at every point a `require = any`
    criterion could still pass on, or at any point of a `require = all` one."""
    futile = []
    for name in spec.keep_if:
        c = spec.criteria[name]
        if c.rule not in STAGED_RULES or c.margin is None:
            continue
        ps = [beyond_margin_p(results[p][c.treatment], results[p][c.control], c.margin)
              for p in (c.points or sorted(results))]
        out = [p is not None and p <= alpha for p in ps]
        if (any if c.require == "all" else all)(out):
            futile.append(f"{name} (p {min(p for p in ps if p is not None):.4f})")
    return futile


def _judge_at_look(spec: Spec, results: dict[str, dict[str, Arm]], index: int) -> Verdict:
    """Judge a staged spec at one look, on each arm's first seeds.

    Look 1 stops the pair when it settles: KEEP or KILL at look 1's bar, or the
    parity claim futile (the gap shown beyond the margin, at the same bar). That
    stop is INCONCLUSIVE, with the reason: the claim can no longer be shown, and
    any kill is the spec's own branch to fire. Otherwise look 1 is INCONCLUSIVE and
    not settled, and look 2's seeds decide, at look 2's bar. Futility only ever
    stops a parity claim, so the overall error stays at or under ALPHA.
    """
    assert spec.stages is not None
    planned = looks(spec.stages)
    have = _seeds_a_side(spec, results)
    if index not in range(1, len(planned) + 1):
        raise ValueError(f"{spec.id}: registers looks 1 to {len(planned)}, not look {index}")
    at = planned[index - 1]
    if have < at.seeds:
        raise ValueError(f"{spec.id}: look {index} judges {at.seeds} seeds a side, and an arm has {have}, "
                         f"short of it")
    results = _first_seeds(results, at.seeds)
    verdict = replace(_judge(spec, results, at.alpha), look=at)
    if at.index == at.of or verdict.outcome in (KEEP, KILL):
        return verdict
    futile = _futile(spec, results, at.alpha) if spec.stages.futility else []
    if futile:
        return replace(verdict, reason=f"futile at look {at.index}: {', '.join(futile)} shows the gap beyond "
                                       f"its margin, so parity can no longer be shown")
    return replace(verdict, settled=False, reason=f"look {at.index} settles nothing ({verdict.reason}); "
                                                  f"run look {at.index + 1}'s seeds")


def load_spec(path) -> Spec:
    """Read a spec TOML through its typed tables (instruments/spec_file.py): a
    malformed spec must fail before a run burns GPU hours, not after."""
    file = SpecFile.load(path)
    criteria = {name: Criterion(name=name, rule=c.rule, treatment=c.treatment, control=c.control,
                                sigmas=c.sigmas, points=c.points, require=c.require, min_delta=c.min_delta,
                                margin=c.margin)
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
        stages=file.stages,
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
