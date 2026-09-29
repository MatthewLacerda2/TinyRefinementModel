"""Does the optimal learning rate transfer across width? The #483 judge.

    python -m experiments.recipe.lr_transfer experiments/recipe/specs/483-<id>.toml

The referee (`instruments/verdict.py`) compares two arms; this question is about
where an optimum sits on a grid, at several widths, so it gets its own rule. The
rule is committed here before any sweep exists, so the bar cannot move once the
numbers are in (CLAUDE.md rule 3):

- Arms are named `w<width>_lr<peak lr>` (`arm_name`: the LR as an integer mantissa
  and exponent, `w576_lr15e-5`, since TOML reads a dot in a key as a nested table),
  and their metric is tokens-to-target
  (lower is better), per seed, under `[results.run]` of the sweep's spec.
- **A width's optimal set** is every LR on its grid that cannot be told from that
  width's best (lowest mean) LR: mean gap within `SIGMAS` x sigma_pooled, the
  repo's noise-floor convention (verdict.pooled_sigma). One LR alone is the optimum
  only when the rest are clearly worse.
- **Transfer holds at a narrow width** when its optimal set and the full width's
  share an LR, or hold two LRs one grid step (a factor `GRID_FACTOR`) apart:
  #483's "within one grid step of the optimum at full width", with the noise in it.
- **No optimum located is not a pass** (rule 1: two failures agree trivially). A
  width whose best LR sits on the grid's edge, or whose optimal set covers more than
  `MAX_OPTIMAL_SPAN` grid points, did not locate its optimum; the verdict is then
  INCONCLUSIVE, however the sets happen to overlap.

Verdict: HOLDS when every narrow width transfers, FAILS when any does not, and
INCONCLUSIVE when a width located no optimum. What each means for the recipe is
#483's (and the spec's) to say, not this module's.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

from instruments import results
from instruments.verdict import Arm, load_recorded_results, mean_sigma, pooled_sigma

SIGMAS = 2.0
GRID_FACTOR = 2.0
MAX_OPTIMAL_SPAN = 3
HOLDS, FAILS, INCONCLUSIVE = "HOLDS", "FAILS", "INCONCLUSIVE"

_ARM = re.compile(r"^w(?P<width>\d+)_lr(?P<lr>[0-9.eE+-]+)$")


@dataclass(frozen=True)
class Optimum:
    """One width's optimum: the best LR, the set it cannot be told from, and
    whether the grid located it at all (None when it did, else why not)."""

    width: int
    best: float
    optimal: tuple[float, ...]
    unlocated: str | None


def grid_steps(a: float, b: float) -> float:
    """How many grid steps apart two LRs are."""
    return abs(math.log(a / b, GRID_FACTOR))


def optimum(width: int, arms: dict[float, Arm]) -> Optimum:
    grid = sorted(arms)
    best = min(grid, key=lambda lr: mean_sigma(arms[lr])[0])
    best_mean = mean_sigma(arms[best])[0]
    optimal = tuple(lr for lr in grid
                    if mean_sigma(arms[lr])[0] - best_mean <= SIGMAS * pooled_sigma(arms[lr], arms[best]))
    unlocated = None
    if best in (grid[0], grid[-1]):
        unlocated = f"best LR {best:g} is on the grid's edge; the optimum may lie outside it"
    elif round(grid_steps(optimal[0], optimal[-1])) + 1 > MAX_OPTIMAL_SPAN:
        unlocated = (f"{len(optimal)} LRs from {optimal[0]:g} to {optimal[-1]:g} cannot be told apart "
                     f"at {SIGMAS:g} sigma; the sweep is too noisy to place the optimum")
    return Optimum(width, best, optimal, unlocated)


def transfers(narrow: Optimum, full: Optimum) -> bool:
    """Some LR the narrow width cannot tell from its best is within one grid step
    of one the full width cannot tell from its best."""
    return any(grid_steps(a, b) <= 1 + 1e-9 for a in narrow.optimal for b in full.optimal)


def judge(grid: dict[int, dict[float, Arm]]) -> tuple[str, list[Optimum], str]:
    """(verdict, the optimum per width narrowest first, why). The widest width
    is the full-width model every narrower one is judged against."""
    optima = [optimum(width, grid[width]) for width in sorted(grid)]
    if len(optima) < 2:
        raise ValueError("transfer needs at least two widths")
    unlocated = [o for o in optima if o.unlocated]
    if unlocated:
        return INCONCLUSIVE, optima, "; ".join(f"w{o.width}: {o.unlocated}" for o in unlocated)
    full = optima[-1]
    failed = [o for o in optima[:-1] if not transfers(o, full)]
    if failed:
        return FAILS, optima, ", ".join(
            f"w{o.width} optimum {o.best:g} vs w{full.width} {full.best:g}" for o in failed)
    return HOLDS, optima, f"every narrow optimum is within one grid step of w{full.width}'s {full.best:g}"


def arm_name(width: int, lr: float) -> str:
    """`w<width>_lr<m>e<k>` with an integer mantissa: no dot, so a TOML key."""
    for exponent in range(0, -16, -1):
        mantissa = lr / 10**exponent
        if abs(mantissa - round(mantissa)) < 1e-9 * mantissa:
            return f"w{width}_lr{round(mantissa)}e{exponent}"
    raise ValueError(f"lr {lr!r} has no short decimal form")


def grid_from_arms(arms: dict[str, Arm]) -> dict[int, dict[float, Arm]]:
    """{width: {lr: per-seed values}} from arms named w<width>_lr<lr>. A name that
    does not parse is refused: silently dropping it would judge a smaller grid."""
    grid: dict[int, dict[float, Arm]] = {}
    for name, values in arms.items():
        match = _ARM.match(name)
        if not match:
            raise ValueError(f"arm {name!r} is not w<width>_lr<lr>")
        grid.setdefault(int(match["width"]), {})[float(match["lr"])] = values
    return grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", help="the sweep's spec, with [results.<point>] recorded")
    ap.add_argument("--point", default="run", help="the sweep point to judge (tokens_to_ce emits 'run')")
    args = ap.parse_args(argv)

    verdict, optima, why = judge(grid_from_arms(load_recorded_results(args.spec)[args.point]))
    for o in optima:
        print(f"w{o.width}: best {o.best:g}, cannot be told from {', '.join(f'{lr:g}' for lr in o.optimal)}"
              + (f"  [{o.unlocated}]" if o.unlocated else ""))
    print(f"{verdict}: {why}")
    results.emit("transfer", holds=float(verdict == HOLDS), fails=float(verdict == FAILS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
