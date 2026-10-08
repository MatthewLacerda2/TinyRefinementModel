"""What a matched pair did, beside the verdict the referee reached (#564).

    python -m instruments.pair_report experiments/recipe/specs/494-postnorm-long-pair.toml
    python -m instruments.pair_report <spec> --runs runs/pairs-494-long

The referee judges one pre-registered number. This prints the rest of the story
in plain numbers, so a disagreement between the bar and the endpoint is on screen
without anyone typing a snippet (#361 was KEEP by the bar and worse at the end):

- held-out CE at fixed fractions of the run, mean over seeds, with Δ against the control;
- `final_val_ce` per seed, mean, σ, and whether every treatment seed lands on the
  same side of every control seed;
- the peak and final `act_max` / `branch_max` per arm;
- seconds per optimizer step per arm.

It reads the sweep's journal (`<runs>/experiments/<spec id>/results.jsonl`) and,
for each arm and seed, the `metrics.csv` in the `run_dir` its RESULT line named. A
journal row from before #564 names no run dir: its curves are left out and the
report says so. `--runs` points at a pair folder moved out of `runs/`; a run dir is
looked up by its name under it.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from instruments._common import REPO_ROOT
from instruments.curves import ValCurve
from instruments.verdict import load_spec

REPORTS = {
    "held-out CE": ("measured", "metrics.csv val_ce at its val_step, mean over seeds"),
    "final_val_ce": ("measured", "the harness's RESULT line, per seed"),
    "act_max / branch_max": ("sampled", "the logged rows only: peak over seeds and rows, mean of the last"),
    "s/opt step": ("estimated", "median wall_clock delta between log rows, mean over seeds"),
}

# Where on the run the CE table reads, as fractions of the last probed step.
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
# Telemetry columns whose peak and end tell whether an arm ran near the f16 range.
MARGIN_COLUMNS = ("act_max", "branch_max")


@dataclass
class Curve:
    """One run's metrics.csv, reduced to what the report reads."""
    val: dict[int, float] = field(default_factory=dict)          # val_step -> held-out CE
    margins: dict[str, list[float]] = field(default_factory=dict)  # column -> readings in order
    seconds_per_step: float | None = None


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pace(stamps):
    """Median seconds per opt step between consecutive log rows. The median, so a
    pause and resume (a gap of hours between two rows) does not count as training."""
    rates = [(t1 - t0) / (s1 - s0) for (s0, t0), (s1, t1) in zip(stamps, stamps[1:], strict=False) if s1 > s0]
    return statistics.median(rates) if rates else None


def read_curve(metrics: pathlib.Path) -> Curve:
    curve, stamps = Curve(), []
    with metrics.open() as f:
        for row in csv.DictReader(f):
            step = _number(row.get("step"))
            for column in MARGIN_COLUMNS:
                if (value := _number(row.get(column))) is not None:
                    curve.margins.setdefault(column, []).append(value)
            stamp = row.get("wall_clock")
            if step is not None and stamp:
                stamps.append((step, _timestamp(stamp)))
    curve.seconds_per_step = _pace(stamps)
    val = ValCurve.read(metrics)
    curve.val = {s: ce for s, ce in zip(val.steps, val.ces, strict=True) if math.isfinite(ce)}
    return curve


def _timestamp(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def latest_rows(rows):
    """The last journal row per (arm, seed, point): a `--force` re-read appends, it
    does not replace."""
    latest = {}
    for row in rows:
        latest[(row["arm"], row["seed"], row["point"])] = row
    return list(latest.values())


def find_run(row, runs: pathlib.Path) -> pathlib.Path | None:
    if "run_dir" not in row:
        return None
    recorded = pathlib.Path(row["run_dir"])
    for candidate in (recorded, runs / recorded.name):
        if (candidate / "metrics.csv").exists():
            return candidate
    return None


def control_arm(spec) -> str | None:
    """The one arm whose role is "control"; None for a spec that is not a pair."""
    arms = spec.file.arms if spec.file is not None else {}
    controls = [name for name, arm in arms.items() if arm.role == "control"]
    return controls[0] if len(controls) == 1 else None


def _mean(values):
    return sum(values) / len(values) if values else None


def _fmt(value, spec=".4f"):
    return "—" if value is None else format(value, spec)


def ce_table(curves: dict[str, list[Curve]], control: str) -> list[str]:
    """Held-out CE at FRACTIONS of the run, read at the latest probe at or before each
    fraction that every seed of every arm has."""
    common = set.intersection(*(set(c.val) for cs in curves.values() for c in cs))
    if not common:
        return ["  no probe step shared by every run"]
    steps = sorted(common)
    picked = sorted({max(s for s in steps if s <= f * steps[-1]) for f in FRACTIONS if f * steps[-1] >= steps[0]})
    arms = [control, *(a for a in curves if a != control)]
    lines = ["  step  " + "".join(f"{a:>12}" for a in arms) + "".join(f"{'Δ ' + a:>14}" for a in arms[1:])]
    for step in picked:
        means = {a: _mean([c.val[step] for c in curves[a]]) for a in arms}
        lines.append(f"  {step:>4}  " + "".join(f"{means[a]:>12.4f}" for a in arms)
                     + "".join(f"{means[a] - means[control]:>+14.4f}" for a in arms[1:]))
    return lines


def seed_table(finals: dict[str, dict[int, float]], control: str) -> list[str]:
    """final_val_ce per seed, the spread, and whether the arms separate completely."""
    seeds = sorted({s for by_seed in finals.values() for s in by_seed})
    lines = ["  arm         " + "".join(f"{'s' + str(s):>9}" for s in seeds) + f"{'mean':>9}{'σ':>8}"]
    for arm, by_seed in finals.items():
        values = list(by_seed.values())
        sigma = statistics.stdev(values) if len(values) > 1 else None
        lines.append(f"  {arm:<12}" + "".join(f"{_fmt(by_seed.get(s)):>9}" for s in seeds)
                     + f"{_fmt(_mean(values)):>9}{_fmt(sigma):>8}")
    base = list(finals.get(control, {}).values())
    for arm, by_seed in finals.items():
        if arm == control or not base or not by_seed:
            continue
        values = list(by_seed.values())
        side = ("every seed below every control seed" if max(values) < min(base)
                else "every seed above every control seed" if min(values) > max(base)
                else "the seeds overlap the control's")
        lines.append(f"  {arm}: {side}")
    return lines


def margin_table(curves: dict[str, list[Curve]]) -> list[str]:
    lines = ["  arm         " + "".join(f"{c + ' peak':>18}{c + ' end':>16}" for c in MARGIN_COLUMNS)]
    for arm, cs in curves.items():
        cells = []
        for column in MARGIN_COLUMNS:
            series = [c.margins[column] for c in cs if c.margins.get(column)]
            cells.append(f"{_fmt(max((max(s) for s in series), default=None), '.1f'):>18}"
                         f"{_fmt(_mean([s[-1] for s in series]), '.1f'):>16}")
        lines.append(f"  {arm:<12}" + "".join(cells))
    return lines


def pace_table(curves: dict[str, list[Curve]]) -> list[str]:
    return [f"  {arm:<12}{_fmt(_mean([c.seconds_per_step for c in cs if c.seconds_per_step]), '.1f'):>8} s/opt step"
            for arm, cs in curves.items()]


def render(spec, rows, runs: pathlib.Path) -> str:
    """The report, or "" for a spec with no single control arm to compare against."""
    control = control_arm(spec)
    if control is None:
        return ""
    rows = latest_rows(rows)
    finals, curves, missing = {}, {}, []
    for row in sorted(rows, key=lambda r: (r["arm"] != control, r["arm"], r["seed"])):
        if (final := _number(row.get("final_val_ce"))) is not None:
            finals.setdefault(row["arm"], {})[row["seed"]] = final
        run = find_run(row, runs)
        if run is None:
            missing.append(f"{row['arm']} s{row['seed']}")
        else:
            curves.setdefault(row["arm"], []).append(read_curve(run / "metrics.csv"))

    out = [f"== {spec.id}: pair report (control: {control}) =="]
    if finals:
        out += ["", "final_val_ce per seed", *seed_table(finals, control)]
    if curves.get(control) and len(curves) > 1:
        out += ["", "held-out CE at fractions of the run (mean over seeds)", *ce_table(curves, control)]
    if curves:
        out += ["", "activation margins (peak over seeds, mean end)", *margin_table(curves),
                "", "pace (median between log rows, mean over seeds)", *pace_table(curves)]
    if missing:
        out += ["", f"no curves for {', '.join(missing)}: the journal names no run dir that exists "
                    f"(rows from before #564 never did; --runs points at a moved pair folder)"]
    return "\n".join(out)


def main(argv=None) -> int:
    from instruments.experiment import read_journal

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=pathlib.Path)
    ap.add_argument("--runs", type=pathlib.Path, default=REPO_ROOT / "runs",
                    help="the folder holding experiments/<spec id>/results.jsonl and the run dirs")
    args = ap.parse_args(argv)
    spec = load_spec(args.spec)
    journal = args.runs / "experiments" / spec.id / "results.jsonl"
    rows = read_journal(journal)
    if not rows:
        raise SystemExit(f"no journal rows at {journal}")
    report = render(spec, rows, args.runs)
    if not report:
        raise SystemExit(f"{spec.id}: no single arm with role = \"control\" to compare against")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
