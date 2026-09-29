"""The base run's stall rule, read from its spec's [stall] table (#468).

    python -m instruments.stall --spec experiments/base/specs/001-plain-base.toml --run runs/run_x
    python -m instruments.stall --spec ... --run runs/run_x --floor

The default prints the rule's reading now: each quantity's held-out CE gain over the
last `windows` windows of `window_hours`, against its bar, and STALLED when every
window of every quantity gains less than its bar. `--floor` prints what a bar has to
clear, measured from a finished run's own log: reading-to-reading noise and window
gains in the first and last third after the ramp, for every quantity the rule could
read. That is how 001's 0.01 was set from the champion (the spec's notes).

What `reads` can be — the three options of #468:

  web           val_ce, the fineweb-edu probe alone (001's registered rule).
  mixture       one number: the probes in `sources`, weighted by the run's own end
                mix (its last `mix` cell), renormalized over those sources.
  every-source  each probe in `sources` against its own bar; STALLED only when all
                of them are. `min_gain` is then a table, one bar per source.

The rule starts where the mixture ramp ends: the first row from which the run's `mix`
cell never changes again. A window is the drop in CE from the last reading at or
before its start to the last reading at or before its end, in wall-clock hours; a
window that reaches back before the ramp's end is not read.

Probes differ in footing: fineweb-edu reads a fixed skip, the others their corpus's
tail (#363). A gain is a difference, so a level offset cancels; the noise does not,
which is why every bar is measured per quantity.
"""

from __future__ import annotations

import argparse
import datetime
import statistics
import tomllib
from dataclasses import dataclass

from instruments import runlog

REPORTS = {
    "window gain": ("measured", "held-out CE at a window's start minus at its end, from metrics.csv readings"),
    "reading noise": ("measured", "stdev of successive held-out readings' differences, per third of the run"),
}

MODES = ("web", "mixture", "every-source")
WEB = "fineweb-edu"   # val_ce's corpus; every other probe is a val_by_source cell
# Probes whose held-out rows reappear near-verbatim in training (#485): their drops
# are recall events, not learning. The rule may read one; the report always says so.
CONTAMINATED = {"codeparrot": "#485"}

STALLED, LEARNING, NOT_YET = "STALLED", "LEARNING", "NOT YET"


@dataclass(frozen=True)
class Rule:
    reads: str
    window_hours: float
    windows: int
    min_gain: float | dict[str, float]
    decay_opt_steps: int
    sources: tuple[str, ...] = (WEB,)

    def bar(self, quantity: str) -> float:
        return self.min_gain[quantity] if isinstance(self.min_gain, dict) else self.min_gain


def load_rule(spec_path) -> Rule:
    """The spec's [stall] table, checked. A new quantity carries no default bar: the
    one registered for web CE was measured on web CE and does not transfer."""
    with open(spec_path, "rb") as f:
        table = tomllib.load(f).get("stall")
    if not table:
        raise ValueError(f"{spec_path}: no [stall] table")
    reads = table["reads"]
    if reads not in MODES:
        raise ValueError(f"{spec_path}: [stall] reads = {reads!r}, not one of {MODES}")
    sources = (WEB,) if reads == "web" else tuple(table["sources"])
    min_gain = table["min_gain"]
    if reads == "every-source":
        if not isinstance(min_gain, dict) or set(min_gain) != set(sources):
            raise ValueError(f"{spec_path}: every-source needs min_gain as a table with one bar per source "
                             f"in {list(sources)}")
    elif not isinstance(min_gain, (int, float)):
        raise ValueError(f"{spec_path}: {reads} reads one quantity, so min_gain is one number")
    return Rule(reads=reads, window_hours=float(table["window_hours"]), windows=int(table["windows"]),
                min_gain=min_gain, decay_opt_steps=int(table["decay_opt_steps"]), sources=sources)


def parse_mix(label) -> dict[str, float]:
    """`fineweb-edu=0.350 codeparrot=0.400` (the trainer's mixture_label) as a dict."""
    return {name: float(w) for name, _, w in (part.partition("=") for part in (label or "").split())}


def ramp_end_step(log) -> int | None:
    """The step from which the run's `mix` never changes again, or None if it logged none."""
    mixed = [row for row in log.metrics if row.get("mix")]
    if not mixed:
        return None
    end = mixed[-1]["mix"]
    first = len(mixed) - 1
    while first > 0 and mixed[first - 1]["mix"] == end:
        first -= 1
    return mixed[first]["step"]


def readings(log, sources) -> dict[str, list[tuple[datetime.datetime, float]]]:
    """{source: [(wall clock, CE)]} for every probe named, after the ramp. fineweb-edu
    is val_ce; the others are cells of val_by_source."""
    start = ramp_end_step(log) or 0
    series = {s: [] for s in sources}
    for row in log.metrics:
        when = row.get("wall_clock")
        if when is None or row["step"] < start:
            continue
        cells = dict(row.get("val_by_source") or {})
        if row.get("val_ce") is not None:
            cells[WEB] = row["val_ce"]
        for s in sources:
            if s in cells:
                series[s].append((when, cells[s]))
    return series


def mixture_series(log, sources) -> list[tuple[datetime.datetime, float]]:
    """The end-mix-weighted CE at every row that read all `sources`."""
    mix = parse_mix(next((row["mix"] for row in reversed(log.metrics) if row.get("mix")), None))
    missing = [s for s in sources if s not in mix]
    if missing:
        raise ValueError(f"the run's end mix {mix} has no weight for {missing}")
    total = sum(mix[s] for s in sources)
    per_source = readings(log, sources)
    by_time = {}
    for s in sources:
        for when, ce in per_source[s]:
            by_time.setdefault(when, {})[s] = ce
    return [(when, sum(mix[s] / total * cells[s] for s in sources))
            for when, cells in sorted(by_time.items()) if len(cells) == len(sources)]


def quantities(rule, log) -> dict[str, list[tuple[datetime.datetime, float]]]:
    if rule.reads == "mixture":
        return {"mixture": mixture_series(log, rule.sources)}
    return readings(log, rule.sources)


def value_at(series, when):
    """The last reading at or before `when`, or None if the series starts after it."""
    before = [ce for t, ce in series if t <= when]
    return before[-1] if before else None


def window_gain(series, end, hours):
    start_ce, end_ce = value_at(series, end - datetime.timedelta(hours=hours)), value_at(series, end)
    return None if start_ce is None or end_ce is None else start_ce - end_ce


def recent_gains(series, rule) -> list[float] | None:
    """The gains of the last `windows` windows, newest first; None until all exist."""
    if not series:
        return None
    last = series[-1][0]
    gains = [window_gain(series, last - datetime.timedelta(hours=k * rule.window_hours), rule.window_hours)
             for k in range(rule.windows)]
    return None if None in gains else gains


def check(rule, log) -> tuple[str, dict[str, list[float] | None]]:
    """STALLED when every quantity's every recent window gains under its bar."""
    gains = {q: recent_gains(series, rule) for q, series in quantities(rule, log).items()}
    if any(g is None for g in gains.values()):
        return NOT_YET, gains
    flat = all(g < rule.bar(q) for q, gs in gains.items() for g in gs)
    return (STALLED if flat else LEARNING), gains


def floor_row(series, hours):
    """(noise, gain median, min, max) per third of the post-ramp readings — the first
    (still learning) and the last (flat, if the run flattened)."""
    third = len(series) // 3
    out = {}
    for name, part in (("first third", series[:third]), ("last third", series[-third:] if third else [])):
        steps = [b - a for (_, a), (_, b) in zip(part, part[1:])]
        gains = [g for g in (window_gain(series, t, hours) for t, _ in part) if g is not None]
        out[name] = (statistics.stdev(steps) if len(steps) > 1 else None,
                     statistics.median(gains) if gains else None,
                     min(gains, default=None), max(gains, default=None))
    return out


def contamination_notes(sources) -> list[str]:
    return [f"note: the {s} probe is contaminated ({issue}): its drops track near-duplicates "
            f"passing in training, not only learning" for s, issue in CONTAMINATED.items() if s in sources]


def _fmt(x):
    return "   —   " if x is None else f"{x:+.4f}"


def report_floor(rule, log) -> list[str]:
    probes = sorted({WEB} | {s for row in log.metrics for s in (row.get("val_by_source") or {})})
    clean = [s for s in probes if s not in CONTAMINATED]
    candidates = {s: readings(log, [s])[s] for s in probes}
    candidates[f"mixture({'+'.join(probes)})"] = mixture_series(log, probes)
    if clean != probes:
        candidates[f"mixture({'+'.join(clean)})"] = mixture_series(log, clean)
    lines = [f"{log.run_id}: after the ramp (step {ramp_end_step(log)}), windows of {rule.window_hours:g}h",
             f"{'quantity':<44} {'third':<12} {'noise':>8} {'gain med':>8} {'min':>8} {'max':>8}"]
    for name, series in candidates.items():
        for third, (noise, med, lo, hi) in floor_row(series, rule.window_hours).items():
            lines.append(f"{name:<44} {third:<12} {_fmt(noise)} {_fmt(med)} {_fmt(lo)} {_fmt(hi)}")
    return lines + contamination_notes(probes)


def report_check(rule, log) -> list[str]:
    outcome, gains = check(rule, log)
    lines = [f"{log.run_id}: stall rule reads {rule.reads} over {rule.windows} x {rule.window_hours:g}h "
             f"after the ramp (step {ramp_end_step(log)})"]
    for q, gs in gains.items():
        shown = "not enough post-ramp readings" if gs is None else ", ".join(_fmt(g) for g in gs)
        lines.append(f"  {q:<14} gains (newest first) {shown}   bar {rule.bar(q)}")
    lines += contamination_notes(rule.sources)
    action = f": start the {rule.decay_opt_steps:,}-opt-step decay" if outcome == STALLED else ""
    return lines + [f"{outcome}{action}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="a base-run spec with a [stall] table")
    ap.add_argument("--run", default=None, help="the run directory (default: the newest under runs/)")
    ap.add_argument("--floor", action="store_true", help="print the noise and gains a bar must clear")
    args = ap.parse_args(argv)
    rule, log = load_rule(args.spec), runlog.load(args.run)
    print("\n".join(report_floor(rule, log) if args.floor else report_check(rule, log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
