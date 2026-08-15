"""Read one training run's recorded metrics — once, and properly.

Every consumer of `runs/<run>/metrics.csv` used to re-parse it inline, and each
re-parse made the same mistake: a blank cell read as `0.0`. A blank does not
mean zero. It means this architecture does not measure that quantity (#105) —
`avg_forget_cost`, `diversity_loss` and `temporal_drift` are empty in every
refiner run because the refiner has no forget gate and no slots. Reading them
as zero invents a measurement, and a plotter then draws a confident flat line
through data that was never collected.

So here: a blank is `None`, `has(name)` says whether a column holds anything at
all (the test a caller uses to *omit* a panel instead of drawing zeros), and
`column(name)` hands back only the rows that actually carry a value.

The loader is also the one place that knows the shape of the artifact:

  * **replayed rows** — CSVs written before resume-trimming existed contain
    non-monotonic step ranges (a resume restores to the last *best* step and
    re-logs forward from there). Only the first occurrence of each advancing
    step is kept.
  * **torn rows** — the last row of a live run can be half-written. A row whose
    `step` does not parse is dropped, not guessed at.
  * **missing metadata** — `run_metadata.json` may not exist (an old run, or a
    dir assembled by hand). That is `{}`, not a crash.

    from instruments.runlog import load
    log = load()                      # newest run under runs/
    steps, ce = log.column("ce")
    if log.has("val_ce"): ...
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field

from trm.config import TOKENS_PER_OPT_STEP

METRICS_FILENAME = "metrics.csv"
METADATA_FILENAME = "run_metadata.json"

# DictReader parks fields beyond the header under this key; naming it keeps a
# widened row from inventing a column called `None`.
_OVERFLOW = "__extra__"


def _parse_float(raw):
    """A cell's value, or None if the cell is empty or unreadable.

    Non-finite values (nan/inf) are *kept*: the trainer warns about them and
    they mean something went wrong, which is exactly what a reader should see.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_step(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        # A torn final row ("27" of "2750,3.71,...") or a corrupted line. There
        # is nothing to recover — a guessed step is worse than a dropped row.
        return None


@dataclass
class RunLog:
    """One run's metrics, parsed. Blanks are absent, never zero."""

    run_id: str
    metrics: list[dict]      # rows; every CSV column, value float or None
    metadata: dict           # run_metadata.json contents, {} if absent

    # Provenance and what the parse had to throw away. Everything below has a
    # default so `RunLog(run_id, metrics, metadata)` stays constructible.
    fields: list[str] = field(default_factory=list)   # CSV header, in order
    csv_path: str = ""
    run_dir: str = ""
    replayed_rows: int = 0   # non-monotonic rows dropped (resume replay)
    torn_rows: int = 0       # rows with no readable step

    def column(self, name):
        """(steps, values) for one column, blank cells dropped.

        An unknown or entirely-blank column is two empty lists, not an error:
        the arch-optional columns are the normal case, not a mistake.
        """
        steps, values = [], []
        for row in self.metrics:
            value = row.get(name)
            if value is None:
                continue
            steps.append(row["step"])
            values.append(value)
        return steps, values

    def has(self, name):
        """True iff any row carries a value for this column."""
        return any(row.get(name) is not None for row in self.metrics)

    def is_constant(self, name, value=None):
        """True when a column has values and every one of them is identical
        (and equal to `value`, if given).

        The blank-cell convention only arrived with #105. Runs before it wrote a
        literal `0.0000` for quantities their architecture did not measure, so
        `has()` is True and the column is still not a measurement — it is the
        flat-zero panel the old plotter drew three of. `is_constant(name, 0.0)`
        is how a caller tells that apart from a real, genuinely flat signal
        (which is worth seeing, and rare).
        """
        _, values = self.column(name)
        if not values:
            return False
        first = values[0]
        if value is not None and first != value:
            return False
        return all(v == first for v in values)

    @property
    def last_step(self):
        """The last optimizer step logged (0 for an empty or header-only CSV)."""
        return self.metrics[-1]["step"] if self.metrics else 0

    @property
    def tokens(self):
        """Tokens consumed, at the *current* config's tokens-per-opt-step.

        A run whose recipe differed (BATCH_SIZE x ACCUMULATION_STEPS) is
        mis-scaled by this; `run_metadata.json` records what the run actually
        used, and report.py cross-checks it.
        """
        return self.last_step * TOKENS_PER_OPT_STEP

    @property
    def wall_seconds(self):
        """Total recorded wall-clock across the run's sessions, or None.

        The tracker writes one section per launch and fills in its duration as
        the run proceeds; a session still in flight before its first update has
        `duration_seconds: null` and is skipped.
        """
        durations = [s.get("duration_seconds") for s in self.metadata.get("sections", [])]
        known = [float(d) for d in durations if isinstance(d, (int, float))]
        return sum(known) if known else None


def _discover_latest_csv(runs_root="runs"):
    """The newest run directory that actually has a metrics.csv."""
    for run_dir in sorted(glob.glob(os.path.join(runs_root, "run_*")), reverse=True):
        candidate = os.path.join(run_dir, METRICS_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"no run under {runs_root!r} has a {METRICS_FILENAME} — train first, "
        f"or pass a path explicitly."
    )


def _read_metadata(run_dir):
    path = os.path.join(run_dir, METADATA_FILENAME)
    try:
        with open(path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        # A run assembled by hand, or an old one from before the tracker. The
        # metrics are still perfectly readable without it.
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _read_rows(csv_path):
    """Parsed rows + the header, with torn and replayed rows counted out."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, restkey=_OVERFLOW, restval="")
        fields = list(reader.fieldnames or [])
        if fields and "step" not in fields:
            raise ValueError(
                f"{csv_path} has no 'step' column (header: {fields}) — this is not "
                f"a training metrics log."
            )
        rows, torn = [], 0
        for raw in reader:
            raw.pop(_OVERFLOW, None)
            step = _parse_step(raw.get("step"))
            if step is None:
                torn += 1
                continue
            row = {"step": step}
            for name in fields:
                if name != "step":
                    row[name] = _parse_float(raw.get(name))
            rows.append(row)

    # Keep only advancing steps: a pre-trimming resume re-logs a range it
    # already wrote, and the replayed copy is the stale one.
    monotonic, previous = [], -1
    for row in rows:
        if row["step"] > previous:
            monotonic.append(row)
            previous = row["step"]
    return fields, monotonic, len(rows) - len(monotonic), torn


def load(path=None):
    """Load a run: a metrics.csv path, a run directory, or None for the latest.

    Raises FileNotFoundError if there is nothing to read — a caller that wants
    to survive that should catch it and say so, rather than plot an empty run.
    """
    if path is None:
        csv_path = _discover_latest_csv()
    else:
        path = os.path.abspath(path)
        csv_path = os.path.join(path, METRICS_FILENAME) if os.path.isdir(path) else path
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"no metrics CSV at {csv_path}")

    csv_path = os.path.abspath(csv_path)
    run_dir = os.path.dirname(csv_path)
    fields, rows, replayed, torn = _read_rows(csv_path)
    return RunLog(
        run_id=os.path.basename(run_dir),
        metrics=rows,
        metadata=_read_metadata(run_dir),
        fields=fields,
        csv_path=csv_path,
        run_dir=run_dir,
        replayed_rows=replayed,
        torn_rows=torn,
    )
