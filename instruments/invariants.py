"""Metrics whose correct range is fixed by construction, used as integrity checks.

Some logged quantities cannot leave a range, no matter what the model, the data
mixture or the schedule does: a fraction lies in [0, 1], a norm is non-negative,
an entropy over V outcomes cannot exceed ln(V). A row outside one is not a
measurement of anything; it is proof that the row's accumulator broke (#194, where
a resume wrote a row whose every value was scaled by the same wrong count).

A violated invariant condemns the WHOLE ROW, not just the offending column. The
accumulator is shared, so if one metric on the row is wrong the rest are too.

Deliberately NOT an invariant: an upper bound on cross-entropy. It is tempting
to bound CE by ln(VOCAB_SIZE) = 10.83 on the grounds that no model should be
worse than uniform, and that is wrong — a freshly initialised model can be worse
than uniform, and this project has logged 11.08 at step 5. An invariant that
fires on a healthy run is worse than no invariant, because it teaches people to
ignore it.
"""

import math

from instruments._common import F16_MAX  # the ceiling act_max is measured against
from trm.config import VOCAB_SIZE

# (column, low, high, why) for every quantity with a known-correct range.
INVARIANTS = (
    ("zero_frac_dense_max", 0.0, 1.0, "it is a fraction"),
    ("applied_zero_frac_dense_max", 0.0, 1.0, "it is a fraction"),
    ("applied_grad_norm", 0.0, float("inf"), "a norm is never negative"),
    # Entropy IS bounded above by the uniform distribution's, unlike CE:
    # entropy is a property of the model's own output distribution, and no
    # distribution over V outcomes has entropy above ln(V).
    ("out_entropy", 0.0, math.log(VOCAB_SIZE),
     f"entropy of a distribution over {VOCAB_SIZE:,} outcomes cannot exceed "
     f"ln(V) = {math.log(VOCAB_SIZE):.2f}"),
    ("ce", 0.0, math.inf, "cross-entropy is non-negative"),
    ("seg1_ce", 0.0, math.inf, "cross-entropy is non-negative"),
    ("val_ce", 0.0, math.inf, "cross-entropy is non-negative"),
    ("max_abs_logit", 0.0, math.inf, "it is an absolute value"),
    # Not a law of arithmetic like the others — a MARGIN: half the f16 ceiling,
    # so the row goes suspect while there is still a run to save (#229, #235).
    ("act_max", 0.0, F16_MAX / 2,
     f"peak activation past half of f16's {F16_MAX:,.0f} ceiling leaves under "
     f"2x headroom; the 4B run ended at 65,120 and that is what overflow (#229) "
     f"looks like on the way in"),
    ("grad_norm_avg", 0.0, math.inf, "it is a norm"),
)


def row_violations(row):
    """Every invariant this row breaks, as human-readable strings. Empty if clean.

    A blank cell is not a violation — it is a column the run did not log (#105),
    the normal case. Non-finite values are, since NaN passes every comparison.
    """
    broken = []
    for column, low, high, why in INVARIANTS:
        value = row.get(column)
        if value is None:
            continue
        if not math.isfinite(value):
            broken.append(f"{column} is {value} (not finite) — {why}")
        elif value < low or value > high:
            broken.append(f"{column} {value:.4f} outside [{low:.2f}, "
                          f"{'inf' if high == math.inf else f'{high:.2f}'}] — {why}")
    return broken


def suspect_rows(log):
    """{step: [reasons]} for every row that breaks an invariant.

    The step is the key because the verdict applies to the whole row: the
    accumulator that produced the impossible value produced the rest of the row
    too, so nothing on it can be trusted.
    """
    suspect = {}
    for row in log.metrics:
        broken = row_violations(row)
        if broken:
            suspect[row["step"]] = broken
    return suspect


def clean_column(log, name, suspect=None):
    """`log.column(name)` with suspect rows removed.

    The drop-in replacement for RunLog.column wherever a number is going to be
    reported, plotted, or reduced to a best/mean. Callers that genuinely want the
    raw series (a diagnostic view of the anomaly itself) should keep using
    log.column.
    """
    bad = suspect_rows(log) if suspect is None else suspect
    steps, values = log.column(name)
    kept = [(s, v) for s, v in zip(steps, values, strict=True) if s not in bad]
    return [s for s, _ in kept], [v for _, v in kept]


def describe(suspect, limit=3):
    """One-line-per-row summary for a terminal or a figure footer."""
    if not suspect:
        return []
    lines = []
    for step in sorted(suspect)[:limit]:
        lines.append(f"row {step}: {suspect[step][0]}")
    if len(suspect) > limit:
        lines.append(f"...and {len(suspect) - limit} more suspect rows")
    return lines
