"""Metrics whose correct value is fixed by construction, used as integrity checks.

Some logged quantities cannot move, no matter what the model, the data mixture or
the schedule does. `depth_avg` is the sharpest: depth is drawn uniformly from
1..MAX_STEPS_LIMIT once per micro-step, and one optimizer step averages
ACCUMULATION_STEPS of those draws — so its value is a sample mean of a
distribution we know exactly, forever. A row where it reads 2.76 is not a
measurement of anything; it is proof that the row's accumulator divided by the
wrong count.

That is worth having as a check because of what it lets you *rule out*. On
2026-08-16 a resume wrote `ce = 1.8746` into metrics.csv, a 40% drop from the
surrounding rows. Two explanations were available and they are very far apart in
consequence: a harmless logging artifact, or the data pipeline re-seeking wrong
and re-serving text the model had already trained on — the silent failure
`samples_seen` exists to prevent (see split_samples' docstring in the trainer:
"No crash, no warning, just a bad run"). What settled it in minutes was
`depth_avg = 2.7641` on the same row, scaled by the same factor as the CE. One
number with a known answer collapsed the question.

A violated invariant condemns the WHOLE ROW, not just the offending column. The
accumulator is shared, so if one metric on the row is wrong the rest are too —
and that is exactly what was observed: CE and depth both fell to ~0.6x.

Deliberately NOT an invariant: an upper bound on cross-entropy. It is tempting
to bound CE by ln(VOCAB_SIZE) = 10.83 on the grounds that no model should be
worse than uniform, and that is wrong — a freshly initialised model can be worse
than uniform, and this project has logged 11.08 at step 5. An invariant that
fires on a healthy run is worse than no invariant, because it teaches people to
ignore it.
"""

import math

from trm.config import ACCUMULATION_STEPS, MAX_STEPS_LIMIT, VOCAB_SIZE

# How many standard deviations of the sampling distribution `depth_avg` may sit
# from its mean before the row is called suspect. The observed artifact was 8.6
# sigma out, and a real run's rows land inside ~2, so five leaves a wide corridor
# on both sides. Wide on purpose: a false positive on a healthy run would train
# the reader to disregard the check.
DEPTH_SIGMA_TOLERANCE = 5.0


def depth_expectation():
    """(mean, sigma) of `depth_avg` for the current config.

    Depth is uniform over the integers 1..K per micro-step, so mean = (K+1)/2 and
    variance = (K^2 - 1)/12. One logged row averages N = ACCUMULATION_STEPS of
    them, which divides the variance by N.

    Derived rather than hardcoded so the check follows the config: change
    MAX_STEPS_LIMIT or ACCUMULATION_STEPS and the corridor moves with it, instead
    of silently becoming wrong.
    """
    k = MAX_STEPS_LIMIT
    mean = (k + 1) / 2
    sigma = math.sqrt((k * k - 1) / 12 / ACCUMULATION_STEPS)
    return mean, sigma


def _depth_bounds():
    mean, sigma = depth_expectation()
    spread = DEPTH_SIGMA_TOLERANCE * sigma
    # Never wider than the support: a mean of draws in [1, K] cannot leave [1, K]
    # however loose the sigma corridor gets on a small ACCUMULATION_STEPS.
    return max(1.0, mean - spread), min(float(MAX_STEPS_LIMIT), mean + spread)


def build_invariants():
    """(column, low, high, why) for every quantity with a known-correct range.

    Built on call rather than at import so a test can vary the config and get
    matching bounds.
    """
    depth_low, depth_high = _depth_bounds()
    mean, sigma = depth_expectation()
    return (
        ("depth_avg", depth_low, depth_high,
         f"mean of {ACCUMULATION_STEPS} uniform draws over 1..{MAX_STEPS_LIMIT}: "
         f"{mean:.2f} +/- {sigma:.3f}, so {DEPTH_SIGMA_TOLERANCE:.0f} sigma is "
         f"[{depth_low:.2f}, {depth_high:.2f}]"),
        ("zero_frac_dense_max", 0.0, 1.0, "it is a fraction"),
        ("forget_density", 0.0, 1.0, "it is a fraction"),
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
        ("grad_norm_avg", 0.0, math.inf, "it is a norm"),
    )


def row_violations(row, invariants=None):
    """Every invariant this row breaks, as human-readable strings. Empty if clean.

    A blank cell is not a violation — it is an arch-optional column (#105), the
    normal case. Non-finite values are, since NaN passes every comparison.
    """
    broken = []
    for column, low, high, why in (invariants or build_invariants()):
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
    invariants = build_invariants()
    suspect = {}
    for row in log.metrics:
        broken = row_violations(row, invariants)
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
    kept = [(s, v) for s, v in zip(steps, values) if s not in bad]
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
