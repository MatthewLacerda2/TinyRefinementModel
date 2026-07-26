"""The one line a harness prints so a machine can read its numbers.

A harness prints a table for a human. The runner (`instruments/experiment.py`)
needs the same numbers without guessing at column positions, so harnesses also
emit one machine-readable line per measurement:

    RESULT {"point": "d8", "acc": 0.7123, "ce": 0.8310}

`point` names the sweep point — whatever axis the harness is walking (a depth, a
K, a task size). Everything else is a metric; the spec says which one is judged.

Deliberately *not* in the line: the arm and the seed. The runner launched the
process, so it already knows both, and a harness that had to be told its own
arm name would need to understand the spec format — which would couple every
research line to the runner. The harness reports what it measured; the runner
knows what it asked for.
"""

from __future__ import annotations

import json
from typing import Any

PREFIX = "RESULT "


def emit(point: Any, **metrics: float) -> None:
    """Print one machine-readable measurement. Flushed, because a sweep is long
    and a runner reading a dead process's buffer gets nothing."""
    row = {"point": str(point), **{k: float(v) for k, v in metrics.items()}}
    print(PREFIX + json.dumps(row, sort_keys=True), flush=True)


def parse(text: str) -> list[dict]:
    """Pull the RESULT lines out of a harness's stdout, ignoring the rest.

    A malformed RESULT line is an error, not something to skip: it means the
    protocol drifted, and silently dropping it would turn a broken harness into
    a sweep that quietly measured fewer seeds than it claims.
    """
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith(PREFIX):
            continue
        try:
            row = json.loads(line[len(PREFIX):])
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: malformed RESULT line: {line!r}") from exc
        if "point" not in row:
            raise ValueError(f"line {lineno}: RESULT line has no 'point': {line!r}")
        rows.append(row)
    return rows
