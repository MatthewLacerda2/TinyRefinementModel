"""A run's held-out CE curve, read once from its metrics.csv, and where it crosses a target.

Every reader of a val curve goes through `ValCurve.read`: the pair harness's
tokens-to-target, the pair report. A reading is placed at the probe's own step
(`val_step`, #351) when the run recorded it, else at the logged row's step, which is
late by up to LOG_REAL_STEPS - 1 opt steps; `aligned` says which.

Two crossings (#547). `first_at_or_below` is the first probe at or below the target,
so it is quantized to the probe interval: three seeds inside one interval repeat
exactly, and their sigma is 0. `crossing` interpolates linearly between the last
probe above the target and the first at or below it, so the metric is continuous
and the seeds' spread is measured, not rounded away.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ValCurve:
    steps: tuple[int, ...]
    ces: tuple[float, ...]
    aligned: bool

    @classmethod
    def read(cls, metrics_csv: pathlib.Path, cap_steps: int | None = None) -> ValCurve:
        """The probes up to `cap_steps` (all of them when None), in step order."""
        readings, aligned = {}, False
        with metrics_csv.open() as f:
            for row in csv.DictReader(f):
                try:
                    val = row.get("val_ce") or ""
                    probe = row.get("val_step") or ""
                    aligned = aligned or bool(probe)
                    step = int(probe) if probe else int(row["step"])
                except (KeyError, ValueError):
                    continue
                if not val or (cap_steps is not None and step > cap_steps):
                    continue
                readings[step] = float(val)
        steps = tuple(sorted(readings))
        return cls(steps, tuple(readings[s] for s in steps), aligned)

    @property
    def final(self) -> float | None:
        return self.ces[-1] if self.ces else None

    def at(self, step: float) -> float | None:
        """The latest reading at or before `step`."""
        before = [ce for s, ce in zip(self.steps, self.ces, strict=True) if s <= step]
        return before[-1] if before else None

    def first_at_or_below(self, target: float) -> int | None:
        return next((s for s, ce in zip(self.steps, self.ces, strict=True) if ce <= target), None)

    def crossing(self, target: float) -> float | None:
        """The step, interpolated between bracketing probes, where CE first reaches
        `target`; the first probe's own step when it already does; None when never."""
        previous = None
        for step, ce in zip(self.steps, self.ces, strict=True):
            if ce <= target:
                if previous is None:
                    return float(step)
                s0, c0 = previous  # c0 > target >= ce, so the slope is never zero
                return s0 + (step - s0) * (c0 - target) / (c0 - ce)
            previous = (step, ce)
        return None
