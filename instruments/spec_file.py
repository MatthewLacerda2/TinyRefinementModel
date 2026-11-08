"""An experiment spec TOML, parsed into typed tables at the boundary (#478).

Every reader of a spec goes through `SpecFile.load`, so a misspelled key or a wrong
type fails when the spec is read, naming the table and the key, and not as a
KeyError two hours into a sweep (or as nothing at all). The format is
docs/design/experiment-spec.md.

Only [experiment] (its id, title and hypothesis), [criteria] and [verdict] are
required: that is all the referee needs. The runner checks what it needs beside
them (instruments/experiment.py: load_execution).

Strict where the machine acts on a key: [experiment], [arms.*], [criteria.*],
[execution], [verdict], [readouts] refuse a key they do not know. [protocol] is the
human record of how the run was set up, and retrofitted specs carry whatever their
finding recorded (`dim`, `K`, `heldout`, …), so beyond the keys a tool reads it
keeps the rest as written. [stall] belongs to the base-run line's stall rule
(instruments/stall.py, trm/runtime/cold.py), which validates it itself. [stages] is
optional (#604): without it a spec is judged once, at its full seed count.
"""

from __future__ import annotations

import tomllib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentTable(_Table):
    id: str
    title: str
    hypothesis: str
    status: Literal["open", "recorded", "retrofit"] | None = None
    issue: int | None = None
    finding: str | None = None
    commit: str | None = None
    recorded: str | None = None


class ProtocolTable(BaseModel):
    """The keys tools read are typed; the rest of the record (`matched`, `floor_note`,
    `notes`, a retrofit's own setup) is kept as written."""
    model_config = ConfigDict(extra="allow", frozen=True)

    metric: str | None = None
    metric_key: str | None = None
    cap: float | None = None
    budget_tokens: int | None = None
    # Registered seeds, for a spec with no [execution] (retrofits); a base run's is prose.
    seeds: tuple[int, ...] | str | None = None


class ArmTable(_Table):
    role: str
    flags: tuple[str, ...] = ()
    # Not run: its values are supplied in [results] (an absolute floor, a reference).
    constant: bool = False
    config_delta: dict[str, Any] | None = None
    note: str | None = None


class CriterionTable(_Table):
    rule: str
    treatment: str
    control: str
    sigmas: float
    points: tuple[str, ...] = ()
    require: Literal["all", "any"] = "all"
    min_delta: float | None = None
    # `equivalent` only: the parity margin in the metric's own units, in place of
    # sigmas * sigma_pooled (#604, the owner's decision of 2026-11-06).
    margin: float | None = None


class LegTable(_Table):
    flags: tuple[str, ...] = ()
    seeds: tuple[int, ...] | None = None
    arms: tuple[str, ...] | None = None


class ExecutionTable(_Table):
    command: tuple[str, ...]
    seeds: tuple[int, ...] = ()
    seed_flag: str = "--seed"
    env: dict[str, str] = {}
    legs: dict[str, LegTable] = {}

    @field_validator("command", mode="before")
    @classmethod
    def _a_list(cls, value):
        if isinstance(value, str):
            raise ValueError(
                'must be a LIST of arguments, not a string. Write ["python", "-m", "pkg.mod", '
                '"--flag", "value"], because a string here would be iterated character by character')
        return value


# Fewer seeds than this is a noise floor that is itself noise (rule 1); the same
# floor tests/apparatus/test_spec_seed_lint.py holds a registered seed list to.
MIN_LOOK_SEEDS = 3


class StagesTable(_Table):
    """Two pre-registered looks at a parity pair (#604): judge at `seeds[0]` seeds a
    side with that look's alpha-spent bar, stop if it settles, otherwise run on to
    `seeds[1]` and judge at the final bar. instruments/verdict.py holds the
    arithmetic; docs/design/experiment-spec.md the format."""
    seeds: tuple[int, int]  # seeds per arm at look 1 and at look 2, cumulative
    # The Lan-DeMets spending function that splits alpha across the two looks. Pocock by
    # the owner's call (2026-11-08): 6.2 seeds a side vs O'Brien-Fleming's 7.6, for power
    # 0.945 vs 0.968; the table is in docs/design/experiment-spec.md.
    spending: Literal["obrien-fleming", "pocock"] = "pocock"
    # Stop at look 1 when the gap is shown beyond the margin (the owner's option (b)).
    futility: bool = True

    @field_validator("seeds")
    @classmethod
    def _two_growing_looks(cls, value):
        first, final = value
        if first < MIN_LOOK_SEEDS or final <= first:
            raise ValueError(f"looks must be at least {MIN_LOOK_SEEDS} seeds a side and grow, "
                             f"got {list(value)}")
        return value


class VerdictTable(_Table):
    keep_if: tuple[str, ...] = ()
    kill_if: tuple[str, ...] = ()
    recorded: str | None = None
    recorded_note: str | None = None


class ReadoutsTable(_Table):
    notes: str | None = None
    names: tuple[str, ...] = ()
    readouts_note: str | None = None


class VoidTable(_Table):
    """Seeds missing for a stated reason, declared rather than silently absent."""
    seeds: tuple[int, ...]
    reason: str


class RunsTable(_Table):
    """Where an arm's runs left their logs, for the audit's budget-stop rule."""
    logs: tuple[str, ...] = ()
    dirs: tuple[str, ...] = ()


class SummaryTable(_Table):
    """A result that only exists summarized: an arm's mean, sigma and seed count."""
    mean: float
    sigma: float
    n: int


class SpecFile(_Table):
    experiment: ExperimentTable
    protocol: ProtocolTable = ProtocolTable()
    arms: dict[str, ArmTable] = {}
    criteria: dict[str, CriterionTable]
    verdict: VerdictTable
    execution: ExecutionTable | None = None
    readouts: ReadoutsTable = ReadoutsTable()
    # {point: {arm: per-seed values, or a summary}}, written by the runner after the run.
    results: dict[str, dict[str, list[float] | SummaryTable]] = {}
    # {point: {arm: the seeds declared void}}.
    void: dict[str, dict[str, VoidTable]] = {}
    runs: dict[str, RunsTable] = {}
    stall: dict[str, Any] | None = None
    stages: StagesTable | None = None

    @classmethod
    def load(cls, path) -> SpecFile:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise ValueError(f"{path}: " + "; ".join(
                f"[{'.'.join(map(str, e['loc']))}] {e['msg']}" for e in error.errors())) from None

    def runnable_arms(self) -> dict[str, ArmTable]:
        return {name: arm for name, arm in self.arms.items() if not arm.constant}

    def constant_arms(self) -> set[str]:
        return {name for name, arm in self.arms.items() if arm.constant}
