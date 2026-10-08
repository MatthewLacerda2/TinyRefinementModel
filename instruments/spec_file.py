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
(instruments/stall.py, trm/runtime/cold.py), which validates it itself.
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
