"""A run's run_metadata.json, parsed into a typed object at the boundary (#578).

RunTracker writes it; the trainer's resume check, the run budget and the instruments
read it. Every one of them goes through `RunMetadata`, so a misspelled or mistyped
key fails where the file is read, naming it, instead of reading as a silent default.

`parameters` is the Config of the run's own era, recorded whole. It stays a mapping:
a run from before a knob existed (or after one retired) records a different set, and
reading old runs is the point of recording them.

Missing, unreadable or half-written is `read`'s None, never an exception: a run
assembled by hand has no file, and RunTracker rewrites it at session start and end,
so a reader beside training can catch it torn. A caller that needs it says so itself.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

METADATA_FILENAME = "run_metadata.json"


class Section(BaseModel):
    """One session of the run: a launch or a resume, until it last reported."""
    model_config = ConfigDict(extra="forbid")

    start_time: str
    end_time: str | None = None
    duration_seconds: float | None = None


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    parameters: dict[str, Any] = {}
    sections: list[Section] = []

    @classmethod
    def read(cls, run_dir) -> RunMetadata | None:
        """The run's metadata, or None when there is no readable file (see the module)."""
        path = os.path.join(run_dir, METADATA_FILENAME)
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise ValueError(f"{path}: " + "; ".join(
                f"[{'.'.join(map(str, e['loc']))}] {e['msg']}" for e in error.errors())) from None

    def write(self, run_dir) -> None:
        with open(os.path.join(run_dir, METADATA_FILENAME), "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @property
    def hours(self) -> float:
        """Running time over every session that reported one."""
        return sum(s.duration_seconds or 0.0 for s in self.sections) / 3600
