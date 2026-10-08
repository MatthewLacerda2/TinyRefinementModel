"""The JSON side of a checkpoint: everything a resume rebuilds the run from.

Saved beside the weights as orbax's `monitor_state` item and checked on load: an
unknown key or a wrong type fails naming the field, instead of resuming with a
default (#477). The trainer and the supervisor ask the same question of the files
on disk before a launch (`rewind.unresumable`, #505). A field has a default only
because checkpoints older than it lack it, and names the issue that added it.

Jax-free, like `trm/settings.py`.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

class ResumeState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    # The plateau detector (LossMonitor): its window of recent val readings and its bests.
    ce_history: list[float]
    best_ce: float
    best_loss: float
    best_avg_ce: float
    last_improvement_step: int

    # ── Absent from checkpoints written before the issue that added them ────
    # #222: absent reads as no best yet, so the first val probe after resume sets one.
    best_val_ce: float = float("inf")
    # #24: samples actually served. Absent means the run trained at BATCH_SIZE=1,
    # so the micro-step count is the exact value (see `restore`).
    samples_seen: int | None = None
    # #424: the data loader's exact position. Its inner shape is the mixer's
    # (`load_state`); absent, the resume estimates the position from samples_seen.
    data_state: dict[str, Any] | None = None

    @classmethod
    def of(cls, monitor, run_id):
        """The state of a LossMonitor, as a save records it."""
        return cls(
            run_id=run_id,
            # A copy: an asynchronous save serializes after this returns (#218).
            ce_history=list(monitor.ce_history),
            best_ce=monitor.best_ce,
            best_loss=monitor.best_loss,
            best_avg_ce=monitor.best_avg_ce,
            best_val_ce=monitor.best_val_ce,
            last_improvement_step=monitor.last_improvement_step,
            # Counted as served, never re-derived from the step count (#24).
            samples_seen=monitor.samples_seen,
            # A fresh dict per batch that nothing mutates afterwards (#424).
            data_state=monitor.data_state,
        )

    def saved(self):
        """The JSON a checkpoint writes."""
        return self.model_dump()

    @classmethod
    def load(cls, saved, where):
        """The state a checkpoint recorded, or a refusal naming every bad field
        (SystemExit, like a bad knob in `Config.from_env`). `where` names the
        checkpoint in the refusal."""
        try:
            return cls.model_validate(saved)
        except ValidationError as error:
            raise SystemExit(
                f"❌ {where}: its resume state does not match ResumeState "
                f"(trm/runtime/resume_state.py):\n" + "\n".join(
                    f"  {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())) from None

    def restore(self, monitor, micro_step):
        """Put the recorded state back into a fresh LossMonitor. `micro_step` is
        the checkpoint's step, the exact sample count of a pre-#24 checkpoint."""
        monitor.ce_history = list(self.ce_history)
        monitor.best_ce = self.best_ce
        monitor.best_loss = self.best_loss
        monitor.best_avg_ce = self.best_avg_ce
        monitor.best_val_ce = self.best_val_ce
        monitor.last_improvement_step = self.last_improvement_step
        monitor.samples_seen = micro_step if self.samples_seen is None else self.samples_seen
        monitor.data_state = self.data_state
