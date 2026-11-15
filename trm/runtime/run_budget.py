"""The LR horizon belongs to the run, not to the shell that relaunched it (#197).

`TRAIN_TOKEN_BUDGET` sets the LR horizon (`Schedules.decay_steps`). That is right for a
first launch, where a human types the budget on the command line. It is wrong for a
*resume*: there the environment is whatever the relaunching process happened to
inherit, and when it carries no budget the schedule quietly falls back to
`schedules.DEFAULT_DECAY_STEPS`. A mains power cut on the #157 base run is how we
learned it, and the supervisor's crash relaunch is unattended by definition (#197).

The run has recorded the answer in its own `run_metadata.json` the whole time.
`with_recorded_budget` reads it back into the Config the trainer is handed, so a resume
is restored from the run rather than from the shell. `horizon_mismatch` is the backstop
for a horizon that still disagrees: a budget set explicitly to something else.

Stdlib only: `trm.runtime.launch` reads BUDGET_ENV from here and stays jax-free.
"""

import os

from trm.runtime.run_metadata import METADATA_FILENAME, RunMetadata

BUDGET_ENV = "TRAIN_TOKEN_BUDGET"


def metadata_for_checkpoint(checkpoint_path):
    """`runs/<run_id>/checkpoints` -> that run's metadata, or {} when there is none.

    A missing or unreadable file is the normal case for a brand-new run, so it
    returns empty rather than raising.
    """
    if not checkpoint_path:
        return {}
    return recorded_parameters(os.path.dirname(os.path.abspath(checkpoint_path)))


def recorded_parameters(run_dir):
    """The `parameters` block of the run's metadata, or {} when it has no readable file."""
    metadata = RunMetadata.read(run_dir)
    return metadata.parameters if metadata is not None else {}


def wsd_decay_start(decay_steps, decay_fraction, decay_start=None):
    """The opt step WSD's final decay starts at: the explicit one, else the fraction's.
    Here, not beside the schedule it shapes (trm/train/schedules.py, which imports it),
    so the supervisor can ask where a run's decay starts without importing jax."""
    return decay_start if decay_start is not None else round((1 - decay_fraction) * decay_steps)


def last_full_rate_checkpoint(run_dir):
    """The highest step a checkpoint of this run can be named and still hold a state no
    decay has touched: the one written at the opt step its WSD decay starts, from the
    run's own record. The cold tier keeps the last rolling checkpoint at or below it
    (#640).

    None when the record cannot say: no record yet, a schedule with no decay start (the
    cosine anneals from the warmup on), or a WSD record missing one of the numbers.
    None, not an error, because the cold tier asks every tick before its window copy
    and its pruning, and this copy must never cost a run those.

    The record is the launch's: RunTracker writes `parameters` once and a resume leaves
    them as they were. A resume of the same run that moves the decay start or the
    batch size is therefore not followed; a branch is, since it is a new run."""
    recorded = recorded_parameters(run_dir)
    horizon, fraction, accumulation = (recorded.get(key) for key in
                                       ("DECAY_STEPS", "WSD_DECAY_FRACTION", "ACCUMULATION_STEPS"))
    if recorded.get("LR_SCHEDULE") != "wsd" or None in (horizon, fraction, accumulation):
        return None
    start = wsd_decay_start(horizon, fraction, recorded.get("WSD_DECAY_START"))
    # Orbax names a checkpoint by the last micro-step it holds, so the one written
    # after opt step k is k * ACCUMULATION_STEPS - 1 (rewind.checkpoints_in reads it back).
    return start * accumulation - 1


def with_recorded_budget(config, checkpoint_path):
    """`config` with the resumed run's own token budget, when it set none itself.

    Unchanged for a new run, a run with no metadata, or a run that genuinely had no
    budget. A budget the launch set explicitly still wins, so a deliberate override
    reaches `horizon_mismatch` instead of being replaced.
    """
    if config.TRAIN_TOKEN_BUDGET is not None:
        return config
    budget = metadata_for_checkpoint(checkpoint_path).get(BUDGET_ENV)
    if budget is None:
        return config
    return config.model_copy(update={BUDGET_ENV: int(budget)})


def horizon_mismatch(run_dir, decay_steps):
    """The complaint to raise when the live LR horizon isn't the run's own, else None.

    A stopped run is recoverable; fifteen thousand steps at the wrong learning rate
    is not.
    """
    recorded = recorded_parameters(run_dir).get("DECAY_STEPS")
    if recorded is None or int(recorded) == int(decay_steps):
        return None
    return (
        f"LR horizon mismatch: this run recorded DECAY_STEPS={int(recorded):,} but the "
        f"current configuration resolves to {int(decay_steps):,}. Resuming would train on a "
        f"different learning-rate schedule than the run was built for, silently. "
        f"Relaunch with the run's own budget, e.g. {BUDGET_ENV}=<tokens> (see "
        f"{os.path.join(run_dir, METADATA_FILENAME)}), or leave {BUDGET_ENV} unset so it is "
        f"recovered from the run."
    )
