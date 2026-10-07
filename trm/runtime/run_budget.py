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

import json
import os

BUDGET_ENV = "TRAIN_TOKEN_BUDGET"
METADATA_FILENAME = "run_metadata.json"


def metadata_for_checkpoint(checkpoint_path):
    """`runs/<run_id>/checkpoints` -> that run's metadata, or {} when there is none.

    A missing or unreadable file is the normal case for a brand-new run, so it
    returns empty rather than raising.
    """
    if not checkpoint_path:
        return {}
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), METADATA_FILENAME)
    return read_metadata(path)


def read_metadata(path):
    """The `parameters` block of a run_metadata.json, or {} if it can't be read."""
    try:
        with open(path) as handle:
            return json.load(handle).get("parameters") or {}
    except (OSError, ValueError):
        return {}


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
    recorded = read_metadata(os.path.join(run_dir, METADATA_FILENAME)).get("DECAY_STEPS")
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
