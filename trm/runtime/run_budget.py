"""The LR horizon belongs to the run, not to the shell that relaunched it.

`TRAIN_TOKEN_BUDGET` is read from the environment by `trm.config` and resolved into
`schedules.DECAY_STEPS` at import time. That is right for a first launch, where a
human types the budget on the command line. It is wrong for a *resume*: there the
environment is whatever the relaunching process happened to inherit, and when it
carries no budget the cosine quietly falls back to `_DEFAULT_DECAY_STEPS`.

Not hypothetical. A mains power cut killed the #157 base run at opt step 10,715 on
2026-08-17. The relaunch resolved DECAY_STEPS=15,000 for a run built with 30,518,
which would have annealed the learning rate to zero with half the run left to
train. Nothing crashed and nothing was logged as wrong — the launch banner was the
only evidence, and it only helps if a human happens to read it. Worse, the crash
relaunch in `trm.runtime.supervisor` is unattended by definition: without this,
every automatic restart after a budget-less start trains on the wrong schedule
forever.

The run has recorded the answer in its own `run_metadata.json` the whole time. This
module reads it back, so resume is restored from the run rather than from the shell
— the same correction #183 made for the environment snapshot.

Stdlib only, deliberately: it must be importable from `trm.train.start` *before*
`trm.config` is, because DECAY_STEPS is resolved at import time and cannot be
changed afterwards.
"""

import json
import os

BUDGET_ENV = "TRAIN_TOKEN_BUDGET"
METADATA_FILENAME = "run_metadata.json"


def checkpoint_path_from_argv(argv):
    """The `--checkpoint-path` value, read straight off argv.

    argparse proper runs much later, after JAX has loaded; the budget has to be in
    the environment before that, so this reads the one flag it needs by hand.
    """
    for index, arg in enumerate(argv):
        if arg == "--checkpoint-path" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--checkpoint-path="):
            return arg.split("=", 1)[1]
    return None


def metadata_for_checkpoint(checkpoint_path):
    """`runs/<run_id>/checkpoints` -> that run's metadata, or {} when there is none.

    A missing or unreadable file is the normal case for a brand-new run, so it
    returns empty rather than raising — the caller falls back to the environment,
    which is exactly today's behaviour.
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


def adopt_recorded_budget(checkpoint_path, environ=None):
    """Put the resumed run's own token budget back into the environment.

    Returns the budget adopted, or None if there was nothing to adopt (new run, no
    metadata, or a run that genuinely had no budget). Uses setdefault semantics: an
    explicit `TRAIN_TOKEN_BUDGET` in the shell still wins, so a deliberate override
    — extending a run's horizon on purpose — behaves as before. Must be called
    before `trm.config` is imported.
    """
    environ = os.environ if environ is None else environ
    if environ.get(BUDGET_ENV):
        return None
    budget = metadata_for_checkpoint(checkpoint_path).get(BUDGET_ENV)
    if budget is None:
        return None
    environ[BUDGET_ENV] = str(int(budget))
    return int(budget)


def horizon_mismatch(run_dir, decay_steps):
    """The complaint to raise when the live LR horizon isn't the run's own, else None.

    Checked separately from adoption because adoption can't cover every path — a
    resume that lets the trainer auto-discover its run has no checkpoint path to
    read at import time. This runs once the run directory is known and turns that
    remaining gap from silently-wrong into loudly-stopped. A stopped run is
    recoverable; fifteen thousand steps at the wrong learning rate is not.
    """
    recorded = read_metadata(os.path.join(run_dir, METADATA_FILENAME)).get("DECAY_STEPS")
    if recorded is None or int(recorded) == int(decay_steps):
        return None
    return (
        f"LR horizon mismatch: this run recorded DECAY_STEPS={int(recorded):,} but the "
        f"current environment resolves to {int(decay_steps):,}. Resuming would train on a "
        f"different learning-rate schedule than the run was built for, silently. "
        f"Relaunch with the run's own budget, e.g. {BUDGET_ENV}=<tokens> (see "
        f"{os.path.join(run_dir, METADATA_FILENAME)}), or pass --checkpoint-path so it is "
        f"recovered automatically."
    )
