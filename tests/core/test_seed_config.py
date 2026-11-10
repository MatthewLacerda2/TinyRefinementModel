"""Per-run seed configurability (#17 prep): the noise-floor protocol needs
same-config runs that differ ONLY in seed, so both seeds must be overridable
from the environment and recorded in run metadata. A Config is a pure function of
the environment it is handed (#475), so both cases are built here."""

from trm.settings import Config


def test_seeds_default_and_override():
    default, override = Config.from_env({}), Config.from_env({"MODEL_SEED": "7", "DATA_SEED": "1234"})
    assert (default.MODEL_SEED, default.DATA_SEED) == (42, 42)
    assert (override.MODEL_SEED, override.DATA_SEED) == (7, 1234)


def test_seeds_recorded_in_run_metadata():
    from trm.runtime.run_tracker import RunTracker
    params = RunTracker.get_hyperparameters(Config.from_env({"MODEL_SEED": "7", "DATA_SEED": "1234"}))
    assert (params["MODEL_SEED"], params["DATA_SEED"]) == (7, 1234)
