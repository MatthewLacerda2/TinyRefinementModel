"""Per-run seed configurability (#17 prep): the noise-floor protocol needs
same-config runs that differ ONLY in seed, so both seeds must be overridable
from the environment and recorded in run metadata. config reads the env at import
time, so the cases are fresh imports, both in one child interpreter
(`import_config_under` in tests/conftest.py, #325)."""


def test_seeds_default_and_override(import_config_under):
    default, override = import_config_under(
        [{}, {"MODEL_SEED": "7", "DATA_SEED": "1234"}], attrs=("MODEL_SEED", "DATA_SEED"))
    assert default["values"] == {"MODEL_SEED": 42, "DATA_SEED": 42}
    assert override["values"] == {"MODEL_SEED": 7, "DATA_SEED": 1234}


def test_seeds_recorded_in_run_metadata():
    from trm.runtime.run_tracker import RunTracker
    params = RunTracker().get_hyperparameters()
    assert params["MODEL_SEED"] == 42
    assert params["DATA_SEED"] == 42
