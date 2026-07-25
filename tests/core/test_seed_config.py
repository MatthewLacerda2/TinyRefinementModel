"""Per-run seed configurability (#17 prep): the noise-floor protocol needs
same-config runs that differ ONLY in seed, so both seeds must be overridable
from the environment and recorded in run metadata. Subprocess-based because
config.py reads the env at import time."""

import json
import os
import pathlib
import subprocess
import sys

# Anchored on the marker file, not on a fixed number of parent hops — the old
# `dirname(dirname(__file__))` silently became `tests/` the day this file moved
# into a tier folder, and the subprocess failed with a bare ModuleNotFoundError.
REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def _read_config(env_overrides):
    env = {**os.environ, **env_overrides}
    out = subprocess.check_output(
        [sys.executable, "-c",
         "import json, config; print(json.dumps({'model': config.MODEL_SEED, 'data': config.DATA_SEED}))"],
        env=env, cwd=REPO,
    )
    return json.loads(out)


def test_seeds_default_and_override():
    assert _read_config({}) == {"model": 42, "data": 42}
    assert _read_config({"MODEL_SEED": "7", "DATA_SEED": "1234"}) == {"model": 7, "data": 1234}


def test_seeds_recorded_in_run_metadata():
    from run_tracker import RunTracker
    params = RunTracker().get_hyperparameters()
    assert params["MODEL_SEED"] == 42
    assert params["DATA_SEED"] == 42
