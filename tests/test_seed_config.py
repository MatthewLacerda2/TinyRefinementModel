"""Per-run seed configurability (#17 prep): the noise-floor protocol needs
same-config runs that differ ONLY in seed, so both seeds must be overridable
from the environment and recorded in run metadata. Subprocess-based because
config.py reads the env at import time."""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
