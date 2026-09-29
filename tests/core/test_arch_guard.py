"""MODEL_ARCH must fail closed (#104): the selector's fallthrough default means
a typo would silently train the wrong architecture for a whole run — on the
base run, ~9 GPU-hours discovered late or never. Validation when the Config is read
turns that into an immediate, explicit launch failure.

Building a Config is a pure function of a mapping (#475), so each case is built here,
in-process; a refused case is one whose build raised SystemExit, exactly what
`Config.from_env` does to a launch.
"""

import pytest

from trm.settings import Config

CASES = {
    "arch=refnier": {"MODEL_ARCH": "refnier"},
    "arch=plain": {"MODEL_ARCH": "plain"},
    "arch=refiner": {"MODEL_ARCH": "refiner"},
    "arch=reasoner": {"MODEL_ARCH": "reasoner"},
    "arch unset": {},
    "time_signal=sinsuoidal": {"TIME_SIGNAL": "sinsuoidal"},
    "time_signal=sinusoidal": {"TIME_SIGNAL": "sinusoidal"},
    "time_signal=table": {"TIME_SIGNAL": "table"},
    "time_signal unset": {},
}


def _outcome(environ):
    try:
        Config.from_env(environ)
    except SystemExit as refused:
        return {"ok": False, "error": str(refused.code)}
    return {"ok": True}


@pytest.fixture(scope="module")
def outcomes():
    return {case: _outcome(environ) for case, environ in CASES.items()}


def test_unknown_model_arch_fails_closed_before_anything_builds(outcomes):
    r = outcomes["arch=refnier"]
    assert not r["ok"], "a typo'd MODEL_ARCH must refuse to start"
    assert "refnier" in r["error"], "the error must echo the bad value"
    assert all(name in r["error"].split("use one of", 1)[-1] for name in ("plain", "refiner", "reasoner")), \
        "the error must list every valid name, the default included"


def test_known_arches_and_unset_default_still_launch(outcomes):
    # plain is the default, so it is the one arch a fail-closed guard must never refuse.
    for case in ("arch=plain", "arch=refiner", "arch=reasoner", "arch unset"):
        r = outcomes[case]
        assert r["ok"], f"{case} must be accepted: {r.get('error')}"


def test_unknown_time_signal_fails_closed(outcomes):
    """#86: same fail-closed contract as MODEL_ARCH — the time signal picks the
    refiner's param tree, so a typo must refuse to launch, not silently train
    a different model."""
    r = outcomes["time_signal=sinsuoidal"]
    assert not r["ok"]
    assert "sinsuoidal" in r["error"] and "sinusoidal" in r["error"] and "table" in r["error"]


def test_known_time_signals_and_unset_default_launch(outcomes):
    for case in ("time_signal=sinusoidal", "time_signal=table", "time_signal unset"):
        r = outcomes[case]
        assert r["ok"], f"{case} must be accepted: {r.get('error')}"
