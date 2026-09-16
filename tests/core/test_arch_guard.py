"""MODEL_ARCH must fail closed (#104): the selector's fallthrough default means
a typo would silently train the wrong architecture for a whole run — on the
base run, ~9 GPU-hours discovered late or never. Import-time validation turns
that into an immediate, explicit launch failure.

config validates at import, and this process has long since imported it, so every
case is a fresh import of trm.config. All nine run in one child interpreter
(`import_config_under` in tests/conftest.py) rather than nine cold starts (#325); a
refused case is one whose import raised, exactly what would kill a launch.
"""

import pytest

CASES = {
    "arch=refnier": {"MODEL_ARCH": "refnier"},
    "arch=plain": {"MODEL_ARCH": "plain"},
    "arch=refiner": {"MODEL_ARCH": "refiner"},
    "arch=reasoner": {"MODEL_ARCH": "reasoner"},
    "arch unset": {"MODEL_ARCH": None},
    "time_signal=sinsuoidal": {"TIME_SIGNAL": "sinsuoidal"},
    "time_signal=sinusoidal": {"TIME_SIGNAL": "sinusoidal"},
    "time_signal=table": {"TIME_SIGNAL": "table"},
    "time_signal unset": {"TIME_SIGNAL": None},
}


@pytest.fixture(scope="module")
def outcomes(import_config_under):
    return dict(zip(CASES, import_config_under(list(CASES.values()))))


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
