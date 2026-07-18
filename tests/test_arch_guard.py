"""MODEL_ARCH must fail closed (#104): the selector's fallthrough default means
a typo would silently train the wrong architecture for a whole run — on the
base run, ~9 GPU-hours discovered late or never. Import-time validation turns
that into an immediate, explicit launch failure.

Subprocess-based: config validates at import, and this process has long since
imported it, so each case gets a fresh interpreter.
"""

import os
import subprocess
import sys

def _import_config_with(arch):
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    if arch is None:
        env.pop("MODEL_ARCH", None)
    else:
        env["MODEL_ARCH"] = arch
    return subprocess.run([sys.executable, "-c", "import config"],
                          env=env, capture_output=True, text=True)

def test_unknown_model_arch_fails_closed_before_anything_builds():
    r = _import_config_with("refnier")
    assert r.returncode != 0, "a typo'd MODEL_ARCH must refuse to start"
    assert "refnier" in r.stderr, "the error must echo the bad value"
    assert "refiner" in r.stderr and "reasoner" in r.stderr, \
        "the error must list the valid names"

def test_known_arches_and_unset_default_still_launch():
    for arch in ("refiner", "reasoner", None):
        r = _import_config_with(arch)
        assert r.returncode == 0, f"MODEL_ARCH={arch!r} must be accepted: {r.stderr}"
