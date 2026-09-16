"""Milestone report wrapper: cheap CPU checks, no checkpoint or GPU needed.

The wrapper must tolerate a failing diagnostic (report it, keep going) and
degrade gracefully when there is nothing to report on.
"""

import os
import pathlib
import subprocess
import sys

# Marker-anchored, not a fixed parent-hop count (see tests/core/test_seed_config.py).
REPO_ROOT = str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()))
# Invoked as a module (#143), the way milestone_report runs its own sub-tools.
MODULE = "instruments.milestone_report"


def test_section_failure_is_tolerated_and_labeled():
    from instruments.milestone_report import run_section

    failed = run_section("boom", lambda: 1 / 0)
    assert failed["status"] == "FAILED"
    assert "ZeroDivisionError" in failed["body"]

    ok = run_section("fine", lambda: "all good")
    assert ok["status"] == "ok"
    assert ok["body"] == "all good"


def _run(args, cwd):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, JAX_PLATFORMS="cpu", FORCE_F32_COMPUTE="1")
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=300)


def test_help_is_fast_and_clean(tmp_path):
    proc = _run(["--help"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "--ckpt" in proc.stdout


def test_no_checkpoint_degrades_gracefully(tmp_path):
    # Empty cwd: no runs/ at all. Must exit nonzero with a clear message,
    # not a traceback.
    proc = _run([], cwd=tmp_path)
    assert proc.returncode != 0
    assert "No checkpoint found" in proc.stderr + proc.stdout
    assert "Traceback" not in proc.stderr


def test_the_depth_section_says_plain_has_no_dial_not_that_it_is_the_reasoner():
    """#314: plain fell through to the reasoner's explanation."""
    from instruments.milestone_report import section_depth_curve

    body = section_depth_curve("plain", [], None, None)
    assert "plain" in body and "reasoner" not in body


def test_the_val_ce_section_drives_the_real_probe(monkeypatch, tmp_path):
    """#314: section 3 called `trainer.ValidationProbe()` without its data_root and read
    constants (VAL_BATCHES among them) that exist nowhere, so every report printed it as
    FAILED. It must build the probe the trainer builds and name the probe's real knobs."""
    from instruments.milestone_report import section_val_ce
    from trm.runtime import restore
    from trm.train import validation

    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert section_val_ce("unused").startswith("skipped")

    built = {}

    class FakeProbe:
        def __init__(self, data_root):
            built["data_root"] = data_root

        def run(self, model):
            built["model"] = model
            return 3.25

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(restore, "restore_model", lambda path: (f"model@{path}", 7))
    monkeypatch.setattr(validation, "ValidationProbe", FakeProbe)
    body = section_val_ce("ckpt")
    assert built == {"data_root": str(tmp_path), "model": "model@ckpt"}
    assert body.startswith("validation CE: 3.2500 nats")
    assert f"fixed depth {validation.VAL_FIXED_DEPTH}, {validation.VAL_ROWS} rows" in body
