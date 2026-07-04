"""Milestone report wrapper: cheap CPU checks, no checkpoint or GPU needed.

The wrapper must tolerate a failing diagnostic (report it, keep going) and
degrade gracefully when there is nothing to report on.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "tools", "milestone_report.py")


def test_section_failure_is_tolerated_and_labeled():
    from tools.milestone_report import run_section

    failed = run_section("boom", lambda: 1 / 0)
    assert failed["status"] == "FAILED"
    assert "ZeroDivisionError" in failed["body"]

    ok = run_section("fine", lambda: "all good")
    assert ok["status"] == "ok"
    assert ok["body"] == "all good"


def _run(args, cwd):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, JAX_PLATFORMS="cpu", FORCE_F32_COMPUTE="1")
    return subprocess.run([sys.executable, SCRIPT, *args], cwd=cwd,
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
