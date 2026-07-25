"""Unit tests for the time machine's pure logic (tools/timemachine.py).

The reconstruction itself (worktree, venv build, GPU eval) is validated by hand on
the card; these pin the decisions that must never silently go wrong — above all that
the arch resolver *refuses to guess* (a wrong arch corrupts the restore) and that the
metric it compares against is read correctly.
"""

import os
import json

import pytest

import tools.timemachine as tm


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Point the module at a throwaway runs/ tree."""
    monkeypatch.setattr(tm, "RUNS_ROOT", str(tmp_path))
    return tmp_path


def _make_run(runs, run_id, *, meta=None, snapshot=None, metrics=None):
    d = runs / run_id
    d.mkdir()
    if meta is not None:
        (d / "run_metadata.json").write_text(json.dumps(meta))
    if snapshot is not None:
        (d / "system_snapshot.txt").write_text(snapshot)
    if metrics is not None:
        (d / "metrics.csv").write_text(metrics)
    return d


def test_resolve_arch_prefers_machine_readable_metadata(runs):
    _make_run(runs, "r", meta={"parameters": {"MODEL_ARCH": "reasoner"}})
    assert tm.resolve_arch("r", tm.load_meta("r")) == "reasoner"


def test_resolve_arch_reads_structured_snapshot_line(runs):
    _make_run(runs, "r", meta={"parameters": {}}, snapshot="python 3.14\nMODEL_ARCH reasoner\n")
    assert tm.resolve_arch("r", tm.load_meta("r")) == "reasoner"


def test_resolve_arch_reads_legacy_arm_marker(runs):
    _make_run(runs, "r", meta={"parameters": {}}, snapshot="arm=refiner (#16)\n")
    assert tm.resolve_arch("r", tm.load_meta("r")) == "refiner"


def test_resolve_arch_refuses_to_guess(runs):
    # No metadata arch, no snapshot: must return None so the caller demands --arch
    # rather than defaulting and rebuilding the wrong (incompatible) skeleton.
    _make_run(runs, "r", meta={"parameters": {}})
    assert tm.resolve_arch("r", tm.load_meta("r")) is None


def test_recorded_val_ce_takes_last_non_empty(runs):
    metrics = "step,val_ce\n10,5.10\n20,\n30,4.8947\n"
    _make_run(runs, "r", meta={"parameters": {}}, metrics=metrics)
    assert tm.recorded_val_ce("r") == pytest.approx(4.8947)


def test_recorded_val_ce_missing_file_is_none(runs):
    _make_run(runs, "r", meta={"parameters": {}})
    assert tm.recorded_val_ce("r") is None


def test_venv_key_is_deterministic_and_content_addressed(runs):
    a = _make_run(runs, "a", meta={"parameters": {}})
    b = _make_run(runs, "b", meta={"parameters": {}})
    (a / "env_freeze.txt").write_text("jax==0.9.1\n")
    (b / "env_freeze.txt").write_text("jax==0.9.1\n")
    ka, _ = tm.venv_key("a")
    kb, _ = tm.venv_key("b")
    assert ka is not None and ka == kb  # identical freeze -> shared venv

    (b / "env_freeze.txt").write_text("jax==0.9.2\n")
    kb2, _ = tm.venv_key("b")
    assert kb2 != ka  # different freeze -> different venv


def test_venv_key_none_without_freeze(runs):
    _make_run(runs, "r", meta={"parameters": {}})
    assert tm.venv_key("r") == (None, None)
