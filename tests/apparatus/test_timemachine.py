"""Unit tests for the time machine's pure logic (instruments.timemachine).

The reconstruction itself (worktree, venv build, GPU eval) is validated by hand on
the card; these pin the decisions that must never silently go wrong — above all that
the arch resolver *refuses to guess* (a wrong arch corrupts the restore) and that the
metric it compares against is read correctly.
"""

import json

import pytest

import instruments.timemachine as tm


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


def test_recorded_val_ces_are_the_finite_rows_by_opt_step(runs):
    metrics = "step,val_ce\n10,5.10\n20,\n30,nan\n40,4.8947\n"
    _make_run(runs, "r", meta={"parameters": {}}, metrics=metrics)
    assert tm.recorded_val_ces("r") == {10: pytest.approx(5.10), 40: pytest.approx(4.8947)}


def test_a_real_micro_step_checkpoint_matches_its_own_val_row(runs):
    """#348: orbax names checkpoints by micro-step. run_287's opt step 184 is checkpoint
    23551 at 128 accumulation steps; matching 23551 against the opt-step keys never
    fired, so every revival silently compared against the last logged val CE."""
    metrics = "step,val_ce\n120,5.30\n184,4.9021\n248,4.71\n"
    _make_run(runs, "r", meta={"parameters": {"ACCUMULATION_STEPS": 128}}, metrics=metrics)
    val_ces = tm.recorded_val_ces("r")
    accumulation = tm.runlog.recorded_params(tm.load_meta("r"))["ACCUMULATION_STEPS"]
    assert tm.checkpoint_opt_step(23551, accumulation) == 184
    value, chosen = tm.val_ce_for_checkpoint(val_ces, 23551, accumulation)
    assert value == pytest.approx(4.9021), "the checkpoint's own row, not the last one (4.71)"
    assert "opt step 184" in chosen


def test_a_checkpoint_between_probes_says_which_row_it_is_compared_with():
    val_ces = {120: 5.30, 184: 4.9021, 248: 4.71}
    value, chosen = tm.val_ce_for_checkpoint(val_ces, 25599, 128)  # opt step 200
    assert value == pytest.approx(4.9021)
    assert "opt step 200" in chosen and "no val row" in chosen and "opt step 184" in chosen


def test_nothing_honest_to_compare_with_is_none_and_says_why():
    val_ces = {120: 5.30}
    value, why = tm.val_ce_for_checkpoint(val_ces, 12799, None)
    assert value is None and "ACCUMULATION_STEPS" in why, "today's config is never used to guess"
    value, why = tm.val_ce_for_checkpoint(val_ces, 1279, 128)  # opt step 10, before the first row
    assert value is None and "before the first val row" in why
    assert tm.val_ce_for_checkpoint({}, 23551, 128)[0] is None


def test_recorded_val_ces_missing_file_is_empty(runs):
    _make_run(runs, "r", meta={"parameters": {}})
    assert tm.recorded_val_ces("r") == {}


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
