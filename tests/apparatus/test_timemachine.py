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


# Val rows copied by hand from runs/run_287_adamw_lr0.0001_s0/metrics.csv, around its
# checkpoints. That run probes every 8 opt steps and logs a row every 5, so each probe's
# value lands on the first multiple of 5 at or after it: probe 56 -> row 60, 64 -> 65,
# 176 -> 180, 184 -> 185, 248 -> 250, 256 -> 260, 264 -> 265, 504 -> 505, 512 -> 515.
RUN_287_VAL_ROWS = "step,val_ce\n60,7.8468\n65,7.6431\n180,6.5459\n185,6.4883\n250,6.1998\n" \
                   "260,6.1670\n265,6.1478\n505,5.8352\n515,5.8346\n"
RUN_287_ACCUMULATION = 128


@pytest.mark.parametrize("checkpoint, opt_step, row, value, previous_probe", [
    (8191, 64, 65, 7.6431, 7.8468),    # the first checkpoint
    (23551, 184, 185, 6.4883, 6.5459),
    (32767, 256, 260, 6.1670, 6.1998),
    (65535, 512, 515, 5.8346, 5.8352),  # the latest
])
def test_a_real_checkpoint_reads_its_own_probe_from_run_287(runs, checkpoint, opt_step, row, value, previous_probe):
    """#348: orbax names checkpoints by micro-step, metrics.csv by opt step. #351: a probe's
    val CE is written on the next logging row, not its own step. None of run_287's
    checkpoints has a row at its exact opt step, so matching either way failed on real
    data: first against opt-step keys that micro-steps never hit, then against the
    previous probe's row."""
    _make_run(runs, "r", meta={"parameters": {"ACCUMULATION_STEPS": RUN_287_ACCUMULATION}},
              metrics=RUN_287_VAL_ROWS)
    val_ces = tm.recorded_val_ces("r")
    accumulation = tm.runlog.recorded_params(tm.load_meta("r"))["ACCUMULATION_STEPS"]
    assert tm.checkpoint_opt_step(checkpoint, accumulation) == opt_step
    assert opt_step not in val_ces, "run_287 never logs a val row at a checkpoint's own opt step"
    got, how = tm.val_ce_for_checkpoint(val_ces, checkpoint, accumulation)
    assert got == pytest.approx(value) and got != pytest.approx(previous_probe)
    assert f"row {row}" in how


def test_the_first_checkpoint_matches_even_with_no_earlier_row():
    """The old fallback returned None when no row sat at or before the checkpoint, so a
    CSV whose first val row is the first probe's (row 65 for opt 64) lost it."""
    got, _ = tm.val_ce_for_checkpoint({65: 7.6431, 185: 6.4883}, 8191, RUN_287_ACCUMULATION)
    assert got == pytest.approx(7.6431)


def test_no_row_in_the_window_is_none_never_another_probe():
    """run_287 has no val row in [190, 195): probe 192 is on row 195. Checkpoint 24319 is
    opt 190, and returning row 185 or 195 would compare the weights with another probe."""
    val_ces = {180: 6.5459, 185: 6.4883, 195: 6.4487}  # run_287's rows; row 190 is blank there
    got, why = tm.val_ce_for_checkpoint(val_ces, 24319, RUN_287_ACCUMULATION)
    assert got is None and "[190, 195)" in why


def test_nothing_to_place_the_checkpoint_by_is_none_and_says_why():
    got, why = tm.val_ce_for_checkpoint({65: 7.6431}, 8191, None)
    assert got is None and "ACCUMULATION_STEPS" in why, "today's config is never used to guess"
    assert tm.val_ce_for_checkpoint({}, 8191, RUN_287_ACCUMULATION)[0] is None


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
