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
RUN_287_VAL_EVERY = 8


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
    _make_run(runs, "r", metrics=RUN_287_VAL_ROWS, meta={"parameters": {
        "ACCUMULATION_STEPS": RUN_287_ACCUMULATION, "VAL_EVERY_OPT_STEPS": RUN_287_VAL_EVERY}})
    val_ces = tm.recorded_val_ces("r")
    params = tm.runlog.recorded_params(tm.load_meta("r"))
    assert tm.checkpoint_opt_step(checkpoint, params["ACCUMULATION_STEPS"]) == opt_step
    assert opt_step not in val_ces, "run_287 never logs a val row at a checkpoint's own opt step"
    got, how = tm.val_ce_for_checkpoint(val_ces, checkpoint, params["ACCUMULATION_STEPS"],
                                        params["VAL_EVERY_OPT_STEPS"])
    assert got == pytest.approx(value) and got != pytest.approx(previous_probe)
    assert f"row {row}" in how


def test_the_first_checkpoint_matches_even_with_no_earlier_row():
    """The old fallback returned None when no row sat at or before the checkpoint, so a
    CSV whose first val row is the first probe's (row 65 for opt 64) lost it."""
    got, _ = tm.val_ce_for_checkpoint({65: 7.6431, 185: 6.4883}, 8191, RUN_287_ACCUMULATION, RUN_287_VAL_EVERY)
    assert got == pytest.approx(7.6431)


def _micro(opt_step, accumulation=RUN_287_ACCUMULATION):
    """The checkpoint name orbax gives an opt step: the inverse of checkpoint_opt_step."""
    return opt_step * accumulation - 1


@pytest.mark.parametrize("val_every, opt_step, rows, other_probe", [
    # Simulated cadences (values illustrative). Probes every 10: row 260 holds probe 260,
    # and opt 256 is no probe at all.
    (10, 256, {250: 6.20, 260: 6.17}, "260"),
    # Probes every 3: row 65 holds probe 63 (the last in (60, 65]), not opt 64.
    (3, 64, {60: 7.85, 65: 7.64}, "63"),
    # The fit gate probes every opt step: row 10 holds probe 10, not opt 7.
    (1, 7, {5: 10.9, 10: 10.7}, "10"),
])
def test_a_row_that_holds_another_probe_is_never_reported_as_this_ones(val_every, opt_step, rows, other_probe):
    """#350 review: the window rule assumed the checkpoint is a probe step and at most one
    probe lands in a logging interval. The trainer overwrites the held value, so a row holds
    the last probe in (row - 5, row]. Each case here would have read probe `other_probe`'s
    value and called it this checkpoint's."""
    got, why = tm.val_ce_for_checkpoint(rows, _micro(opt_step), RUN_287_ACCUMULATION, val_every)
    assert got is None, f"row value belongs to probe {other_probe}"
    assert "not a probe step" in why or "closer than" in why


def test_a_probe_step_with_no_row_in_its_window_is_none_never_another_probe():
    """Probe 192 of run_287 is logged on row 195. With that row missing (a torn final row
    is dropped), row 185 or 200 would compare the weights with another probe."""
    got, why = tm.val_ce_for_checkpoint({185: 6.4883, 200: 6.4206}, _micro(192), RUN_287_ACCUMULATION,
                                        RUN_287_VAL_EVERY)
    assert got is None and "[192, 197)" in why


def test_nothing_to_tie_the_checkpoint_by_is_none_and_says_why():
    got, why = tm.val_ce_for_checkpoint({65: 7.6431}, 8191, None, RUN_287_VAL_EVERY)
    assert got is None and "ACCUMULATION_STEPS" in why, "today's config is never used to guess"
    got, why = tm.val_ce_for_checkpoint({65: 7.6431}, 8191, RUN_287_ACCUMULATION, None)
    assert got is None and "VAL_EVERY_OPT_STEPS" in why
    assert tm.val_ce_for_checkpoint({}, 8191, RUN_287_ACCUMULATION, RUN_287_VAL_EVERY)[0] is None


@pytest.mark.parametrize("expected, measured, code, word", [
    (6.4883, 6.5100, 0, "REPRODUCED"),
    (6.4883, 6.6000, 1, "DRIFTED"),
    (None, 6.5100, 2, "cannot verify"),
])
def test_the_exit_code_is_zero_only_for_a_value_within_the_noise_floor(expected, measured, code, word):
    """The DoD gate reads the exit code: an un-checkable run must not read as reproduced."""
    got_code, verdict = tm.reproduction_verdict(expected, measured, tolerance=0.06)
    assert got_code == code and word in verdict


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
