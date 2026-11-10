"""A resumed run must still describe the environment it ran in.

The registry doctrine treats the recipe as the irreplaceable artifact — commit,
config, seed, and the environment that produced the weights. `capture_environment_snapshot`
writes the last part. It used to run only for brand-new runs, which meant the
*correct* way to launch a supervised long run produced none of it: pinning
`--checkpoint-path` sends start.py down its resume branch, and `--new-run` is not
an option because a crash-relaunch would then start a fresh run from scratch and
throw away days of work.

Nothing failed loudly. run_metadata.json is written either way, so the run dir
looked populated and the omission surfaced only when someone tried to revive the
weights.
"""

import json

from trm.runtime.run_metadata import RunMetadata
from trm.runtime.run_tracker import RunTracker
from trm.settings import CONFIG

ARTIFACTS = ("env_freeze.txt", "system_snapshot.txt", "worktree.patch",
             "worktree.untracked.txt")


def test_a_resumed_session_still_captures_the_environment(tmp_path):
    tracker = RunTracker(CONFIG, runs_root=str(tmp_path))
    run_id = tracker.start_session()
    for name in ARTIFACTS:
        (tmp_path / run_id / name).unlink(missing_ok=True)

    resumed = RunTracker(CONFIG, runs_root=str(tmp_path))
    resumed.start_session(run_id=run_id)

    missing = [n for n in ARTIFACTS if not (tmp_path / run_id / n).exists()]
    assert not missing, f"a resumed run must still be revivable; missing {missing}"


def test_resuming_appends_a_session_rather_than_replacing_the_record(tmp_path):
    """Re-capturing must not cost the run its history — the sections list is how
    a multi-session run accounts for its own wall-clock."""
    tracker = RunTracker(CONFIG, runs_root=str(tmp_path))
    run_id = tracker.start_session()
    RunTracker(CONFIG, runs_root=str(tmp_path)).start_session(run_id=run_id)

    meta = json.loads((tmp_path / run_id / "run_metadata.json").read_text())
    assert len(meta["sections"]) == 2
    assert meta["run_id"] == run_id


# --- the resume check compares what the model's tree is built from (#317) ---


def _resume_with(tmp_path, **recorded):
    """'refused' or 'resumed': a resume onto a run whose metadata recorded
    today's parameters with `recorded` changed (None: the key was never recorded)."""
    params = {**RunTracker.get_hyperparameters(CONFIG), **recorded}
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {k: v for k, v in params.items() if v is not None}}))
    try:
        RunTracker(CONFIG, runs_root=str(tmp_path))._check_compatibility(RunMetadata.read(tmp_path))
    except SystemExit:
        return "refused"
    return "resumed"


def test_an_unchanged_resume_goes_through(tmp_path):
    assert _resume_with(tmp_path) == "resumed"


def test_a_resume_with_a_different_layer_count_refuses(tmp_path):
    assert _resume_with(tmp_path, PLAIN_LAYERS=CONFIG.PLAIN_LAYERS + 1) == "refused"


def test_a_refusal_is_raised_with_its_guidance_not_exited(tmp_path, monkeypatch):
    """#324: a library method raises; the message names the mismatch and the ways out,
    and a string SystemExit left uncaught still exits the trainer with code 1."""
    import pytest

    from trm.runtime import run_tracker
    PLAIN_LAYERS = CONFIG.PLAIN_LAYERS
    monkeypatch.setattr(run_tracker.sys, "exit", lambda *a: pytest.fail("sys.exit called"))
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {**RunTracker.get_hyperparameters(CONFIG),
                                               "PLAIN_LAYERS": PLAIN_LAYERS + 1}}))
    with pytest.raises(SystemExit) as refused:
        RunTracker(CONFIG, runs_root=str(tmp_path))._check_compatibility(RunMetadata.read(tmp_path))
    message = refused.value.code
    assert isinstance(message, str), "a str code is what makes the uncaught exit status 1"
    assert f"PLAIN_LAYERS: run used {PLAIN_LAYERS + 1}" in message and "--new-run" in message


def test_metadata_that_predates_a_key_is_skipped_not_refused(tmp_path):
    """POST_NORM was first recorded by #317; every older run lacks it."""
    assert _resume_with(tmp_path, POST_NORM=None) == "resumed"


# --- a recipe default that moved since the run started is refused, not adopted (#535) ---


def _resume_as(tmp_path, config, **recorded):
    """_resume_with, but resolving the launch's Config as `config` (its model_fields_set
    is what the launch asked for)."""
    params = {**RunTracker.get_hyperparameters(CONFIG), **recorded}
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {k: v for k, v in params.items() if v is not None}}))
    try:
        RunTracker(config, runs_root=str(tmp_path))._check_compatibility(RunMetadata.read(tmp_path))
    except SystemExit as refused:
        return "refused: " + str(refused.code)
    return "resumed"


def test_a_recipe_default_that_moved_refuses_and_names_the_knob(tmp_path):
    outcome = _resume_as(tmp_path, CONFIG, ADAM_B2=CONFIG.ADAM_B2 + 0.01)
    assert outcome.startswith("refused") and "ADAM_B2" in outcome and "on purpose" in outcome


def test_a_recipe_knob_set_for_the_launch_resumes_on_purpose(tmp_path, capsys):
    asked = CONFIG.model_copy(update={"ADAM_B2": CONFIG.ADAM_B2})
    assert _resume_as(tmp_path, asked, ADAM_B2=CONFIG.ADAM_B2 + 0.01) == "resumed"
    assert "changed on purpose" in capsys.readouterr().out


def test_a_derived_value_follows_the_knob_it_comes_from(tmp_path):
    assert _resume_as(tmp_path, CONFIG, DECAY_STEPS=1).startswith("refused")
    assert _resume_as(tmp_path, CONFIG.model_copy(update={"TRAIN_TOKEN_BUDGET": CONFIG.TRAIN_TOKEN_BUDGET}), DECAY_STEPS=1) == "resumed"


def test_the_tree_never_changes_even_when_asked(tmp_path):
    asked = CONFIG.model_copy(update={"PLAIN_LAYERS": CONFIG.PLAIN_LAYERS})
    assert _resume_as(tmp_path, asked, PLAIN_LAYERS=CONFIG.PLAIN_LAYERS + 1).startswith("refused")


def test_a_knob_the_code_no_longer_has_is_skipped(tmp_path):
    """#546 removed MODEL_ARCH; older plain runs recorded it."""
    assert _resume_with(tmp_path, MODEL_ARCH="plain") == "resumed"
