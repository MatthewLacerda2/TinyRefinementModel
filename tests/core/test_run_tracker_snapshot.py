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


# --- the resume check compares what the selected arch's tree is built from (#317) ---

# This process's config, as a plain run whatever MODEL_ARCH the shell set.
PLAIN = CONFIG.model_copy(update={"MODEL_ARCH": "plain"})


def _resume_with(tmp_path, **recorded):
    """'refused' or 'resumed': a plain resume onto a run whose metadata recorded
    today's parameters with `recorded` changed (None: the key was never recorded)."""
    params = {**RunTracker.get_hyperparameters(PLAIN), **recorded}
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {k: v for k, v in params.items() if v is not None}}))
    try:
        RunTracker(PLAIN, runs_root=str(tmp_path))._check_compatibility(str(path))
    except SystemExit:
        return "refused"
    return "resumed"


def test_an_unchanged_plain_resume_goes_through(tmp_path):
    assert _resume_with(tmp_path) == "resumed"


def test_a_plain_resume_with_a_different_layer_count_refuses(tmp_path):
    assert _resume_with(tmp_path, PLAIN_LAYERS=PLAIN.PLAIN_LAYERS + 1) == "refused"


def test_a_plain_resume_that_switches_position_encoding_refuses(tmp_path):
    """#444: NoPE adds and removes no parameter, so the param tree alone does not tell
    the arms apart; the restore would refuse later, on RoPE's saved tables, with a far
    less readable orbax error. This refuses first, naming the knob."""
    other = "nope" if PLAIN.POSITION_ENCODING == "rope" else "rope"
    assert _resume_with(tmp_path, POSITION_ENCODING=other) == "refused"


def test_a_refusal_is_raised_with_its_guidance_not_exited(tmp_path, monkeypatch):
    """#324: a library method raises; the message names the mismatch and the ways out,
    and a string SystemExit left uncaught still exits the trainer with code 1."""
    import pytest
    from trm.runtime import run_tracker
    PLAIN_LAYERS = PLAIN.PLAIN_LAYERS
    monkeypatch.setattr(run_tracker.sys, "exit", lambda *a: pytest.fail("sys.exit called"))
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {**RunTracker.get_hyperparameters(PLAIN),
                                               "PLAIN_LAYERS": PLAIN_LAYERS + 1}}))
    with pytest.raises(SystemExit) as refused:
        RunTracker(PLAIN, runs_root=str(tmp_path))._check_compatibility(str(path))
    message = refused.value.code
    assert isinstance(message, str), "a str code is what makes the uncaught exit status 1"
    assert f"PLAIN_LAYERS: run used {PLAIN_LAYERS + 1}" in message and "--new-run" in message


def test_a_resume_under_another_arch_refuses(tmp_path):
    assert _resume_with(tmp_path, MODEL_ARCH="refiner") == "refused"


def test_a_knob_only_retired_arches_read_does_not_refuse_a_plain_resume(tmp_path):
    from trm.config import NUM_BLOCKS, SHARED_SLOTS
    assert _resume_with(tmp_path, NUM_BLOCKS=NUM_BLOCKS + 1, SHARED_SLOTS=SHARED_SLOTS * 2) == "resumed"


def test_metadata_that_predates_a_key_is_skipped_not_refused(tmp_path):
    """POST_NORM was first recorded by #317; every older run lacks it."""
    assert _resume_with(tmp_path, POST_NORM=None) == "resumed"
