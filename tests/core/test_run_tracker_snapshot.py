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

ARTIFACTS = ("env_freeze.txt", "system_snapshot.txt", "worktree.patch",
             "worktree.untracked.txt")


def test_a_resumed_session_still_captures_the_environment(tmp_path):
    tracker = RunTracker(runs_root=str(tmp_path))
    run_id = tracker.start_session()
    for name in ARTIFACTS:
        (tmp_path / run_id / name).unlink(missing_ok=True)

    resumed = RunTracker(runs_root=str(tmp_path))
    resumed.start_session(run_id=run_id)

    missing = [n for n in ARTIFACTS if not (tmp_path / run_id / n).exists()]
    assert not missing, f"a resumed run must still be revivable; missing {missing}"


def test_resuming_appends_a_session_rather_than_replacing_the_record(tmp_path):
    """Re-capturing must not cost the run its history — the sections list is how
    a multi-session run accounts for its own wall-clock."""
    tracker = RunTracker(runs_root=str(tmp_path))
    run_id = tracker.start_session()
    RunTracker(runs_root=str(tmp_path)).start_session(run_id=run_id)

    meta = json.loads((tmp_path / run_id / "run_metadata.json").read_text())
    assert len(meta["sections"]) == 2
    assert meta["run_id"] == run_id


# --- the resume check compares what the selected arch's tree is built from (#317) ---

def _resume_with(tmp_path, monkeypatch, **recorded):
    """'refused' or 'resumed': a plain resume onto a run whose metadata recorded
    today's parameters with `recorded` changed (None: the key was never recorded)."""
    from trm.runtime import run_tracker
    monkeypatch.setattr(run_tracker, "MODEL_ARCH", "plain")
    params = {**RunTracker.get_hyperparameters(), **recorded}
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"parameters": {k: v for k, v in params.items() if v is not None}}))
    try:
        RunTracker(runs_root=str(tmp_path))._check_compatibility(str(path))
    except SystemExit:
        return "refused"
    return "resumed"


def test_an_unchanged_plain_resume_goes_through(tmp_path, monkeypatch):
    assert _resume_with(tmp_path, monkeypatch) == "resumed"


def test_a_plain_resume_with_a_different_layer_count_refuses(tmp_path, monkeypatch):
    from trm.config import PLAIN_LAYERS
    assert _resume_with(tmp_path, monkeypatch, PLAIN_LAYERS=PLAIN_LAYERS + 1) == "refused"


def test_a_resume_under_another_arch_refuses(tmp_path, monkeypatch):
    assert _resume_with(tmp_path, monkeypatch, MODEL_ARCH="refiner") == "refused"


def test_a_knob_only_retired_arches_read_does_not_refuse_a_plain_resume(tmp_path, monkeypatch):
    from trm.config import NUM_BLOCKS, SHARED_SLOTS
    assert _resume_with(tmp_path, monkeypatch, NUM_BLOCKS=NUM_BLOCKS + 1,
                        SHARED_SLOTS=SHARED_SLOTS * 2) == "resumed"


def test_metadata_that_predates_a_key_is_skipped_not_refused(tmp_path, monkeypatch):
    """POST_NORM was first recorded by #317; every older run lacks it."""
    assert _resume_with(tmp_path, monkeypatch, POST_NORM=None) == "resumed"
