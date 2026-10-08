"""Resuming from an earlier checkpoint is one mechanical act, and a refused save is loud (#188)."""

import ast
import datetime
import json
import pathlib

import pytest

from trm.runtime import rewind as rw
from trm.runtime.layout import BEST_SUBDIR
from trm.settings import CONFIG

ACCUM = 128


def _ckpt(directory, opt_step, finalized=True):
    step = opt_step * ACCUM - 1
    path = directory / str(step)
    path.mkdir(parents=True)
    if finalized:
        (path / "_CHECKPOINT_METADATA").write_text("{}")
    return path


def test_listing_reads_opt_steps_from_disk(tmp_path):
    """The #157 recovery, as a listing: finalized checkpoints only, oldest first."""
    for opt in (4928, 4992, 5056):
        _ckpt(tmp_path, opt)
    _ckpt(tmp_path, 5120, finalized=False)
    found = rw.checkpoints_in(tmp_path, ACCUM)
    assert [(c.step, c.opt_step) for c in found] == [(630783, 4928), (638975, 4992), (647167, 5056)]


def test_a_refused_resume_leaves_the_run_untouched(tmp_path):
    """The on-disk check fires on the newest finalized checkpoint and writes
    nothing: run_metadata.json is byte-for-byte what it was."""
    run = tmp_path / "run_X"
    metadata = run / "run_metadata.json"
    run.mkdir()
    metadata.write_text('{"sessions": [{"start": "2026-08-15"}]}\n')
    before = metadata.read_bytes()
    checkpoints = run / "checkpoints"
    _ckpt(checkpoints, 4992)
    newest = _ckpt(checkpoints, 5120)
    (newest / "monitor_state").mkdir()
    (newest / "monitor_state" / "metadata").write_text(json.dumps({**PRETRAINING, "sampels_seen": 5}))
    _ckpt(checkpoints, 5184, finalized=False)   # a torn write is not what a resume loads

    assert "sampels_seen" in rw.unresumable(checkpoints, ACCUM)
    assert metadata.read_bytes() == before


# The smallest resume state ResumeState accepts: its fields without a default.
PRETRAINING = {"run_id": "run_X", "ce_history": [3.4], "best_ce": 3.1, "best_loss": 3.2,
               "best_avg_ce": 3.4, "last_improvement_step": 100}


def _with_state(directory, opt_step, state):
    path = _ckpt(directory, opt_step)
    (path / "monitor_state").mkdir()
    (path / "monitor_state" / "metadata").write_text(json.dumps(state))
    return path


def test_a_resume_state_resumestate_refuses_is_refused_from_disk_naming_the_field(tmp_path):
    """#505: the misspelled key is named before a launch, not after a session."""
    _with_state(tmp_path, 4992, PRETRAINING)
    _with_state(tmp_path, 5056, {**PRETRAINING, "sampels_seen": 5})
    why = rw.unresumable(tmp_path, ACCUM)
    assert "sampels_seen" in why and f"checkpoint step {5056 * ACCUM - 1}" in why


def test_a_pretraining_checkpoint_dir_passes_the_launch_check(tmp_path):
    _ckpt(tmp_path, 4992)
    assert rw.unresumable(tmp_path, ACCUM) is None             # no monitor state: left to restore
    assert rw.unresumable(tmp_path / "absent", ACCUM) is None  # a fresh run
    _with_state(tmp_path, 5056, PRETRAINING)
    assert rw.unresumable(tmp_path, ACCUM) is None


def test_the_trainer_refuses_before_it_starts_a_session():
    """Read, not imported (start.py pulls in jax): the on-disk check must come
    before RunTracker.start_session, which appends to run_metadata.json."""
    source = (pathlib.Path(__file__).parents[2] / "trm" / "train" / "start.py").read_text()
    lines = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            lines[name] = min(lines.get(name, node.lineno), node.lineno)
    assert lines["unresumable"] < lines["start_session"]


def test_rewind_sets_aside_newer_checkpoints_in_both_dirs_and_deletes_nothing(tmp_path):
    for opt in (4928, 4992, 5056):
        _ckpt(tmp_path, opt)
    _ckpt(tmp_path / BEST_SUBDIR, 4928)
    _ckpt(tmp_path / BEST_SUBDIR, 5056)

    chosen, moved = rw.rewind(tmp_path, 5000, ACCUM, now=datetime.datetime(2026, 9, 13))
    assert chosen.opt_step == 4992, "the newest at or below the request"
    assert [c.opt_step for c in rw.checkpoints_in(tmp_path, ACCUM)] == [4928, 4992]
    assert [c.opt_step for c in rw.checkpoints_in(tmp_path / BEST_SUBDIR, ACCUM)] == [4928]
    assert len(moved) == 2 and all(p.exists() for p in moved), "set aside, never deleted"
    assert all(rw.SET_ASIDE_PREFIX in str(p) for p in moved)


def test_rewinding_twice_is_harmless(tmp_path):
    """The reason this is a command and not a trainer flag: a replayed rewind must
    not keep throwing progress away. Once nothing newer remains, it moves nothing."""
    for opt in (4928, 4992):
        _ckpt(tmp_path, opt)
    rw.rewind(tmp_path, 4928, ACCUM)
    _, moved = rw.rewind(tmp_path, 4928, ACCUM)
    assert moved == []


def test_no_checkpoint_at_or_below_is_an_error_not_a_fallback(tmp_path):
    _ckpt(tmp_path, 4992)
    with pytest.raises(SystemExit, match="have opt steps: 4992"):
        rw.rewind(tmp_path, 100, ACCUM)


def test_orbax_refusing_a_save_is_an_error_not_silence(tmp_path, tiny_model):
    """orbax.save() returns False, without a word, for a step below its newest.
    That is how a naive resume-from-earlier would have run for days uncheckpointed."""
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx
    from trm.runtime.checkpoints import save_checkpoint
    from trm.runtime.layout import CHECKPOINT_ITEMS, ROLLING_KEEP
    from trm.runtime.monitor import LossMonitor

    optimizer = nnx.Optimizer(tiny_model, optax.sgd(0.0), wrt=nnx.Param)
    mngr = ocp.CheckpointManager(str(tmp_path), item_names=CHECKPOINT_ITEMS,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=ROLLING_KEEP, create=True))
    save_checkpoint(mngr, 300, tiny_model, optimizer, LossMonitor.of(CONFIG), "run_x")
    with pytest.raises(RuntimeError, match="trm.runtime.rewind"):
        save_checkpoint(mngr, 200, tiny_model, optimizer, LossMonitor.of(CONFIG), "run_x")
