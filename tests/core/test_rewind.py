"""Resuming from an earlier checkpoint is one mechanical act, and a refused save is loud (#188)."""

import datetime

import pytest

from trm.runtime import rewind as rw
from trm.runtime.layout import BEST_SUBDIR

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


# --- a checkpoint from the retired SFT phase is refused, not resumed (#323) ----

def test_a_pretraining_monitor_state_resumes_with_or_without_the_legacy_phase_fields():
    """Checkpoints written before #323 carry sft_active/sft_start_step; new ones
    carry neither. Both read as pretraining."""
    rw.refuse_sft_phase_resume({"samples_seen": 5}, 638975, "ckpts", ACCUM)
    rw.refuse_sft_phase_resume({"sft_active": False, "sft_start_step": None}, 638975, "ckpts", ACCUM)


def test_an_sft_phase_monitor_state_is_refused_and_names_the_rewind():
    """A flip at micro-step 647,167 (opt step 5,056). Resuming a checkpoint past
    it as pretraining would silently change the mixture and the LR, so the loader
    stops and points at the last clean opt step."""
    with pytest.raises(SystemExit) as refused:
        rw.refuse_sft_phase_resume({"sft_active": True, "sft_start_step": 647167},
                                   655359, "runs/run_X/checkpoints", ACCUM)
    message = str(refused.value)
    assert "655359" in message and "647167" in message
    assert "python -m trm.runtime.rewind runs/run_X/checkpoints --to-opt-step 5056" in message


def test_the_suggested_rewind_lands_on_the_last_pretraining_checkpoint(tmp_path):
    """The checkpoint saved on the flip's own boundary was written before the flip,
    so it is clean; the suggested opt step must select it and nothing later."""
    for opt in (4992, 5056, 5120):
        _ckpt(tmp_path, opt)
    flip_micro_step = 5056 * ACCUM - 1
    with pytest.raises(SystemExit) as refused:
        rw.refuse_sft_phase_resume({"sft_start_step": flip_micro_step}, 5120 * ACCUM - 1, tmp_path, ACCUM)
    to_opt_step = int(str(refused.value).rsplit("--to-opt-step ", 1)[1])
    chosen = rw.resolve(rw.checkpoints_in(tmp_path, ACCUM), to_opt_step)
    assert chosen.step == flip_micro_step


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
    save_checkpoint(mngr, 300, tiny_model, optimizer, LossMonitor(), "run_x")
    with pytest.raises(RuntimeError, match="trm.runtime.rewind"):
        save_checkpoint(mngr, 200, tiny_model, optimizer, LossMonitor(), "run_x")
