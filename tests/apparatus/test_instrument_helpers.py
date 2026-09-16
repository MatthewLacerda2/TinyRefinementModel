"""One copy of each helper the instruments share, each with one failure policy (#319).

`instruments/_common.py` holds the small ones and `instruments/runlog.py` the readers of
a run's recorded artefacts. These pin the policies each docstring states, because the
whole point of one copy is that callers can rely on what it does when the thing asked
for is not there. Pure Python: nothing here imports jax.
"""

import argparse
import json
import os

import pytest

from instruments import _common, runlog


# ── _common ──────────────────────────────────────────────────────────────────

def test_git_head_is_none_outside_a_repository(tmp_path):
    assert _common.git_head(cwd=tmp_path) is None


def test_git_head_reads_this_checkout():
    short, full = _common.git_head(), _common.git_head(short=False)
    assert short and full and full.startswith(short) and len(full) == 40


def test_gpu_query_is_none_when_nvidia_smi_is_missing(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert _common.gpu_memory_used_mib() is None


def test_module_env_puts_the_target_tree_first_and_keeps_the_rest(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    env = _common.module_env("/revived/worktree", MODEL_ARCH="refiner", FORCE_F32_COMPUTE=1)
    assert env["PYTHONPATH"].split(os.pathsep) == ["/revived/worktree", "/somewhere/else"]
    assert env["MODEL_ARCH"] == "refiner" and env["FORCE_F32_COMPUTE"] == "1"
    monkeypatch.delenv("PYTHONPATH")
    assert _common.module_env("/x")["PYTHONPATH"] == "/x"


def test_one_checkpoint_flag_keeps_the_old_spellings_working():
    """latents and paired took `--checkpoint`, milestone_report `--ckpt`; findings and
    specs quote those commands, so the aliases must still parse to the same field."""
    for spelling in ("--checkpoint-path", "--checkpoint", "--ckpt"):
        parser = _common.add_checkpoint_argument(argparse.ArgumentParser(), aliases=("--checkpoint", "--ckpt"))
        assert parser.parse_args([spelling, "runs/r/checkpoints"]).checkpoint_path == "runs/r/checkpoints"
    required = _common.add_checkpoint_argument(argparse.ArgumentParser(), required=True)
    with pytest.raises(SystemExit):
        required.parse_args([])


# ── runlog's artefact readers ────────────────────────────────────────────────

def test_metadata_that_is_missing_or_half_written_reads_as_empty(tmp_path):
    """RunTracker rewrites run_metadata.json in place, so a reader beside training can
    catch it torn. That is `{}`, the same as no file, never an exception."""
    assert runlog.read_metadata(tmp_path) == {}
    (tmp_path / "run_metadata.json").write_text('{"parameters": {"MODEL_A')
    assert runlog.read_metadata(tmp_path) == {}
    (tmp_path / "run_metadata.json").write_text(json.dumps({"parameters": {"MODEL_ARCH": "plain"}}))
    assert runlog.recorded_params(runlog.read_metadata(tmp_path)) == {"MODEL_ARCH": "plain"}


def test_tokens_per_opt_step_come_from_the_recipe_or_not_at_all():
    params = {"ACCUMULATION_STEPS": 128, "BATCH_SIZE": 1, "MAX_SEQ_LEN": 512}
    assert runlog.recorded_tokens_per_opt_step(params) == 128 * 1 * 2 * 512
    assert runlog.recorded_tokens_per_opt_step({"ACCUMULATION_STEPS": 128}) is None, \
        "a partial recipe is not guessed at; the caller picks and names its fallback"


def test_checkpoint_steps_are_the_numeric_dirs_only(tmp_path):
    for name in ("64", "128", "128.orbax-checkpoint-tmp-9", "best_val_ce"):
        (tmp_path / name).mkdir()
    assert runlog.checkpoint_steps(tmp_path) == [64, 128]
    assert runlog.checkpoint_steps(tmp_path / "missing") == []


def test_an_absent_column_is_blamed_on_the_architecture_only_when_it_is():
    assert runlog.absence_reason("plain", ["tau"]) == runlog.NOT_MEASURED
    assert runlog.absence_reason("plain", ["grad_norm_avg"]) == runlog.NOT_LOGGED, \
        "plain logs a grad norm; a run missing it did not log it, and the arch is not why"
    assert runlog.absence_reason("reasoner", ["tau"]) == runlog.NOT_LOGGED
    assert runlog.absence_reason(None, ["grad_norm_avg"]) == runlog.NOT_MEASURED, \
        "a run that recorded no arch gives nothing to judge by"


def test_depth_avg_is_a_measurement_only_for_an_arch_with_a_depth_dial():
    """From #316 plain leaves depth_avg blank; before it, plain logged a sampled value the
    model ignored. Either way it is not plain's measurement, and a blank one is not news."""
    assert not runlog.measured_by("plain", "depth_avg")
    assert runlog.measured_by("refiner", "depth_avg") and runlog.measured_by("reasoner", "depth_avg")
    assert runlog.absence_reason("plain", ["depth_avg"]) == runlog.NOT_MEASURED
    assert runlog.absence_reason("refiner", ["depth_avg"]) == runlog.NOT_LOGGED
