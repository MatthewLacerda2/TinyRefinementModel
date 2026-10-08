"""run_metadata.json is read through one typed model (#578)."""

import json
import pathlib

import pytest

from trm.runtime.run_metadata import METADATA_FILENAME, RunMetadata, Section

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Every era this machine still holds; a clone without runs/ has none and skips.
LOCAL = sorted(ROOT.glob("runs/**/" + METADATA_FILENAME))

TODAYS = {"run_id": "run_x", "git_commit": "abc", "git_branch": "main", "git_dirty": False,
          "parameters": {"LATENT_DIM": 960, "TRAIN_TOKEN_BUDGET": None},
          "sections": [{"start_time": "2026-10-08T01:00:00-03:00", "end_time": None, "duration_seconds": None}]}


def _write(run_dir, payload):
    (run_dir / METADATA_FILENAME).write_text(payload if isinstance(payload, str) else json.dumps(payload))


def test_todays_file_reads_and_round_trips(tmp_path):
    _write(tmp_path, TODAYS)
    metadata = RunMetadata.read(tmp_path)
    assert metadata is not None and metadata.parameters["LATENT_DIM"] == 960
    metadata.sections.append(Section(start_time="2026-10-08T02:00:00-03:00", duration_seconds=3600.0))
    metadata.write(tmp_path)
    again = RunMetadata.read(tmp_path)
    assert again == metadata and again.hours == 1.0


def test_missing_or_torn_is_none_never_an_exception(tmp_path):
    """RunTracker rewrites the file in place, so a reader beside training can catch it torn."""
    assert RunMetadata.read(tmp_path) is None
    _write(tmp_path, '{"parameters": {"LATENT')
    assert RunMetadata.read(tmp_path) is None


def test_a_misspelled_key_fails_naming_it(tmp_path):
    _write(tmp_path, {**TODAYS, "git_comit": "abc"})
    with pytest.raises(ValueError, match=r"\[git_comit\] Extra inputs"):
        RunMetadata.read(tmp_path)


def test_a_wrong_type_fails_naming_it(tmp_path):
    _write(tmp_path, {**TODAYS, "sections": [{"start_time": "x", "duration_seconds": "an hour"}]})
    with pytest.raises(ValueError, match=r"\[sections\.0\.duration_seconds\]"):
        RunMetadata.read(tmp_path)


@pytest.mark.skipif(not LOCAL, reason="no runs/ on this clone")
@pytest.mark.parametrize("path", LOCAL, ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_every_recorded_run_reads(path):
    assert RunMetadata.read(path.parent) is not None
