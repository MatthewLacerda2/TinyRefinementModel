"""What `instruments.runlog` must never do again: read a blank cell as zero.

Every one of these is a shape of run directory that exists on disk somewhere in
`runs/` — an arch that doesn't measure half the columns, a resume that replayed
a step range, a CSV whose last row was being written when we read it, a run
folder with no metadata. The loader's job is to make all of them readable
without inventing a single number.
"""

import json

import pytest

from instruments import runlog
from trm.config import TOKENS_PER_OPT_STEP

HEADER = "step,ce,loss,val_ce,avg_forget_cost,grad_norm_avg"

# A refiner run as actually recorded: the reasoner-only column is blank on every
# row, and val_ce only appears on the eval cadence.
REFINER_ROWS = [
    "5,11.0810,22.1671,,,21.1762",
    "10,10.5584,21.1120,,,20.2026",
    "15,9.9001,19.8002,9.8123,,18.4410",
]


def write_run(tmp_path, name="run_20260101_000000", rows=REFINER_ROWS,
              header=HEADER, metadata=None):
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    body = "\n".join([header, *rows])
    (run_dir / "metrics.csv").write_text(body + ("\n" if rows else ""))
    if metadata is not None:
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    return run_dir


def test_a_blank_cell_is_absent_not_zero():
    """The bug this module was written to end. `avg_forget_cost` is empty for
    every refiner row; reading it as 0.0 manufactures a measurement, and a
    plotter then draws a flat line through a quantity nobody measured."""
    log = runlog.RunLog("r", [{"step": 5, "avg_forget_cost": None, "ce": 11.081}], {})
    assert log.metrics[0]["avg_forget_cost"] is None
    assert log.metrics[0]["avg_forget_cost"] != 0.0
    assert not log.has("avg_forget_cost")
    assert log.column("avg_forget_cost") == ([], [])


def test_reads_a_real_shaped_run(tmp_path):
    log = runlog.load(str(write_run(tmp_path) / "metrics.csv"))

    assert log.run_id == "run_20260101_000000"
    assert log.fields == HEADER.split(",")
    assert len(log.metrics) == 3

    steps, ce = log.column("ce")
    assert steps == [5, 10, 15]
    assert ce == [11.0810, 10.5584, 9.9001]

    # Sparse column: only the rows that carry a value, paired with their steps.
    assert log.has("val_ce")
    assert log.column("val_ce") == ([15], [9.8123])

    # Arch-optional column: present in the header, measured by nobody.
    assert not log.has("avg_forget_cost")

    assert log.last_step == 15
    assert log.tokens == 15 * TOKENS_PER_OPT_STEP


def test_a_placeholder_zero_column_is_distinguishable_from_a_measurement(tmp_path):
    """Runs from before #105 wrote a literal 0.0000 for quantities their arch did
    not measure. `has()` is honestly True — the number is in the file — so the
    flat-zero test is what a caller needs to skip the panel."""
    rows = ["5,11.0,22.0,,0.0000,3.0", "10,10.5,21.0,,0.0000,2.0", "15,10.0,20.0,,0.0000,1.0"]
    log = runlog.load(str(write_run(tmp_path, rows=rows)))
    assert log.has("avg_forget_cost")
    assert log.is_constant("avg_forget_cost", 0.0)
    assert not log.is_constant("grad_norm_avg", 0.0)
    assert not log.is_constant("ce")
    # A column with nothing in it is not "constant" — it is absent.
    assert not log.is_constant("val_ce", 0.0)


def test_an_unknown_column_is_empty_not_an_error(tmp_path):
    """Consumers ask about columns that older runs never had. That is a normal
    question with an empty answer, not a KeyError."""
    log = runlog.load(str(write_run(tmp_path)))
    assert not log.has("hunch_drift")
    assert log.column("hunch_drift") == ([], [])


def test_load_accepts_a_run_dir_or_a_csv_path(tmp_path):
    run_dir = write_run(tmp_path)
    from_dir = runlog.load(str(run_dir))
    from_csv = runlog.load(str(run_dir / "metrics.csv"))
    assert from_dir.csv_path == from_csv.csv_path
    assert from_dir.metrics == from_csv.metrics


def test_metadata_is_read_when_present_and_empty_when_not(tmp_path):
    plain = runlog.load(str(write_run(tmp_path, name="run_a")))
    assert plain.metadata == {}
    assert plain.wall_seconds is None

    meta = {"run_id": "run_b", "parameters": {"LATENT_DIM": 960},
            "sections": [{"duration_seconds": 100.0}, {"duration_seconds": 50.5},
                         {"duration_seconds": None}]}
    described = runlog.load(str(write_run(tmp_path, name="run_b", metadata=meta)))
    assert described.metadata["parameters"]["LATENT_DIM"] == 960
    # The in-flight session has no duration yet; it is skipped, not counted as 0.
    assert described.wall_seconds == 150.5


def test_unreadable_metadata_does_not_take_the_metrics_down(tmp_path):
    run_dir = write_run(tmp_path)
    (run_dir / "run_metadata.json").write_text("{ this is not json")
    log = runlog.load(str(run_dir))
    assert log.metadata == {}
    assert log.last_step == 15


def test_a_header_only_csv_is_an_empty_run(tmp_path):
    """A run that has started but not yet logged. Nothing to summarize, nothing
    to crash on."""
    log = runlog.load(str(write_run(tmp_path, rows=[])))
    assert log.metrics == []
    assert log.last_step == 0
    assert log.tokens == 0
    assert not log.has("ce")


def test_an_empty_file_is_an_empty_run(tmp_path):
    run_dir = write_run(tmp_path, rows=[])
    (run_dir / "metrics.csv").write_text("")
    log = runlog.load(str(run_dir))
    assert log.metrics == [] and log.fields == []


@pytest.mark.parametrize("torn", [",9.5,19.0,,,1.0", "not-a-step,9.5,19.0,,,1.0"])
def test_a_final_row_with_no_readable_step_is_dropped(tmp_path, torn):
    """A live run's CSV can be read mid-write. A row with no readable step is
    unrecoverable — dropping it is right; guessing at it is not.

    (A step truncated to a still-valid smaller number, "2" for "2750", lands in
    the monotonic filter below instead: either way it never reaches a caller.)
    """
    log = runlog.load(str(write_run(tmp_path, rows=[*REFINER_ROWS, torn])))
    assert log.last_step == 15
    assert log.torn_rows == 1


def test_a_row_torn_mid_line_keeps_the_fields_it_has(tmp_path):
    """Truncated after the loss column: step/ce/loss are real and survive; the
    columns that never arrived are absent, not zero."""
    log = runlog.load(str(write_run(tmp_path, rows=[*REFINER_ROWS, "20,9.5,19.0"])))
    assert log.last_step == 20
    assert log.metrics[-1]["ce"] == 9.5
    assert log.metrics[-1]["grad_norm_avg"] is None


def test_replayed_rows_from_a_resume_are_dropped(tmp_path):
    """Pre-trimming resumes re-logged a range they had already written. Keep the
    first (advancing) copy so the curve stays monotonic in step."""
    rows = [*REFINER_ROWS, "10,8.0,16.0,,,9.0", "20,7.5,15.0,,,8.5"]
    log = runlog.load(str(write_run(tmp_path, rows=rows)))
    assert [row["step"] for row in log.metrics] == [5, 10, 15, 20]
    assert log.replayed_rows == 1
    assert log.column("ce")[1][1] == 10.5584   # the original step-10 row, not the replay


def test_a_row_with_extra_columns_does_not_invent_a_field(tmp_path):
    log = runlog.load(str(write_run(tmp_path, rows=["5,1.0,2.0,,,3.0,999"])))
    assert set(log.metrics[0]) == set(HEADER.split(","))


def test_non_finite_values_are_kept_not_hidden(tmp_path):
    """A NaN means the run went wrong. Silently dropping it would turn a
    diverged run into a clean-looking curve with a gap."""
    log = runlog.load(str(write_run(tmp_path, rows=["5,nan,2.0,,,3.0"])))
    assert log.has("ce")
    value = log.column("ce")[1][0]
    assert value != value   # NaN


def test_a_csv_without_a_step_column_is_refused(tmp_path):
    write_run(tmp_path, header="epoch,ce", rows=["1,2.0"])
    with pytest.raises(ValueError, match="no 'step' column"):
        runlog.load(str(tmp_path / "runs" / "run_20260101_000000"))


def test_a_missing_csv_raises_something_readable(tmp_path):
    with pytest.raises(FileNotFoundError, match="no metrics CSV"):
        runlog.load(str(tmp_path / "runs" / "nope"))


def test_auto_discovery_picks_the_latest_run_that_has_metrics(tmp_path, monkeypatch):
    """Run ids are timestamps, so lexical order is chronological order. A newer
    run dir with no CSV yet (created, not started) must not shadow the last run
    that actually has data."""
    write_run(tmp_path, name="run_20260101_000000")
    write_run(tmp_path, name="run_20260202_000000")
    (tmp_path / "runs" / "run_20260303_000000").mkdir()

    monkeypatch.chdir(tmp_path)
    assert runlog.load().run_id == "run_20260202_000000"


def test_auto_discovery_says_so_when_there_is_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no run under"):
        runlog.load()


def test_the_live_run_reads_cleanly():
    """Against the real artifact, not a fixture: whatever is in runs/ right now
    must parse, and its arch-optional columns must come back absent."""
    try:
        log = runlog.load()
    except FileNotFoundError:
        pytest.skip("no runs/ on this machine")
    assert log.fields, "the run's CSV has no header"
    if not log.metrics:
        pytest.skip("latest run has no rows yet")
    assert log.has("ce") and log.last_step > 0
    for row in log.metrics:
        assert isinstance(row["step"], int)
        assert all(value is None or isinstance(value, float)
                   for key, value in row.items() if key != "step")
