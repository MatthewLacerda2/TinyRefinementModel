"""metrics.csv carries what a row cannot be read without (#186), and writes every
column it declares.

The second half is not hypothetical: `act_max` sat in the header from #252 on, and
`log()` never filled it — the f16-margin telemetry existed as an always-empty
column, and the invariant reading it could never fire.
"""

import csv
import datetime
from types import SimpleNamespace

import jax.numpy as jnp

from trm.runtime.metrics import MetricsLogger
from trm.settings import CONFIG
from trm.train.schedules import Schedules, mixture_label

PRETRAIN_SOURCES = Schedules.of(CONFIG).sources


def _log_one(tmp_path, **overrides):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(str(path))
    out = SimpleNamespace(diag={k: jnp.array(1.5) for k in logger.diag_keys})
    kwargs = dict(grad_norm_avg=0.5, seg1_ce=3.0, val_ce=3.1,
                  zero_frac_dense_max=0.0, applied_zero_frac_dense_max=0.0, applied_grad_norm=0.7, clip_active=0, val_step=8, val_by_source="codeparrot=2.5", loss_scale="65536", skipped_micro_steps=3, grad_by_source="fineweb-edu=1.0/2.0/0/5",
                  mix="a=1.000")
    kwargs.update(overrides)
    logger.log(10, 3.2, 3.3, out, 0.1, **kwargs)
    with open(path, newline="") as f:
        return logger, list(csv.DictReader(f))


def test_every_declared_column_is_written_when_its_input_exists(tmp_path, monkeypatch):
    from trm.runtime import metrics
    monkeypatch.setattr(metrics, "_arena_peak_mib", lambda: "4112")  # a device that keeps statistics
    monkeypatch.setattr(metrics, "_arena_limit_mib", lambda: "4883")
    logger, rows = _log_one(tmp_path)
    empty = [name for name in logger.fields if rows[0][name] == ""]
    assert not empty, f"declared in the header but never written: {empty}"
    assert float(rows[0]["act_max"]) == 1.5


def test_the_console_line_shows_only_the_diagnostics_the_model_reported(tmp_path, capsys):
    """A diagnostic the model did not report prints nothing, never a 0 that reads as
    a measurement nobody took (#317)."""
    logger = MetricsLogger(str(tmp_path / "metrics.csv"))
    logger.log(10, 3.2, 3.3, SimpleNamespace(diag={"out_entropy": jnp.array(2.5)}), 0.1, seg1_ce=3.0)
    line = capsys.readouterr().out
    assert "H: 2.500" in line and "Compute: 0.100s" in line
    assert "logZ" not in line and "max|logit|" not in line

    logger.log(11, 3.2, 3.3, SimpleNamespace(diag={k: jnp.array(2.5) for k in logger.diag_keys}), 0.1,
               seg1_ce=3.0)
    line = capsys.readouterr().out
    assert "H: 2.500" in line and "logZ: 2.50" in line


def test_a_row_says_when_it_was_written(tmp_path):
    before = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    _, rows = _log_one(tmp_path)
    stamp = datetime.datetime.strptime(rows[0]["wall_clock"], "%Y-%m-%dT%H:%M:%SZ"
                                       ).replace(tzinfo=datetime.timezone.utc)
    assert before <= stamp <= before + datetime.timedelta(minutes=1)


def test_a_row_names_the_mixture_its_ce_was_measured_on(tmp_path):
    label = mixture_label(PRETRAIN_SOURCES, [0.6, 0.25, 0.15])
    assert label == "fineweb-edu=0.600 codeparrot=0.250 finemath=0.150"
    _, rows = _log_one(tmp_path, mix=label)
    assert rows[0]["mix"] == label


def test_the_mixture_label_covers_every_source_the_mixer_serves():
    assert len(PRETRAIN_SOURCES) == len(Schedules.of(CONFIG).start_weights)


def test_a_resume_onto_an_older_csv_rewrites_it_to_the_wider_schema(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("step,ce\n5,3.0\n15,2.9\n")
    MetricsLogger(str(path), start_opt_step=10)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "wall_clock" in reader.fieldnames and "mix" in reader.fieldnames
    assert [r["step"] for r in rows] == ["5"]


def test_arena_peak_is_empty_where_the_allocator_keeps_no_statistics(monkeypatch):
    """CPU and the platform allocator have no peak to report; an empty cell says so,
    where a 0 would read as a measurement (#105)."""
    import jax

    from trm.runtime import metrics

    class Device:
        def __init__(self, stats):
            self.stats = stats

        def memory_stats(self):
            return self.stats

    monkeypatch.setattr(jax, "local_devices", lambda: [Device(None)])
    assert metrics._arena_peak_mib() == ""
    monkeypatch.setattr(jax, "local_devices", lambda: [Device({"peak_bytes_in_use": 4112 * 2**20})])
    assert metrics._arena_peak_mib() == "4112"


def test_a_run_written_before_the_arena_limit_column_resumes_under_the_new_header(tmp_path):
    """#346 added a column. A resume must rewrite an older CSV to the current header
    rather than append rows wider than it, and the old rows keep their values."""
    from trm.runtime.metrics import COLUMNS

    old_fields = [c.name for c in COLUMNS if c.name != "arena_limit_mib"]
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        for step in (5, 10, 15):
            writer.writerow({"step": step, "ce": "3.0", "arena_peak_mib": "4400"})

    MetricsLogger(str(path), start_opt_step=15)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "arena_limit_mib" in reader.fieldnames
    assert [r["step"] for r in rows] == ["5", "10"]
    assert rows[0]["arena_peak_mib"] == "4400" and rows[0]["arena_limit_mib"] == ""


def test_an_older_header_is_appended_to_under_its_own_columns(tmp_path):
    """#292 retired columns (depth_avg, tau, ...). A file that already has a header
    keeps it: a new row goes under the columns the file has, a column it no longer
    writes stays an empty cell, and nothing shifts — a reader by name still finds
    every value where it belongs."""
    from trm.runtime.metrics import COLUMNS

    current = [c.name for c in COLUMNS]
    # An older schema: retired columns in the middle and at the end, and one current
    # column (arena_limit_mib) it predates.
    old_fields = [*current[:3], "depth_avg", *[n for n in current[3:] if n != "arena_limit_mib"], "tau"]
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({"step": 5, "ce": "3.0000", "depth_avg": "1.0000", "tau": "0.5"})

    _log_one(tmp_path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == old_fields, "the file's own header must stay"
        rows = list(reader)
    assert len(rows) == 2 and all(None not in r for r in rows), "a row is wider than the header"
    old, new = rows
    assert old["depth_avg"] == "1.0000" and old["tau"] == "0.5"
    assert new["step"] == "10" and new["depth_avg"] == "" and new["tau"] == ""
    assert float(new["ce"]) == 3.2 and float(new["val_ce"]) == 3.1 and new["mix"] == "a=1.000"


def test_a_resume_keeps_retired_columns_and_adds_current_ones(tmp_path):
    """The resume trim widens an older file rather than replacing its header (#292):
    a retired column's old values are history, and the current columns it lacks are
    added at the end, empty on the old rows."""
    path = tmp_path / "metrics.csv"
    path.write_text("step,ce,depth_avg\n5,3.0,4.1\n15,2.9,4.2\n")
    MetricsLogger(str(path), start_opt_step=10)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames[:3] == ["step", "ce", "depth_avg"] and "mix" in reader.fieldnames
    assert [(r["step"], r["depth_avg"], r["mix"]) for r in rows] == [("5", "4.1", "")]


def test_per_block_readings_go_to_blocks_csv_one_row_per_state(tmp_path):
    """#392: the model's per-block readings land beside metrics.csv; an output that
    reports none writes no file at all (absent, never zero)."""
    logger = MetricsLogger(str(tmp_path / "metrics.csv"))
    diag = {"act_max_blocks": jnp.array([1.5, 20.0, 45.0]), "act_rms_blocks": jnp.array([0.02, 0.9, 1.4])}
    logger.log(10, 3.2, 3.3, SimpleNamespace(diag=diag), 0.1, seg1_ce=3.0)
    logger.log(15, 3.1, 3.2, SimpleNamespace(diag=diag), 0.1, seg1_ce=3.0)
    with open(tmp_path / "blocks.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["step"], r["block"]) for r in rows[:3]] == [("10", "0"), ("10", "1"), ("10", "2")]
    assert rows[2]["act_max"] == "45.00" and len(rows) == 6

    MetricsLogger(str(tmp_path / "metrics.csv"), start_opt_step=15)
    with open(tmp_path / "blocks.csv", newline="") as f:
        assert {r["step"] for r in csv.DictReader(f)} == {"10"}, "a resume trims replayed steps"

    other = tmp_path / "other"
    other.mkdir()
    MetricsLogger(str(other / "metrics.csv")).log(10, 3.2, 3.3, SimpleNamespace(diag={}), 0.1,
                                                   seg1_ce=3.0)
    assert not (other / "blocks.csv").exists()


def test_blocks_csv_carries_each_blocks_branch_outputs_and_widens_an_old_file(tmp_path):
    """#536: row k > 0 is the stream after block k and carries block k's two branch
    peaks; row 0, the embedding, has none. A blocks.csv from before the columns
    existed gains them on resume, its old rows left blank."""
    old = tmp_path / "blocks.csv"
    old.write_text("step,block,act_max,act_rms\n5,0,1.00,0.0100\n")
    logger = MetricsLogger(str(tmp_path / "metrics.csv"), start_opt_step=10)
    diag = {"act_max_blocks": jnp.array([1.5, 20.0, 45.0]), "act_rms_blocks": jnp.array([0.02, 0.9, 1.4]),
            "branch_max_blocks": jnp.array([[3.0, 4.0], [5.0, 6.5]])}
    logger.log(10, 3.2, 3.3, SimpleNamespace(diag=diag), 0.1, seg1_ce=3.0)
    with open(old, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["step"], r["attn_out_max"], r["mlp_out_max"]) for r in rows] == [
        ("5", "", ""), ("10", "", ""), ("10", "3.00", "4.00"), ("10", "5.00", "6.50")]

