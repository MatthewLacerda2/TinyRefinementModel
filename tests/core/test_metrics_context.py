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
from trm.train.trainer import PRETRAIN_SOURCES, mixture_label


def _log_one(tmp_path, **overrides):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(str(path))
    out = SimpleNamespace(diag={k: jnp.array(1.5) for k in logger.diag_keys})
    kwargs = dict(grad_norm_avg=0.5, seg1_ce=3.0, depth_avg=1.0, val_ce=3.1,
                  zero_frac_dense_max=0.0, applied_zero_frac_dense_max=0.0, applied_grad_norm=0.7, clip_active=0, val_step=8, grad_by_source="fineweb-edu=1.0/2.0/0/5",
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
    """A plain model has no tau and no slot drift; `Tau: 0.0000 | Drift: 0.000000`
    printed a measurement nobody took (#317). The CSV schema is unchanged: those
    columns stay, empty, for every reader of old and new runs alike."""
    logger = MetricsLogger(str(tmp_path / "metrics.csv"))
    plain_diag = ("out_entropy", "logz_mean", "max_abs_logit", "act_max")
    logger.log(10, 3.2, 3.3, SimpleNamespace(diag={k: jnp.array(2.5) for k in plain_diag}), 0.1,
               seg1_ce=3.0, depth_avg=1.0)
    line = capsys.readouterr().out
    assert "Tau" not in line and "Drift" not in line
    assert "H: 2.500" in line and "logZ: 2.50" in line and "Compute: 0.100s" in line

    logger.log(11, 3.2, 3.3, SimpleNamespace(diag={k: jnp.array(0.5) for k in logger.diag_keys}), 0.1,
               seg1_ce=3.0, depth_avg=1.0)
    line = capsys.readouterr().out
    assert "Tau: 0.5000" in line and "Drift: 0.500000" in line


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
    from trm.train.schedules import CURRICULUM_START_WEIGHTS
    assert len(PRETRAIN_SOURCES) == len(CURRICULUM_START_WEIGHTS)


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
