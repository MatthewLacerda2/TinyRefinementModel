"""The plotter draws what was measured, and nothing else.

Two properties are worth a test here, and they are the two the previous
plotter got wrong (#177):

  * **A panel with no data is omitted.** Both flavours of "no data": a blank
    column (the arch never measured it, #105) and a column of literal zeros
    (what runs written before the blank convention put there). Either one drawn
    flat reads as a measurement of zero.
  * **The model is never instantiated.** Building a whole network to count its
    parameters is slow, wrong-architecture-prone, and a hazard while the card
    is training. The subprocess test is the one that can actually prove it —
    it checks that no model module was ever imported.

Everything else here is survival: a run three steps old, a header-only CSV, a
torn final row. A live run gets plotted while it is being written to, so the
instrument meets all three routinely.
"""

import os
import pathlib
import subprocess
import sys

import pytest

# Marker-anchored, not a fixed parent-hop count (see tests/core/test_seed_config.py).
REPO_ROOT = str(next(p for p in pathlib.Path(__file__).resolve().parents
                     if (p / "pyproject.toml").exists()))

HEADER = ("step,ce,loss,seg1_ce,grad_norm_avg,zero_frac_dense_max,avg_forget_cost,"
          "diversity_loss,temporal_drift,forget_density,tau,out_entropy,logz_mean,"
          "max_abs_logit,depth_avg,val_ce")


def row(step, **overrides):
    """One refiner-shaped CSV row: the reasoner-only columns blank, as the live
    trainer writes them."""
    cells = {
        "ce": 11.0 - step / 500, "loss": 22.0, "seg1_ce": 10.8,
        "grad_norm_avg": 12.0 + step / 1000, "zero_frac_dense_max": 0.004,
        "avg_forget_cost": "", "diversity_loss": "", "temporal_drift": "",
        "forget_density": "", "tau": "", "out_entropy": 9.0 - step / 800,
        "logz_mean": 11.3, "max_abs_logit": 4.3 + step / 200,
        "depth_avg": 4.5, "val_ce": "",
    }
    cells.update(overrides)
    return ",".join([str(step)] + [str(cells[name]) for name in HEADER.split(",")[1:]])


def write_csv(tmp_path, rows, name="run_20260101_000000"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    return run_dir / "metrics.csv"


def a_run(tmp_path, steps=400, **overrides):
    rows = [row(step, **overrides) for step in range(5, steps + 1, 5)]
    return write_csv(tmp_path, rows)


def build(csv_path, out):
    from instruments import plots

    return {figure["path"].split("/")[-1]: figure for figure in plots.build(str(csv_path), out)}


def panels_of(figures, filename):
    return figures[filename]["panels"] if filename in figures else []


# ── the rule: absent data is omitted, and the reason is said out loud ────────

def test_reasoner_only_columns_never_get_a_panel(tmp_path):
    """The refiner has no forget gate, no slots and no drift (#105). Those
    columns are blank in every refiner run and must not reach a figure."""
    figures = build(a_run(tmp_path), tmp_path)

    drawn = [name for figure in figures.values() for name in figure["panels"]]
    for absent in ("temporal_drift", "avg_forget_cost", "diversity_loss", "forget_density",
                   "tau", "seg1_ce"):
        assert absent not in drawn


def test_blank_column_omits_its_panel_and_says_why(tmp_path):
    figures = build(a_run(tmp_path, grad_norm_avg=""), tmp_path)

    health = figures["optimization_health.png"]
    assert "grad_norm" not in health["panels"]
    assert ("grad_norm", "not measured by this architecture") in health["omitted"]
    assert health["panels"], "the other diagnostics still have data"


def test_literal_zero_column_is_omitted_too(tmp_path, capsys):
    """The case `has()` alone cannot catch: a pre-#105 run wrote 0.0000 into the
    columns it did not measure, so the column is present and still not data."""
    figures = build(a_run(tmp_path, depth_avg="0.0000"), tmp_path)

    health = figures["optimization_health.png"]
    assert "depth" not in health["panels"]
    assert ("depth", "constant 0 throughout — logged, but not a measurement") in health["omitted"]
    # Told, not silently dropped: absence and a flat zero must stay tellable apart.
    assert "constant 0" in capsys.readouterr().out


def test_a_real_flat_signal_is_still_drawn(tmp_path):
    """Constant-at-zero is the tombstone; constant at some other value is a
    measurement (a sampler pinned to depth 4, say), and it gets its panel."""
    figures = build(a_run(tmp_path, depth_avg="4.0000"), tmp_path)

    assert "depth" in figures["optimization_health.png"]["panels"]


def test_every_diagnostic_panel_appears_for_a_full_log(tmp_path):
    figures = build(a_run(tmp_path), tmp_path)

    assert panels_of(figures, "optimization_health.png") == [
        "grad_norm", "depth", "zero_grad", "logits"]
    assert figures["optimization_health.png"]["omitted"] == []


def test_val_ce_is_drawn_only_where_it_exists(tmp_path):
    """val_ce is written every 64 opt steps, so most rows are blank — it is a
    series of its own, never interpolated onto every row."""
    from instruments import plots
    from instruments.runlog import load

    rows = [row(step, val_ce=(6.0 if step % 320 == 0 else "")) for step in range(5, 1001, 5)]
    csv_path = write_csv(tmp_path, rows)
    figures = build(csv_path, tmp_path)
    assert "val_ce" in figures["training_curve.png"]["panels"]

    tokens, values = plots.series(load(str(csv_path)), "val_ce")
    assert len(values) == 3, "one point per row that actually carried a value"

    no_val = build(a_run(tmp_path / "b"), tmp_path / "b")
    assert "val_ce" not in no_val["training_curve.png"]["panels"]


# ── survival: the shapes a live CSV actually takes ───────────────────────────

def test_a_run_three_steps_old(tmp_path):
    figures = build(a_run(tmp_path, steps=15), tmp_path)

    assert "training_curve.png" in figures
    assert pathlib.Path(figures["training_curve.png"]["path"]).stat().st_size > 0


def test_header_only_csv_writes_nothing(tmp_path):
    figures = build(write_csv(tmp_path, []), tmp_path)

    assert figures == {}
    assert not list(tmp_path.glob("*.png"))


def test_torn_final_row(tmp_path):
    """The trainer can be mid-write when the plotter reads. A half-line is
    dropped; every complete row before it still gets drawn."""
    rows = [row(step) for step in range(5, 401, 5)]
    csv_path = write_csv(tmp_path, rows)
    with open(csv_path, "a") as handle:
        handle.write("405,3.9,7.8,4.1,11.\n")

    figures = build(csv_path, tmp_path)
    assert panels_of(figures, "training_curve.png")


def test_non_finite_values_do_not_break_the_curve(tmp_path):
    """A diverged step logs nan/inf. The loader keeps them (they mean something
    went wrong); the plotter must not hand them to a log axis."""
    rows = [row(step) for step in range(5, 401, 5)] + [row(405, ce="nan"), row(410, ce="inf")]
    figures = build(write_csv(tmp_path, rows), tmp_path)

    assert "training_curve.png" in figures


def test_throughput_needs_the_supervisor_log(tmp_path):
    """metrics.csv has no timestamps at all, so with no heartbeats to sample
    there is no honest throughput figure — and none is written."""
    figures = build(a_run(tmp_path), tmp_path)
    assert "throughput_progress.png" not in figures

    csv_path = a_run(tmp_path / "with_log", steps=600)
    (csv_path.parent.parent / f"{csv_path.parent.name}.supervisor.log").write_text(
        "2026-01-01 00:00:00 RUNNING: step 200/30518 (ce=7.0)\n"
        "2026-01-01 01:00:00 RUNNING: step 400/30518 (ce=6.5)\n"
        "2026-01-01 02:00:00 RUNNING: step 600/30518 (ce=6.1)\n")

    figures = build(csv_path, tmp_path / "with_log")
    assert panels_of(figures, "throughput_progress.png") == ["throughput", "progress"]


def test_missing_run_is_an_error_not_an_empty_figure(tmp_path):
    from instruments import plots

    with pytest.raises(FileNotFoundError):
        plots.build(str(tmp_path / "nope" / "metrics.csv"), tmp_path)


# ── the constraint: no model, ever ───────────────────────────────────────────

def test_the_plotter_never_imports_a_model(tmp_path):
    """The defect this rewrite removes: the old plotter built a whole
    UniversalReasoner to count parameters — the wrong architecture, and a
    hazard on a card that is training. Run it for real in a clean interpreter
    and check no model module was ever imported."""
    csv_path = a_run(tmp_path)
    probe = (
        "import sys, runpy;"
        f"sys.argv = ['plots', '--log', {str(csv_path)!r}, '--out', {str(tmp_path)!r}];"
        "runpy.run_module('instruments.plots', run_name='__main__');"
        "print('MODELS:', [m for m in sys.modules if m.startswith('trm.model')])"
    )
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, JAX_PLATFORMS="cpu", FORCE_F32_COMPUTE="1")
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stderr
    assert "MODELS: []" in proc.stdout, "the plotter pulled in a model module"
    assert (tmp_path / "training_curve.png").exists()
