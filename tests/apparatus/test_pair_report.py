"""instruments.pair_report: the readout beside the verdict (#564), on a pair built here."""

import json
import pathlib
import textwrap

from instruments import pair_report, results
from instruments.verdict import load_spec

SPEC = textwrap.dedent("""
    [experiment]
    id = "t-pair"
    title = "toy"
    hypothesis = "toy"

    [arms.control]
    role = "control"
    flags = []

    [arms.knob]
    role = "treatment"
    flags = ["--set", "KNOB=1"]

    [criteria.knob_wins]
    rule = "beats"
    treatment = "knob"
    control = "control"
    sigmas = 2.0

    [verdict]
    keep_if = ["knob_wins"]
""")

HEADER = "step,val_ce,val_step,act_max,branch_max,wall_clock\n"


def _run(root: pathlib.Path, name: str, ces, act_max) -> pathlib.Path:
    """A metrics.csv probed every 8 steps, logged every 8, 10 s per opt step, with one
    pause of an hour between the second and third rows."""
    run = root / name
    run.mkdir()
    rows, clock = [], 0
    for i, (ce, act) in enumerate(zip(ces, act_max, strict=True)):
        clock += 80 + (3600 if i == 2 else 0)
        stamp = f"2026-10-01T{clock // 3600:02d}:{clock % 3600 // 60:02d}:{clock % 60:02d}Z"
        rows.append(f"{8 * (i + 1)},{ce},{8 * (i + 1)},{act},{act / 4},{stamp}\n")
    (run / "metrics.csv").write_text(HEADER + "".join(rows))
    return run


def _pair(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(SPEC)
    curves = {("control", 0): ([5.0, 4.5, 4.2, 4.0, 3.9], [100, 300, 600, 500, 480]),
              ("control", 1): ([5.1, 4.6, 4.3, 4.1, 3.92], [90, 280, 640, 510, 470]),
              ("knob", 0): ([4.9, 4.4, 4.1, 3.95, 3.85], [40, 60, 75, 70, 66]),
              ("knob", 1): ([4.95, 4.45, 4.15, 3.97, 3.86], [45, 62, 77, 71, 65])}
    rows = []
    for (arm, seed), (ces, act) in curves.items():
        run = _run(tmp_path, f"run_{arm}_s{seed}", ces, act)
        rows.append({"arm": arm, "seed": seed, "point": "run", "final_val_ce": ces[-1], "run_dir": str(run)})
    return load_spec(spec_path), rows


def test_the_report_reads_every_section_off_the_runs(tmp_path):
    spec, rows = _pair(tmp_path)
    report = pair_report.render(spec, rows, tmp_path)
    assert "control: control" in report
    assert "knob: every seed below every control seed" in report
    # Last probe: control 3.91, knob 3.855 -> Δ -0.055.
    assert ["40", "3.9100", "3.8550", "-0.0550"] in [line.split() for line in report.splitlines()]
    assert "640.0" in report and "77.0" in report, "the peak is over seeds and steps"
    assert "10.0 s/opt step" in report, "the hour-long pause is not training time"


def test_a_force_re_read_reports_the_latest_row(tmp_path):
    spec, rows = _pair(tmp_path)
    stale = {**rows[0], "final_val_ce": 9.99}
    report = pair_report.render(spec, [stale, *rows], tmp_path)
    assert "9.99" not in report


def test_a_moved_pair_is_found_by_run_name_under_runs(tmp_path):
    spec, rows = _pair(tmp_path)
    moved = [{**r, "run_dir": f"/gone/{pathlib.Path(r['run_dir']).name}"} for r in rows]
    assert "s/opt step" in pair_report.render(spec, moved, tmp_path)


def test_a_journal_without_run_dirs_still_reports_the_seeds_and_says_why_no_curves(tmp_path):
    spec, rows = _pair(tmp_path)
    old = [{k: v for k, v in r.items() if k != "run_dir"} for r in rows]
    report = pair_report.render(spec, old, tmp_path)
    assert "final_val_ce per seed" in report and "s/opt step" not in report
    assert "no curves for control s0" in report and "#564" in report


def test_a_spec_with_no_control_arm_gets_no_report(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(SPEC.replace('role = "control"', 'role = "baseline"'))
    assert pair_report.render(load_spec(spec_path), [], tmp_path) == ""


def test_the_result_line_carries_the_run_dir(capsys):
    results.emit("run", run_dir=pathlib.Path("/r/run_x"), tokens_to_target_M=1)
    (row,) = results.parse(capsys.readouterr().out)
    assert row == {"point": "run", "run_dir": "/r/run_x", "tokens_to_target_M": 1.0}


def test_the_cli_reads_the_journal_under_runs(tmp_path, capsys):
    spec, rows = _pair(tmp_path)
    journal = tmp_path / "experiments" / "t-pair" / "results.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert pair_report.main([str(tmp_path / "spec.toml"), "--runs", str(tmp_path)]) == 0
    assert "pair report" in capsys.readouterr().out
