"""The runner against a real harness — the one claim the stub cannot make.

`tests/apparatus/test_experiment_runner.py` proves the plumbing against a stub
that answers in milliseconds. What it cannot prove is that the RESULT protocol
survives contact with an actual JAX training harness: that
`experiments/depth/ablation_harness.py` really emits parseable lines, that they
carry the metric the criteria name, and that a spec pointed at it comes back
with a verdict rather than a traceback.

So this runs the real thing — at a step count chosen to be fast, not to be
scientifically meaningful. It asserts the pipeline completed and the numbers are
in range, and deliberately asserts NOTHING about which arm won: 40 steps on a
toy task is noise, and a test that demanded a particular winner would be
manufacturing exactly the false result this whole build exists to prevent.
"""

import pathlib
import sys
import textwrap

from instruments import experiment
from instruments.verdict import evaluate, load_recorded_results, load_spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

SPEC = """
[experiment]
id = "runner-e2e"
title = "Does the runner drive a real harness end to end?"
hypothesis = "Plumbing check, not an experiment: 40 steps decides nothing."

[protocol]
metric_key = "acc"
metric = "held-out accuracy on parity (chance 0.5)"

[execution]
command = ["{python}", "-m", "experiments.depth.ablation_harness",
           "--task", "parity", "--dim", "32", "--steps", "40"]
seed_flag = "--seed"
seeds = [0]
env = {{ JAX_PLATFORMS = "cpu", FORCE_F32_COMPUTE = "1" }}

[execution.legs.shallow]
flags = ["--depths", "1"]

[execution.legs.deep]
flags = ["--depths", "2"]

[arms.refiner]
role = "control"
flags = ["--arch", "refiner"]

[criteria.sanity]
rule = "within"
treatment = "refiner"
control = "refiner"
sigmas = 2.0

[verdict]
keep_if = ["sanity"]
"""


def test_the_runner_drives_the_real_ablation_harness(tmp_path, monkeypatch):
    spec_path = tmp_path / "e2e.toml"
    spec_path.write_text(textwrap.dedent(SPEC.format(python=sys.executable)))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")

    assert experiment.main([str(spec_path), "--no-gate"]) == 0

    results = load_recorded_results(spec_path)
    assert set(results) == {"shallow/d1", "deep/d2"}, (
        "both legs measured depth-something and neither overwrote the other")
    for point, arms in results.items():
        (value,) = arms["refiner"]
        assert 0.0 <= value <= 1.0, f"{point}: {value} is not an accuracy"

    # An honest verdict came out the far end, and the draft is a draft.
    verdict = evaluate(load_spec(spec_path), results)
    assert verdict.outcome == "KEEP"
    draft = (tmp_path / "runs" / "runner-e2e" / "finding-draft.md").read_text()
    assert "_To be written by a human._" in draft
