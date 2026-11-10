"""The runner against a real harness — the one claim the stub cannot make.

`tests/apparatus/test_experiment_runner.py` proves the plumbing against a stub
that answers in milliseconds. What it cannot prove is that the RESULT protocol
survives contact with an actual JAX harness: that
`experiments/recipe/clip_under_muon.py` really emits parseable lines, that they
carry the metric the criteria name, and that a spec pointed at it comes back
with a verdict rather than a traceback.

So this runs the real thing — on a toy model, a made-up norm profile and random
tokens, at a step count chosen to be fast, not to be scientifically meaningful.
It asserts the pipeline completed and the numbers are in range, and deliberately
asserts NOTHING about what the replay found: 20 steps of a toy is noise, and a
test that demanded a particular reading would be manufacturing exactly the false
result this whole build exists to prevent.
"""

import pathlib
import sys
import textwrap

import numpy as np

from instruments import experiment
from instruments.verdict import evaluate, load_recorded_results, load_spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STEPS = 20
QUANTITIES = ("muon_cos", "muon_ratio", "adam_cos", "adam_ratio")

SPEC = """
[experiment]
id = "runner-e2e"
title = "Does the runner drive a real harness end to end?"
hypothesis = "Plumbing check, not an experiment: 20 replayed steps decide nothing."

[protocol]
metric_key = "value"
metric = "median clipped-vs-unclipped update cosine / norm ratio, per band"

[execution]
command = ["{python}", "-m", "experiments.recipe.clip_under_muon",
           "--metrics", "{metrics}", "--data", "{data}", "--steps", "{steps}",
           "--dim", "32", "--layers", "1", "--vocab", "256", "--seq", "16", "--batch", "2"]
seed_flag = "--seed"
seeds = [0]
env = {{ JAX_PLATFORMS = "cpu", FORCE_F32_COMPUTE = "1" }}

[execution.legs.smooth]
flags = ["--scatter-scale", "0"]

[execution.legs.noisy]
flags = ["--scatter-scale", "1"]

[arms.clip]
role = "control"
flags = ["--clip", "1.0"]

[criteria.sanity]
rule = "within"
treatment = "clip"
control = "clip"
sigmas = 2.0

[verdict]
keep_if = ["sanity"]
"""


def _inputs(tmp_path):
    """A norm profile logged every other step, falling through the clip, and one
    shard of random token ids."""
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,applied_grad_norm\n" + "".join(
        f"{step},{4.0 * 0.85 ** step:.6f}\n" for step in range(0, STEPS, 2)))
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "chunk_000.npy", np.random.default_rng(0).integers(0, 50_000, 4_096, dtype=np.int32))
    return metrics, data


def test_the_runner_drives_a_real_harness(tmp_path, monkeypatch):
    metrics, data = _inputs(tmp_path)
    spec_path = tmp_path / "e2e.toml"
    spec_path.write_text(textwrap.dedent(SPEC.format(python=sys.executable, metrics=metrics,
                                                     data=data, steps=STEPS)))
    monkeypatch.setattr(experiment, "RUNS_DIR", tmp_path / "runs")

    assert experiment.main([str(spec_path), "--no-gate"]) == 0

    results = load_recorded_results(spec_path)
    assert set(results) == {f"{leg}/{q}@0-{STEPS}" for leg in ("smooth", "noisy") for q in QUANTITIES}, (
        "both legs measured the same band and neither overwrote the other")
    for point, arms in results.items():
        (value,) = arms["clip"]
        if "_cos@" in point:
            assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6, f"{point}: {value} is not a cosine"
        else:
            assert value > 0.0 and np.isfinite(value), f"{point}: {value} is not a norm ratio"

    # An honest verdict came out the far end, and the draft is a draft.
    verdict = evaluate(load_spec(spec_path), results)
    assert verdict.outcome == "KEEP"
    draft = (tmp_path / "runs" / "runner-e2e" / "finding-draft.md").read_text()
    assert "_To be written by a human._" in draft
