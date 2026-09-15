"""The base run is pre-registered, scored and judged like every other experiment (#294)."""

import json
import pathlib

import pytest

from instruments import base_run
from instruments.verdict import INCONCLUSIVE, KEEP, KILL

REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC = REPO / "experiments/base/specs/001-plain-base.toml"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "champion_run_metadata.json"


def test_the_committed_base_spec_is_a_valid_base_spec():
    spec = base_run.load_base_spec(SPEC)
    assert spec.meta["protocol"]["budget_tokens"] > 0
    assert set(spec.criteria) == {"meets_the_bar", "misses_by_a_margin"}


@pytest.mark.parametrize("mutation, message", [
    ("[arms.gpt2_small]\nrole = \"floor\"\nconstant = true\n", "constant="),
    ("[results.run]\ngpt2_small = { mean = 0.3256, sigma = 0.0, n = 1 }\n", "declares no"),
    ("budget_tokens = 4000000000\n", "budget_tokens"),
])
def test_a_spec_missing_its_reference_or_budget_fails_to_load(tmp_path, mutation, message):
    text = SPEC.read_text()
    assert mutation in text
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace(mutation, ""))
    with pytest.raises(ValueError, match=message):
        base_run.load_base_spec(bad)


def _run_dir(tmp_path, acc):
    run = tmp_path / f"run_{acc}"
    run.mkdir()
    (run / base_run.JOURNAL).write_text(
        json.dumps({"step": 999, "limit": 1000, "lambada_acc": 0.01, "lambada_ppl": 9e9}) + "\n"
        + json.dumps({"step": 1000, "limit": None, "lambada_acc": acc, "lambada_ppl": 50.0}) + "\n")
    return run


def test_the_verdict_reads_the_full_set_line_not_a_milestone_subsample(tmp_path):
    assert base_run.verdict_for(SPEC, _run_dir(tmp_path, 0.34)).outcome == KEEP
    assert base_run.verdict_for(SPEC, _run_dir(tmp_path, 0.31)).outcome == INCONCLUSIVE, "within 0.05 below: fix nothing yet"
    assert base_run.verdict_for(SPEC, _run_dir(tmp_path, 0.098)).outcome == KILL, "the champion's number: the pipeline is the bug"


def test_no_full_set_line_means_no_verdict(tmp_path):
    run = tmp_path / "run_y"
    run.mkdir()
    with pytest.raises(ValueError, match="no full-set yardstick"):
        base_run.verdict_for(SPEC, run)


def test_the_card_reproduces_the_champions_fields_from_its_archived_metadata(tmp_path):
    run = tmp_path / "run_20260813_214725"
    run.mkdir()
    (run / "run_metadata.json").write_text(FIXTURE.read_text())
    (run / "metrics.csv").write_text("step,ce,val_ce,arena_peak_mib\n30465,2.3555,3.6474,5002\n")
    f = base_run.card_fields(run)
    assert f["commit"].startswith("5455e4d") and f["arch"] == "refiner" and f["dirty"] is True
    assert f["budget"] == 4000000000 and f["seeds"] == "DATA_SEED=42, MODEL_SEED=42"
    assert f["sections"] == 11 and 250 < f["hours"] < 270, f["hours"]
    assert f["tokens_seen"] == 30465 * 131072 and f["val_ce"] == 3.6474 and f["peak_vram_mib"] == 5002
    assert f["lambada_acc"] is None and f["weights_sha256"] == "n/a"
    card = base_run.render_card(f)
    assert "`5455e4d2eb77ba015b5358ad6dc0ac7ab739eea8`" in card and "3,993,108,480" in card and "not scored" in card


def test_a_launch_refuses_an_uncommitted_or_dirty_spec(tmp_path):
    from trm.runtime import launch
    loose = tmp_path / "loose.toml"
    loose.write_text(SPEC.read_text())
    assert "not committed" in launch.spec_refusal(loose)
    assert launch.spec_refusal(REPO / "experiments/depth/specs/235-post-norm.toml") is None
    assert launch.spec_budget_tokens(SPEC) == 4000000000


def test_the_launcher_needs_a_spec_and_a_matching_budget(tmp_path, monkeypatch):
    from trm.runtime import launch
    with pytest.raises(SystemExit, match="no SPEC"):
        launch.main(["--budget", "4e9", "--dry-run"])
    monkeypatch.setattr(launch, "spec_refusal", lambda p: None)
    with pytest.raises(SystemExit, match="disagrees with the spec"):
        launch.main(["--budget", "1e9", "--spec", str(SPEC), "--dry-run"])
    launch.main(["--budget", "4e9", "--spec", str(SPEC), "--dry-run"])


def test_trm_never_imports_the_instruments_it_drives():
    import re
    for f in ("trm/runtime/launch.py", "trm/runtime/supervisor.py"):
        src = (REPO / f).read_text()
        assert not re.search(r"^\s*(from|import) instruments", src, re.M), f


def test_the_supervisor_scores_each_milestone_once_on_the_cpu(tmp_path, monkeypatch):
    from trm.runtime import supervisor as sup_mod
    from trm.runtime.supervisor import Limits, Supervisor
    run = tmp_path / "run_z"
    for step in ("100", "200"):
        (run / "checkpoints" / "milestones" / step).mkdir(parents=True)
        (run / "checkpoints" / "milestones" / step / "_CHECKPOINT_METADATA").write_text("{}")
    launched = []
    monkeypatch.setattr(sup_mod.subprocess, "Popen", lambda argv, **kw: launched.append(argv) or type("P", (), {"poll": lambda s: 0})())
    sup = Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
                     metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None)
    sup.score_new_milestones()
    sup.score_new_milestones()
    assert len(launched) == 2 and all("--cpu" in a and "--limit" in a for a in launched)
