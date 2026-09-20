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
    ("budget_tokens = 5000000000\n", "budget_tokens"),
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
    # The champion's card records 259.4 h running across 11 sections
    # (docs/registry/run_20260813_214725-refiner-base-4B.md). The band brackets it
    # instead of pinning the float sum of per-section durations.
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
    assert launch.spec_budget_tokens(SPEC) == 5000000000


def test_the_launcher_needs_a_spec_and_a_matching_budget(tmp_path, monkeypatch):
    from trm.runtime import launch
    with pytest.raises(SystemExit, match="no SPEC"):
        launch.main(["--budget", "5e9", "--dry-run"])
    monkeypatch.setattr(launch, "spec_refusal", lambda p: None)
    with pytest.raises(SystemExit, match="disagrees with the spec"):
        launch.main(["--budget", "1e9", "--spec", str(SPEC), "--dry-run"])
    launch.main(["--budget", "5e9", "--spec", str(SPEC), "--dry-run"])


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


def test_a_milestone_is_scored_as_itself_not_as_the_newest_rolling_checkpoint(tmp_path, monkeypatch):
    """#328: the scorer named no checkpoint, so base_run scored the newest rolling one.
    That is not the milestone, and rolling retention (3) can evict it mid-restore. Here
    milestone 100 sits beside rolling checkpoints 150 and 200: the supervisor's command,
    run through base_run, must hand the yardstick milestone 100 and journal it as such."""
    from trm.runtime import supervisor as sup_mod
    from trm.runtime.layout import MILESTONE_SUBDIR
    from trm.runtime.supervisor import Limits, Supervisor

    run = tmp_path / "run_m"
    milestones = run / "checkpoints" / MILESTONE_SUBDIR
    (milestones / "100").mkdir(parents=True)
    (milestones / "100" / "_CHECKPOINT_METADATA").write_text("{}")
    (milestones / "100.orbax-checkpoint-tmp-1").mkdir()
    (milestones / "100.orbax-checkpoint-tmp-1" / "_CHECKPOINT_METADATA").write_text("{}")
    for rolling in ("150", "200"):
        (run / "checkpoints" / rolling).mkdir(parents=True)

    launched = []
    monkeypatch.setattr(sup_mod.subprocess, "Popen",
                        lambda argv, **kw: launched.append(argv) or type("P", (), {"poll": lambda s: 0})())
    Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
               metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None).score_new_milestones()
    assert len(launched) == 1, "one milestone; the orbax tmp dir beside it is not one"
    (argv,) = launched
    assert argv[argv.index("--checkpoint-dir") + 1] == str(milestones)
    assert argv[argv.index("--step") + 1] == "100"

    yardstick_calls = []

    def fake_yardstick(cmd, **kw):
        yardstick_calls.append(cmd)
        out = pathlib.Path(cmd[cmd.index("--json-out") + 1])
        out.write_text(json.dumps({"lambada": {"lambada_acc": 0.25, "lambada_ppl": 80.0, "num_examples": 2}}))
        return type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(base_run.subprocess, "run", fake_yardstick)
    assert base_run.main(argv[argv.index("score"):]) == 0
    (cmd,) = yardstick_calls
    assert cmd[cmd.index("--checkpoint-path") + 1] == str(milestones)
    assert cmd[cmd.index("--step") + 1] == "100"
    line = base_run.journal(run)[-1]
    assert (line["step"], line["source"], line["lambada_acc"]) == (100, "milestone", 0.25)

    # With no checkpoint named (the completion call), the newest rolling step is scored.
    yardstick_calls.clear()
    assert base_run.main(["score", "--run", str(run)]) == 0
    assert yardstick_calls[0][yardstick_calls[0].index("--step") + 1] == "200"
    assert (base_run.journal(run)[-1]["step"], base_run.journal(run)[-1]["source"]) == (200, "rolling")


def test_scoring_restores_the_arch_the_run_recorded_not_the_shells(tmp_path, monkeypatch):
    """#313: the yardstick must rebuild the param tree the run trained, and that is in
    the run's metadata. The champion recorded `refiner`; the default here is `plain`.
    A run that recorded nothing falls back to the yardstick's own MODEL_ARCH default, and
    so does one whose metadata is caught mid-rewrite: RunTracker rewrites it in place, and
    the milestone scorer's output goes to /dev/null, so raising would lose the milestone."""
    calls = []
    failed = type("Proc", (), {"returncode": 1, "stderr": "stubbed", "stdout": ""})()
    monkeypatch.setattr(base_run.subprocess, "run", lambda argv, **kw: calls.append(argv) or failed)
    recorded, bare, torn = tmp_path / "recorded", tmp_path / "bare", tmp_path / "torn"
    for run in (recorded, bare, torn):
        (run / "checkpoints" / "40").mkdir(parents=True)
    (recorded / "run_metadata.json").write_text(FIXTURE.read_text())
    (torn / "run_metadata.json").write_text(FIXTURE.read_text()[:100])

    for run in (recorded, bare, torn):
        assert base_run.main(["score", "--run", str(run), "--limit", "2", "--cpu"]) == 1
        assert base_run.journal(run)[-1]["error"] == "stubbed", "a failed score still leaves its journal line"
    with_meta, without, half_written = calls
    assert with_meta[with_meta.index("--arch") + 1] == "refiner"
    assert "--arch" not in without and "--arch" not in half_written


def test_only_the_runs_own_checkpoint_dir_is_journaled_as_rolling(tmp_path):
    """#334 review: any dir other than milestones/best was labelled 'rolling', so a
    --checkpoint-dir at a rewind's set_aside_* dir would pass for the rolling series."""
    run = tmp_path / "run_s"
    checkpoints = run / "checkpoints"
    assert base_run.checkpoint_source(checkpoints, run) == "rolling"
    assert base_run.checkpoint_source(checkpoints / "milestones", run) == "milestone"
    assert base_run.checkpoint_source(checkpoints / "best_val_ce", run) == "best"
    assert base_run.checkpoint_source(checkpoints / "set_aside_2026", run) == "set_aside_2026"


def test_the_card_shows_only_the_knobs_its_arch_reads():
    """#332 review: every card listed REFINER_ENCODER_LAYERS, a knob plain never reads."""
    plain, refiner = base_run.card_config_keys("plain"), base_run.card_config_keys("refiner")
    assert "PLAIN_LAYERS" in plain and "REFINER_ENCODER_LAYERS" not in plain and "TIME_SIGNAL" not in plain
    assert {"REFINER_ENCODER_LAYERS", "TIME_SIGNAL"} <= set(refiner) and "PLAIN_LAYERS" not in refiner
    assert {"PLAIN_LAYERS", "REFINER_ENCODER_LAYERS"} <= set(base_run.card_config_keys(None)), \
        "a run that recorded no arch shows every knob that might apply"


def test_a_card_without_a_recorded_recipe_does_not_claim_zero_tokens(tmp_path):
    """#343 review: an unrecorded ACCUMULATION/BATCH/SEQ recipe rendered 'Tokens seen 0'."""
    run = tmp_path / "run_norecipe"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"run_id": "run_norecipe", "parameters": {"MODEL_ARCH": "plain"}}))
    (run / "metrics.csv").write_text("step,ce\n40,3.1\n")
    fields = base_run.card_fields(run)
    assert fields["tokens_seen"] is None
    assert "| Tokens seen | unknown (recipe not recorded) (opt step 40) |" in base_run.render_card(fields)
