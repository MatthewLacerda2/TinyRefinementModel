"""The base run is pre-registered, scored and judged like every other experiment (#294)."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

from instruments import base_run
from instruments.verdict import INCONCLUSIVE, KEEP, KILL
from trm.settings import Config

DEFAULTS = Config.from_env({})
FAKE_PID = 2 ** 22 + 1  # above Linux's largest pid_max: never a live process

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
    assert f["commit"].startswith("5455e4d") and f["dirty"] is True
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
    monkeypatch.setattr(sup_mod.subprocess, "Popen", lambda argv, **kw: launched.append(argv) or type("P", (), {"poll": lambda s: 0, "pid": FAKE_PID})())
    sup = Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
                     metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None, config=DEFAULTS)
    sup.score_new_milestones()
    sup.score_new_milestones()
    assert len(launched) == 2 and all("--cpu" in a and "--limit" in a for a in launched)


def _milestones(run, steps):
    for step in steps:
        (run / "checkpoints" / "milestones" / str(step)).mkdir(parents=True)
        (run / "checkpoints" / "milestones" / str(step) / "_CHECKPOINT_METADATA").write_text("{}")


def _fake_scorers(monkeypatch, launched, running):
    """Popen replaced by a scorer that is still running while `running[0]` is true."""
    from trm.runtime import supervisor as sup_mod
    monkeypatch.setattr(sup_mod.subprocess, "Popen", lambda argv, **kw: launched.append(argv) or type(
        "P", (), {"poll": lambda s: None if running[0] else 0, "pid": FAKE_PID})())


def _steps(launched):
    return [argv[argv.index("--step") + 1] for argv in launched]


def test_a_resumed_supervisor_scores_only_what_the_journal_lacks(tmp_path, monkeypatch):
    """#471: every resume makes a fresh supervisor, and it re-scored every milestone on
    disk. The journal says which are scored; its lines here come from base_run's own
    writer. 100, 200 and 300 are scored; 400's yardstick was OOM-killed, which leaves an
    error line and a milestone still owed a pass; a torn last line is not a score."""
    from trm.runtime.layout import MILESTONE_SUBDIR
    from trm.runtime.supervisor import Limits, Supervisor

    run = tmp_path / "run_r"
    _milestones(run, (100, 200, 300, 400))

    def fake_yardstick(cmd, **kw):
        if cmd[cmd.index("--step") + 1] == "400":
            return type("Proc", (), {"returncode": -9, "stderr": "Killed", "stdout": ""})()
        pathlib.Path(cmd[cmd.index("--json-out") + 1]).write_text(
            json.dumps({"lambada": {"lambada_acc": 0.25, "lambada_ppl": 80.0, "num_examples": 2}}))
        return type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(base_run.subprocess, "run", fake_yardstick)
    for step in (100, 200, 300, 400):
        base_run.score_checkpoint(run, run / "checkpoints" / MILESTONE_SUBDIR, step=step,
                                  limit=1000, on_cpu=True)
    with (run / base_run.JOURNAL).open("a") as fh:
        fh.write('{"step": 500, "source": "milest')

    launched = []
    _fake_scorers(monkeypatch, launched, [False])
    Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
               metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None, config=DEFAULTS).score_new_milestones()
    assert _steps(launched) == ["400"]


def test_the_supervisor_runs_one_scorer_at_a_time_oldest_first(tmp_path, monkeypatch):
    """#471: nine scorers spawned in one poll, ~1.5 GB each, OOM-killed the run. Five
    unscored milestones get one scorer while it runs, oldest step first (the incident's
    steps, which sort differently as text), and the next once it has finished."""
    from trm.runtime.supervisor import Limits, Supervisor
    from trm.settings import Config

    run = tmp_path / "run_c"
    _milestones(run, (500031, 62527, 31295, 7871, 3967))
    launched, running = [], [True]
    _fake_scorers(monkeypatch, launched, running)
    sup = Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
                     metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None,
                     config=Config.from_env({}))
    sup.score_new_milestones()
    sup.score_new_milestones()
    assert _steps(launched) == ["3967"], "one live scorer by default, and it is still running"
    running[0] = False
    sup.score_new_milestones()
    assert _steps(launched) == ["3967", "7871"]
    for _ in range(5):
        sup.score_new_milestones()
    assert _steps(launched) == ["3967", "7871", "31295", "62527", "500031"]

    # The knob raises the cap: a second scorer runs beside the first.
    launched.clear()
    running[0] = True
    sup = Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
                     metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None,
                     config=Config.from_env({"MILESTONE_SCORERS": "2"}))
    sup.score_new_milestones()
    sup.score_new_milestones()
    assert _steps(launched) == ["3967", "7871"]


def test_a_live_claim_from_a_predecessor_blocks_a_duplicate_and_a_dead_one_does_not(tmp_path, monkeypatch):
    """#506: a restarted supervisor could not see the scorer its predecessor left
    running, and started a second one on the same step (~1.5 GB, a duplicate journal
    line) — the cap was per supervisor. A claim is honoured while its pid is a live
    scorer; a dead one, or a pid that is now some other process, is not."""
    from trm.runtime.layout import YARDSTICK_CLAIMS
    from trm.runtime.supervisor import Limits, Supervisor

    run = tmp_path / "run_k"
    _milestones(run, (100, 200))
    (run / YARDSTICK_CLAIMS).mkdir()
    # A scorer as the supervisor spawns one: the argv carries the module, --run and --step.
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                                "instruments.base_run", "score", "--run", str(run), "--step", "100"])
    try:
        (run / YARDSTICK_CLAIMS / "100").write_text(str(foreign.pid))
        # The same live scorer, claimed for a step it is not on: a stale claim whose
        # pid a different scorer reused. Not honoured, and deleted.
        (run / YARDSTICK_CLAIMS / "300").write_text(str(foreign.pid))
        launched = []
        _fake_scorers(monkeypatch, launched, [False])

        def fresh(scorers):
            return Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
                              metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None,
                              config=Config.from_env({"MILESTONE_SCORERS": str(scorers)}))

        fresh(1).score_new_milestones()
        assert launched == [], "the predecessor's live scorer fills the cap of one"
        assert sorted(p.name for p in (run / YARDSTICK_CLAIMS).iterdir()) == ["100"], "300's claim is not its scorer's"
        fresh(2).score_new_milestones()
        assert _steps(launched) == ["200"], "a second slot, and never a duplicate of 100"
    finally:
        foreign.kill()
        foreign.wait()

    launched.clear()
    (run / YARDSTICK_CLAIMS / "200").write_text(str(os.getpid()))  # alive, but not a scorer
    fresh(2).score_new_milestones()
    assert _steps(launched) == ["100", "200"]


def test_the_run_end_names_each_milestone_left_unscored_with_its_command(tmp_path, monkeypatch):
    """#506: milestones still queued when run() ends were never scored, and nothing
    said so. The run stops at its first poll with 100 being scored and 200 waiting
    for the one slot; the end names 200 and the exact command, and leaves 100 to its
    running scorer."""
    import shlex
    import textwrap

    from trm.runtime import supervisor as sup_mod
    from trm.runtime.supervisor import BUDGET_COMPLETE, Limits, Supervisor

    run = tmp_path / "run_e"
    _milestones(run, (100, 200))
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import pathlib, time
        pathlib.Path({str(run / "metrics.csv")!r}).write_text("step,ce\\n50,3.0\\n")
        time.sleep(60)
    """))
    real_popen, launched = sup_mod.subprocess.Popen, []

    def popen(argv, **kw):
        if "instruments.base_run" not in argv:
            return real_popen(argv, **kw)  # the trainer stand-in
        launched.append(argv)
        return type("P", (), {"poll": lambda s: None, "pid": FAKE_PID})()

    monkeypatch.setattr(sup_mod.subprocess, "Popen", popen)
    reported = []
    sup = Supervisor(command=(sys.executable, str(child)), limits=Limits(stop_step=10), log_path=tmp_path / "t.log",
                     metrics_csv=run / "metrics.csv", spec=SPEC, report=reported.append, config=DEFAULTS,
                     poll_seconds=0.5)

    assert sup.run() == BUDGET_COMPLETE
    assert _steps(launched) == ["100"]
    (end,) = [line for line in reported if "unscored" in line]
    assert "unscored: 200" in end and "still running on 100" in end
    assert shlex.join(sup.scorer_argv("200")) in end
    assert "--step 100" not in end


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
                        lambda argv, **kw: launched.append(argv) or type("P", (), {"poll": lambda s: 0, "pid": FAKE_PID})())
    Supervisor(command=(), limits=Limits(stop_step=1), log_path=tmp_path / "t.log",
               metrics_csv=run / "metrics.csv", spec=SPEC, report=lambda m: None, config=DEFAULTS).score_new_milestones()
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


def test_a_failed_score_still_leaves_its_journal_line(tmp_path, monkeypatch):
    """The milestone scorer's output goes to /dev/null, so a failure that left no journal
    line would lose the milestone silently."""
    failed = type("Proc", (), {"returncode": 1, "stderr": "stubbed", "stdout": ""})()
    monkeypatch.setattr(base_run.subprocess, "run", lambda argv, **kw: failed)
    run = tmp_path / "run_f"
    (run / "checkpoints" / "40").mkdir(parents=True)
    assert base_run.main(["score", "--run", str(run), "--limit", "2", "--cpu"]) == 1
    assert base_run.journal(run)[-1]["error"] == "stubbed"


def test_only_the_runs_own_checkpoint_dir_is_journaled_as_rolling(tmp_path):
    """#334 review: any dir other than milestones/best was labelled 'rolling', so a
    --checkpoint-dir at a rewind's set_aside_* dir would pass for the rolling series."""
    run = tmp_path / "run_s"
    checkpoints = run / "checkpoints"
    assert base_run.checkpoint_source(checkpoints, run) == "rolling"
    assert base_run.checkpoint_source(checkpoints / "milestones", run) == "milestone"
    assert base_run.checkpoint_source(checkpoints / "best_val_ce", run) == "best"
    assert base_run.checkpoint_source(checkpoints / "set_aside_2026", run) == "set_aside_2026"


def test_a_card_without_a_recorded_recipe_does_not_claim_zero_tokens(tmp_path):
    """#343 review: an unrecorded ACCUMULATION/BATCH/SEQ recipe rendered 'Tokens seen 0'."""
    run = tmp_path / "run_norecipe"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"run_id": "run_norecipe", "parameters": {"PLAIN_LAYERS": 8}}))
    (run / "metrics.csv").write_text("step,ce\n40,3.1\n")
    fields = base_run.card_fields(run)
    assert fields["tokens_seen"] is None
    assert "| Tokens seen | unknown (recipe not recorded) (opt step 40) |" in base_run.render_card(fields)
