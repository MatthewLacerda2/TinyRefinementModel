"""instruments.pair: the card's sweep as one user unit (#565). Nothing here starts a unit:
systemctl and systemd-run are replaced, and the commands they would get are checked."""

import json
import subprocess

import pytest

from instruments import pair

SPEC = "experiments/recipe/specs/494-postnorm-long-pair.toml"


@pytest.fixture
def card(tmp_path, monkeypatch):
    """runs/ in tmp, and every subprocess recorded instead of run."""
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="inactive\n", stderr="")

    monkeypatch.setattr(pair, "RUNS", tmp_path)
    monkeypatch.setattr(pair, "CURRENT", tmp_path / "pair.current")
    monkeypatch.setattr(pair.subprocess, "run", run)
    monkeypatch.setattr(pair.shutil, "which", lambda name: None)  # no gh: no comment posted
    return calls


def _launches(calls):
    return [c for c in calls if c[0] == "systemd-run"]


def test_start_runs_the_runner_as_the_pair_unit_protected_and_logged(card, tmp_path):
    assert pair.main(["start", SPEC]) == 0
    (argv,) = _launches(card)
    assert f"--unit={pair.UNIT}" in argv and "OOMScoreAdjust=100" in argv
    assert f"StandardOutput=append:{tmp_path / '494-postnorm-long.log'}" in argv
    assert "KillMode=process" in argv, "a stop reaches the runner alone, which stops its arm in order"
    assert f"TimeoutStopSec={pair.STOP_TIMEOUT_S}" in argv
    assert argv[argv.index("--") + 1:][-2:] == ["instruments.experiment", SPEC]
    assert json.loads((tmp_path / "pair.current").read_text()) == {"spec": SPEC}


def test_resume_restarts_the_same_spec_without_the_gate(card):
    pair.main(["start", SPEC])
    pair.main(["resume"])
    assert _launches(card)[-1][-3:] == ["instruments.experiment", SPEC, "--no-gate"]


def test_a_second_start_is_refused_while_the_unit_runs(card, monkeypatch):
    monkeypatch.setattr(pair, "unit_state", lambda: "active")
    with pytest.raises(SystemExit, match="one sweep at a time"):
        pair.main(["start", SPEC])
    assert not _launches(card)


def test_pause_stops_the_unit(card):
    pair.main(["pause"])
    assert ["systemctl", "--user", "stop", pair.UNIT] in card


def test_resume_with_nothing_started_says_what_to_do(card):
    with pytest.raises(SystemExit, match="make pair SPEC="):
        pair.main(["resume"])


def test_the_running_arm_is_the_last_trainer_the_log_launched():
    log = ("$ python -m experiments.recipe.tokens_to_ce --seed 0\n"
           "run_a_s0: trainer pid 11, to opt step 2000\n"
           "RESULT {\"point\": \"run\"}\n"
           "run_a_s1: trainer pid 12, to opt step 2000\n")
    assert pair.running_arm(log) == ("run_a_s1", 2000)
    assert pair.running_arm("nothing launched yet\n") is None


def test_status_reads_the_journal_the_running_arm_and_the_disk(card, tmp_path, monkeypatch):
    pair.main(["start", SPEC])
    journal = tmp_path / "results.jsonl"
    journal.write_text(json.dumps({"arm": "control", "seed": 0, "point": "run"}) + "\n")
    monkeypatch.setattr(pair, "journal_path", lambda spec_id: journal)
    (tmp_path / "494-postnorm-long.log").write_text("run_x_s1: trainer pid 5, to opt step 2000\n")
    (tmp_path / "run_x_s1").mkdir()
    (tmp_path / "run_x_s1" / "metrics.csv").write_text("step,ce\n8,9.1\n16,8.2\n")
    report = pair.status()
    assert "1/6 arm-seeds recorded" in report
    assert "current arm run_x_s1: opt step 16 of 2000" in report
    assert "disk free on runs/" in report
