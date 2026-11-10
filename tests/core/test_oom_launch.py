"""A long run is not born the kernel's first OOM victim (#524): it starts through the
user manager at OOMScoreAdjust=100 where there is one, and says so loudly where
there is not."""

import os
import pathlib
import sys

import pytest

from trm.runtime import oom

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the lint job (#325).
pytestmark = pytest.mark.jaxfree

ARGV = ["python", "-m", "trm.runtime.supervisor", "--stop-step", "512"]


def _flags(cmd):
    return cmd[:cmd.index("--")]


def test_a_detached_launch_asks_for_oom_score_adj_100(tmp_path):
    cmd = oom.systemd_run(ARGV, {"TRAIN_TOKEN_BUDGET": "4e9", "BASH_FUNC_x%%": "()"}, tmp_path,
                          "trm-run.service", stdout=tmp_path / "out")
    assert cmd[:2] == ["systemd-run", "--user"]
    assert cmd[cmd.index("OOMScoreAdjust=100") - 1] == "-p"
    assert cmd[cmd.index("--") + 1:] == ARGV, "the run's own argv, untouched, after --"
    flags = _flags(cmd)
    assert f"StandardOutput=append:{tmp_path / 'out'}" in flags and "--wait" not in flags
    assert "KillMode=process" in flags, "a milestone scorer outlives its supervisor, as before"
    assert f"--working-directory={tmp_path}" in flags
    assert "--setenv=TRAIN_TOKEN_BUDGET=4e9" in flags, "a knob read from the environment reaches the run"
    assert not any("BASH_FUNC" in f for f in flags), "a name systemd would refuse is not passed"


def test_a_foreground_rerun_pipes_and_waits(tmp_path):
    flags = _flags(oom.systemd_run(ARGV, {}, tmp_path, "trm-x.service"))
    assert "OOMScoreAdjust=100" in flags and {"--pipe", "--wait"} <= set(flags)
    assert not any(f.startswith("StandardOutput") for f in flags)


def test_without_a_user_manager_the_run_still_starts_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(oom, "user_manager_problem", lambda: "systemd --user is offline")
    out = tmp_path / "supervisor.out"
    pid = oom.detach([sys.executable, "-c", "print('started')"], {}, tmp_path, out, label="run_x")
    assert pid is not None
    assert "systemd --user is offline" in capsys.readouterr().out
    os.waitpid(pid, 0)
    assert out.read_text().strip() == "started"


def test_a_failed_systemd_run_falls_back_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(oom, "user_manager_problem", lambda: None)
    monkeypatch.setattr(oom, "systemd_run", lambda *a, **k: [sys.executable, "-c", "import sys; sys.exit(3)"])
    out = tmp_path / "supervisor.out"
    pid = oom.detach([sys.executable, "-c", "pass"], {}, tmp_path, out, label="run_x")
    assert pid is not None and "systemd-run failed (exit 3" in capsys.readouterr().out


def test_a_rerun_is_skipped_inside_itself_or_without_a_user_manager(monkeypatch, capsys):
    monkeypatch.setattr(oom, "oom_score_adj", lambda pid="self": 200)
    assert oom.rerun_protected(ARGV, {oom.RERUN_MARKER: "1"}, pathlib.Path("."), "spec") is None
    monkeypatch.setattr(oom, "user_manager_problem", lambda: "there is no systemd-run")
    assert oom.rerun_protected(ARGV, {}, pathlib.Path("."), "spec") is None
    assert "OOM killer may take it first" in capsys.readouterr().out
    monkeypatch.setattr(oom, "oom_score_adj", lambda pid="self": 100)
    assert oom.rerun_protected(ARGV, {}, pathlib.Path("."), "spec") is None, "already protected"


def test_the_banner_is_loud_above_100(monkeypatch):
    monkeypatch.setattr(oom, "oom_score_adj", lambda pid="self": 200)
    assert "⚠" in oom.banner(4242) and "200" in oom.banner(4242)
    monkeypatch.setattr(oom, "oom_score_adj", lambda pid="self": 100)
    assert "⚠" not in oom.banner(4242) and "oom_score_adj 100" in oom.banner(4242)
    assert "unknown" in oom.banner(None)


def test_unit_names_are_valid_and_carry_the_label():
    name = oom.unit_name("run_20260929 x/y")
    assert name.startswith("trm-run_20260929-x-y-") and name.endswith(".service")


def test_a_term_to_the_waiting_runner_stops_its_unit(monkeypatch, tmp_path):
    """#519's TERM handling lives inside the unit; the waiting outer runner forwards
    a TERM as `systemctl --user stop`, or the sweep would stay on the card."""
    import signal
    import subprocess
    import threading
    monkeypatch.setattr(oom, "oom_score_adj", lambda pid="self": 200)
    monkeypatch.setattr(oom, "user_manager_problem", lambda: None)
    monkeypatch.setattr(oom, "systemd_run", lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(60)"])
    stopped, real_run = [], subprocess.run
    popen = subprocess.Popen

    def fake_popen(cmd, *a, **k):
        fake_popen.proc = popen(cmd, *a, **k)
        return fake_popen.proc

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["systemctl", "--user", "stop"]:
            stopped.append(cmd[3])
            fake_popen.proc.terminate()
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(oom.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(oom.subprocess, "run", fake_run)
    before = signal.getsignal(signal.SIGTERM)
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
    code = oom.rerun_protected(ARGV, {}, tmp_path, "spec")
    assert code == -signal.SIGTERM and len(stopped) == 1 and stopped[0].startswith("trm-spec-")
    assert signal.getsignal(signal.SIGTERM) is before, "the runner's own handler is restored"
