"""The fit gate: the real trainer, briefly, before a launch commits the card (#168).
Split out of test_supervisor.py (#325)."""

import sys
import textwrap


from trm.runtime.supervisor import (
    GpuLock,
)


# --- the fit gate: the real trainer, briefly, before committing the card (#168) ---

def _gate(tmp_path, body, **kw):
    from trm.runtime.supervisor import preflight_fit
    script = tmp_path / "fake_trainer.py"
    script.write_text(textwrap.dedent(body))
    return preflight_fit(command=[sys.executable, str(script)], workroot=tmp_path / "runs",
                         poll_s=0.1, tokens_per_opt_step=1000, **kw)


def test_the_gate_passes_on_a_first_logged_row_and_leaves_nothing_behind(tmp_path):
    (tmp_path / "runs").mkdir()
    result = _gate(tmp_path, """
        import os, pathlib, time
        assert os.environ["VAL_EVERY_OPT_STEPS"] == "1" and os.environ["CHECKPOINT_EVERY_OPT_STEPS"] == "1"
        print("Step 0005 | CE: 9.1 | Compute: 2.000s", flush=True)
        p = pathlib.Path("runs/run_fitgate/metrics.csv")
        p.write_text("step,ce,arena_peak_mib\\n5,9.1,4112\\n")
        time.sleep(60)
    """)
    assert result.ok and result.arena_peak_mib == 4112.0
    assert result.tokens_per_second == 5 * 1000 / 2.0
    assert list((tmp_path / "runs").iterdir()) == [], "the probe's directory must be removed"


def test_the_gate_refuses_an_out_of_memory(tmp_path):
    (tmp_path / "runs").mkdir()
    result = _gate(tmp_path, """
        import time
        print("2026-08-13 E external/xla: RESOURCE_EXHAUSTED: Out of memory allocating 626MiB", flush=True)
        time.sleep(60)
    """)
    assert not result.ok and "does not fit" in result.reason
    assert list((tmp_path / "runs").iterdir()) == []


def test_the_gate_refuses_a_trainer_that_dies_or_never_logs(tmp_path):
    (tmp_path / "runs").mkdir()
    died = _gate(tmp_path, """
        import sys
        print("Traceback: data root missing", flush=True)
        sys.exit(3)
    """)
    assert not died.ok and "exited (3)" in died.reason and "data root missing" in died.reason
    hung = _gate(tmp_path, "import time\ntime.sleep(60)\n", timeout_s=1.0)
    assert not hung.ok and "no logged row" in hung.reason
    assert list((tmp_path / "runs").iterdir()) == []


def test_the_gate_never_points_the_probe_at_the_real_run():
    from trm.runtime.supervisor import _without_path_flags
    assert _without_path_flags(["--checkpoint-path", "runs/run_real/checkpoints", "--new-run", "--x", "1"]) == ["--x", "1"]
    assert _without_path_flags(["--checkpoint-path=runs/run_real/checkpoints"]) == []


def test_a_refused_gate_stops_the_launch_and_releases_the_card(tmp_path, monkeypatch):
    from trm.runtime import supervisor as sup_mod

    launched = []

    class Stub:
        def __init__(self, **kw):
            pass

        def run(self):
            launched.append(True)
            return sup_mod.BUDGET_COMPLETE

    lock_path = tmp_path / "gpu.lock"
    monkeypatch.setattr(sup_mod, "Supervisor", Stub)
    monkeypatch.setattr(sup_mod, "GpuLock", lambda label="": GpuLock(lock_path, label))
    monkeypatch.setattr(sup_mod, "preflight_fit",
                        lambda args: sup_mod.FitResult(False, "out of memory — this config does not fit"))
    code = sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log")])
    assert code == 1 and launched == []
    assert not lock_path.exists(), "a refused launch must not keep holding the card"

    monkeypatch.setattr(sup_mod, "preflight_fit", lambda args: (_ for _ in ()).throw(AssertionError("gate ran")))
    assert sup_mod.main(["--stop-step", "10", "--run-dir", str(tmp_path / "run_x"), "--log", str(tmp_path / "t.log"),
                         "--skip-fit-gate"]) == 0
    assert launched == [True]


def test_a_killed_gates_leftovers_are_swept_by_the_next_gate(tmp_path):
    runs = tmp_path / "runs"
    (runs / ".fitgate_20260101_000000" / "runs").mkdir(parents=True)
    (runs / "run_real").mkdir()
    _gate(tmp_path, "import sys\nsys.exit(1)\n")
    assert sorted(p.name for p in runs.iterdir()) == ["run_real"], "sweep probe dirs, never a real run"
