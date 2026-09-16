"""Milestone report wrapper: cheap CPU checks, no checkpoint or GPU needed.

The wrapper must tolerate a failing diagnostic (report it, keep going) and
degrade gracefully when there is nothing to report on.
"""

import os
import pathlib
import subprocess
import sys

# Marker-anchored, not a fixed parent-hop count (see tests/core/test_seed_config.py).
REPO_ROOT = str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()))
# Invoked as a module (#143), the way milestone_report runs its own sub-tools.
MODULE = "instruments.milestone_report"


def test_section_failure_is_tolerated_and_labeled():
    from instruments.milestone_report import run_section

    failed = run_section("boom", lambda: 1 / 0)
    assert failed["status"] == "FAILED"
    assert "ZeroDivisionError" in failed["body"]

    ok = run_section("fine", lambda: "all good")
    assert ok["status"] == "ok"
    assert ok["body"] == "all good"


def _run(args, cwd):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, JAX_PLATFORMS="cpu", FORCE_F32_COMPUTE="1")
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=300)


def test_help_is_fast_and_clean(tmp_path):
    proc = _run(["--help"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "--ckpt" in proc.stdout


def test_no_checkpoint_degrades_gracefully(tmp_path):
    # Empty cwd: no runs/ at all. Must exit nonzero with a clear message,
    # not a traceback.
    proc = _run([], cwd=tmp_path)
    assert proc.returncode != 0
    assert "No checkpoint found" in proc.stderr + proc.stdout
    assert "Traceback" not in proc.stderr


def test_the_depth_section_says_plain_has_no_dial_not_that_it_is_the_reasoner():
    """#314: plain fell through to the reasoner's explanation."""
    from instruments.milestone_report import section_depth_curve

    body = section_depth_curve("plain", [], None, None)
    assert "plain" in body and "reasoner" not in body


def test_the_val_ce_section_drives_the_real_probe(monkeypatch, tmp_path):
    """#314: section 3 called `trainer.ValidationProbe()` without its data_root and read
    constants (VAL_BATCHES among them) that exist nowhere, so every report printed it as
    FAILED. It must build the probe the trainer builds and name the probe's real knobs."""
    from instruments.milestone_report import section_val_ce
    from trm.runtime import restore
    from trm.train import validation

    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert section_val_ce("unused").startswith("skipped")

    built = {}

    class FakeProbe:
        def __init__(self, data_root):
            built["data_root"] = data_root

        def run(self, model):
            built["model"] = model
            return 3.25

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(restore, "restore_model", lambda path: (f"model@{path}", 7))
    monkeypatch.setattr(validation, "ValidationProbe", FakeProbe)
    body = section_val_ce("ckpt")
    assert built == {"data_root": str(tmp_path), "model": "model@ckpt"}
    assert body.startswith("validation CE: 3.2500 nats")
    assert f"fixed depth {validation.VAL_FIXED_DEPTH}, {validation.VAL_ROWS} rows" in body


def _dump_transcripts_stdout(capsys, path):
    """What dump_transcripts really prints at the end: its human line, then the contract line."""
    from instruments.dump_transcripts import announce_written

    capsys.readouterr()
    print(f"\n✨ {path}")
    announce_written(path)
    return "▶ 2 prompts x 1 depths = 2 completions, seed 42, on cpu\n" + capsys.readouterr().out


def test_the_transcripts_section_embeds_the_file_dump_transcripts_wrote(tmp_path, monkeypatch, capsys):
    """#338: the section scanned for "Saved " while dump_transcripts printed "✨ <path>", so
    every report embedded the progress output instead of the transcript."""
    from instruments import milestone_report

    transcript = tmp_path / "step_000184_cpu.md"
    transcript.write_text("---\nstep: 184\n---\n# Transcripts — opt step 184\n")
    stdout = _dump_transcripts_stdout(capsys, transcript)
    monkeypatch.setattr(milestone_report, "run_tool", lambda module, argv, timeout: stdout)

    body = milestone_report.section_transcripts([], [], None)
    assert body.startswith(f"(from {transcript})")
    assert "# Transcripts — opt step 184" in body and "▶ 2 prompts" not in body


def test_no_transcript_named_keeps_the_output_and_says_so(monkeypatch):
    from instruments import milestone_report

    monkeypatch.setattr(milestone_report, "run_tool", lambda module, argv, timeout: "▶ progress only")
    body = milestone_report.section_transcripts([], [], None)
    assert "named no transcript file" in body and "▶ progress only" in body


def test_main_quick_hands_dump_transcripts_the_quick_argv(tmp_path, monkeypatch):
    """#338 review: the #335 test drove section_transcripts with QUICK_TRANSCRIPT_ARGS
    directly, so main() could pass anything under --quick. Drive main() itself. Imports
    trm.config and orbax, so it runs in CI's pytest job."""
    import orbax.checkpoint as ocp

    from instruments import milestone_report
    from instruments.dump_transcripts import build_arg_parser

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def latest_step(self):
            return 23551

    sent = []
    monkeypatch.setattr(ocp, "CheckpointManager", Manager)
    monkeypatch.setattr(milestone_report, "run_tool", lambda module, argv, timeout: sent.append((module, argv)) or "")
    monkeypatch.setattr(milestone_report, "section_val_ce", lambda path: "stubbed")
    checkpoints = tmp_path / "run_q" / "checkpoints"
    checkpoints.mkdir(parents=True)

    milestone_report.main(["--quick", "--checkpoint-path", str(checkpoints), "--out", str(tmp_path / "report.md")])

    (argv,) = [argv for module, argv in sent if module == "instruments.dump_transcripts"]
    args = build_arg_parser().parse_args(argv)
    assert (args.checkpoint_path, args.prompts, args.max_new_tokens) == (str(checkpoints), 2, 32)
