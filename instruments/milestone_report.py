"""One milestone report per checkpoint — every offline diagnostic, one document.

Convenience layer over the tools we already have; it adds no new instrumentation.
Against the latest (or a given) checkpoint it runs, in order:

  1. Fixed-prompt transcripts — instruments.dump_transcripts (the systematized vibes eval)
  2. Held-out validation CE — trm.train.validation.ValidationProbe, the same fixed
     rows the training loop scores, so the number is comparable to the run's curve.

Section 1 runs as a subprocess (the tool is CLI-shaped, and isolation means one
crash costs one section, not the report); section 2 runs in-process, last, so this
process holds no accelerator memory while the children run. A failed section is
reported as FAILED with its error and the rest keep going.

The consolidated report prints to stdout and is saved under the run dir
(runs/<run>/milestone_report_step_<n>.md — gitignored with the rest of runs/).

Run when a checkpoint is worth a look (GPU free, or on CPU while training):
    PYTHONPATH=. python -m instruments.milestone_report [--ckpt DIR] [--quick]
On CPU prepend FORCE_F32_COMPUTE=1 JAX_PLATFORMS=cpu (see config.py).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import datetime
import subprocess
import sys
import time
import traceback

from instruments._common import REPO_ROOT, add_checkpoint_argument, git_head, module_env
from trm.runtime.layout import CHECKPOINT_ITEMS  # jax-free, so --help stays instant

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # assembles other tools' sections; each number is declared by the tool that produced it

# Where this differs from production's environment, and why (#166).
ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "an eval that may share the card with a training run"}

# What `--quick` hands dump_transcripts. A name, so a test can feed this exact argv to
# that tool's real parser: it once passed a flag the tool did not define (#335).
QUICK_TRANSCRIPT_ARGS = ("--prompts", "2", "--max-new-tokens", "32")


def run_tool(module, extra_args=(), timeout=None):
    """Run a sibling CLI as a subprocess and return its stdout, raising on failure.

    Invoked as `python -m <module>` (#143): the diagnostics live in packages now,
    so a module path is what identifies them — and it keeps working wherever the
    file sits inside its package."""
    cmd = [sys.executable, "-m", module, *extra_args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=module_env(), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(
            f"{module} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout.strip()}\n--- stderr (tail) ---\n{stderr_tail}"
        )
    return proc.stdout.strip()


def run_section(title, fn):
    """Run one diagnostic and tolerate its failure: capture the error, keep going."""
    print(f"\n=== {title} ===")
    start = time.time()
    try:
        body, status = fn(), "ok"
    except Exception:
        body, status = traceback.format_exc(), "FAILED"
    return {"title": title, "status": status, "elapsed": time.time() - start, "body": body}


def section_transcripts(fwd_args, quick_args, timeout):
    """Run dump_transcripts and embed the transcript file it wrote, found through its
    contract line (`dump_transcripts.written_transcript`), not its human output. The old
    scan looked for "Saved " while the tool printed "✨ <path>", so no report ever
    embedded a transcript (#338). If no file is named, the raw output is kept and says so."""
    from instruments.dump_transcripts import written_transcript

    out = run_tool("instruments.dump_transcripts", [*fwd_args, *quick_args], timeout)
    path = written_transcript(out)
    if path and os.path.exists(path):
        with open(path) as f:
            return f"(from {path})\n\n{f.read().strip()}"
    reason = f"named {path}, which does not exist" if path else "named no transcript file"
    return f"(dump_transcripts {reason}; its output follows)\n\n{out}"


def section_val_ce(checkpoint_path):
    from trm.config import resolve_root
    from trm.runtime.restore import restore_model
    from trm.settings import CONFIG
    from trm.train.validation import ValidationProbe

    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        return "skipped: DATA_ROOT is not set — no held-out data to score"
    model, _ = restore_model(CONFIG, checkpoint_path)
    probe = ValidationProbe.of(CONFIG, resolve_root(data_root))
    val_ce = probe.run(model)
    if val_ce is None:
        return f"no held-out validation data under {data_root}"
    return (f"validation CE: {val_ce:.4f} nats "
            f"({probe.rows} rows, "
            f"skip {probe.skip:,} — same probe the training loop logs)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="run all diagnostics against a checkpoint, emit one report")
    add_checkpoint_argument(parser, aliases=("--ckpt",))
    parser.add_argument("--out", default=None,
                        help="report path (default: <run dir>/milestone_report_step_<n>.md)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny settings (the first 2 standard prompts, 32 new tokens) "
                             "for a fast smoke pass")
    parser.add_argument("--section-timeout", type=float, default=None,
                        help="seconds before a diagnostic subprocess is killed (default: none)")
    args = parser.parse_args(argv)

    # Heavy imports after arg parsing so --help stays instant.
    from trm.runtime.checkpoints import discover_latest_checkpoint_run
    import orbax.checkpoint as ocp

    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        report_dir = os.path.dirname(checkpoint_path)
        source = checkpoint_path
    else:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpoint found: no run under runs/ has a readable checkpoint "
                             "(train first, or point at one with --ckpt).")
        checkpoint_path = os.path.abspath(checkpoint_path)
        report_dir = os.path.join("runs", run_id)
        source = f"latest run {run_id}"
        print(f"🔎 Using latest checkpointed run: {run_id}")

    step = ocp.CheckpointManager(checkpoint_path, item_names=CHECKPOINT_ITEMS).latest_step()
    if step is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}.")
    print(f"📋 Milestone report: {source}, step {step}")

    # Forward the checkpoint only when the user overrode it; otherwise each tool
    # self-discovers the same latest run (and keeps its own output placement).
    fwd_args = ["--checkpoint-path", checkpoint_path] if args.checkpoint_path else []
    transcript_args = list(QUICK_TRANSCRIPT_ARGS) if args.quick else []

    sections = [
        run_section("Fixed-prompt transcripts", lambda: section_transcripts(
            fwd_args, transcript_args, args.section_timeout)),
        run_section("Held-out validation CE", lambda: section_val_ce(checkpoint_path)),
    ]

    lines = [
        f"# Milestone report — checkpoint step {step}",
        "",
        f"- generated: {datetime.datetime.now().astimezone().isoformat()}",
        f"- checkpoint: {checkpoint_path}",
        f"- commit: {git_head() or 'unknown'}",
        "",
    ]
    for s in sections:
        lines += [f"## {s['title']} — {s['status']} ({s['elapsed']:.0f}s)", "", "```", s["body"], "```", ""]
    report = "\n".join(lines)

    print("\n" + report)
    out_path = args.out or os.path.join(report_dir, f"milestone_report_step_{step}.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"✨ Saved {out_path}")

    if all(s["status"] == "FAILED" for s in sections):
        raise SystemExit("Every section failed — see the report above.")


if __name__ == "__main__":
    main()
