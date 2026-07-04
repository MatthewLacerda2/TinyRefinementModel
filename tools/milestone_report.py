"""One milestone report per checkpoint — every offline diagnostic, one document.

Convenience layer over the tools we already have; it adds no new instrumentation.
Against the latest (or a given) checkpoint it runs, in order:

  1. Depth curve — arch-aware, because the two arches ask different depth questions:
       reasoner: tools/eval_depth_curve.py (window-2 CE vs window-1 hunch depth;
                 its "fresh" baseline IS the carried-vs-fresh hunch comparison —
                 that diagnostic was folded into this tool, and the hunch is proven
                 inert: docs/findings/2026-06-13-cross-window-hunch-inert.md)
       refiner:  tools/eval_refiner_depth_transfer.py (depth-swept held-out CE by domain)
  2. Fixed-prompt transcripts — tools/dump_transcripts.py (the systematized vibes eval)
  3. Held-out validation CE — trainer.ValidationProbe, the same fixed batches and
     depth the training loop scores, so the number is comparable to the run's curve.

Sections 1–2 run as subprocesses (each tool is CLI-shaped, and isolation means one
crash costs one section, not the report); section 3 runs in-process, last, so this
process holds no accelerator memory while the children run. A failed section is
reported as FAILED with its error and the rest keep going.

The consolidated report prints to stdout and is saved under the run dir
(runs/<run>/milestone_report_step_<n>.md — gitignored with the rest of runs/).

Run when a checkpoint is worth a look (GPU free, or on CPU while training):
    PYTHONPATH=. python tools/milestone_report.py [--ckpt DIR] [--quick]
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_ITEMS = ("model", "optimizer", "monitor_state", "step")


def run_tool(script, extra_args=(), timeout=None):
    """Run a tools/ CLI as a subprocess and return its stdout, raising on failure."""
    cmd = [sys.executable, os.path.join("tools", script), *extra_args]
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(
            f"{script} exited {proc.returncode}\n"
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


def section_depth_curve(arch, fwd_args, batches, timeout):
    batch_args = [] if batches is None else ["--batches", str(batches)]
    if arch == "refiner":
        note = ("eval_depth_curve (carried-vs-fresh hunch) skipped: it probes the reasoner's "
                "cross-window hunch, which the refiner does not have. Depth-swept held-out CE "
                "by domain stands in as the depth diagnostic.")
        out = run_tool("eval_refiner_depth_transfer.py", [*fwd_args, *batch_args], timeout)
    else:
        note = ("carried-vs-fresh hunch is the 'fresh' baseline inside this curve; the hunch is "
                "proven inert (docs/findings/2026-06-13-cross-window-hunch-inert.md), so expect flat.")
        out = run_tool("eval_depth_curve.py", [*fwd_args, *batch_args], timeout)
    return f"{note}\n\n{out}"


def section_transcripts(fwd_args, quick_args, timeout):
    out = run_tool("dump_transcripts.py", [*fwd_args, *quick_args], timeout)
    # Inline the transcript file it saved — cleaner than the streamed stdout.
    for line in out.splitlines():
        if "Saved " in line:
            path = os.path.join(REPO_ROOT, line.split("Saved ", 1)[1].strip())
            if os.path.exists(path):
                with open(path) as f:
                    return f"(from {path})\n\n{f.read().strip()}"
    return out


def section_val_ce(checkpoint_path):
    from tools.common import restore_model
    import trainer

    if not trainer.DATA_ROOT:
        return "skipped: DATA_ROOT is not set — no held-out data to score"
    model, _ = restore_model(checkpoint_path)
    val_ce = trainer.ValidationProbe().run(model)
    if val_ce is None:
        return "no held-out validation data available (is DATA_ROOT set?)"
    return (f"validation CE: {val_ce:.4f} nats "
            f"(fixed depth {trainer.VAL_FIXED_DEPTH}, {trainer.VAL_BATCHES} batches, "
            f"skip {trainer.VAL_SKIP_SAMPLES:,} — same probe the training loop logs)")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="run all diagnostics against a checkpoint, emit one report")
    parser.add_argument("--ckpt", "--checkpoint-path", dest="checkpoint_path", default=None,
                        help="Orbax checkpoint dir (default: the latest run's)")
    parser.add_argument("--out", default=None,
                        help="report path (default: <run dir>/milestone_report_step_<n>.md)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny settings (2 batches, 2 prompts, 32 new tokens) for a fast smoke pass")
    parser.add_argument("--section-timeout", type=float, default=None,
                        help="seconds before a diagnostic subprocess is killed (default: none)")
    args = parser.parse_args()

    # Heavy imports after arg parsing so --help stays instant.
    from config import MODEL_ARCH
    from checkpoint_utils import discover_latest_checkpoint_run
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

    step = ocp.CheckpointManager(checkpoint_path, item_names=CKPT_ITEMS).latest_step()
    if step is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}.")
    print(f"📋 Milestone report: {source}, step {step}, arch '{MODEL_ARCH}'")

    # Forward the checkpoint only when the user overrode it; otherwise each tool
    # self-discovers the same latest run (and keeps its own output placement).
    fwd_args = ["--checkpoint-path", checkpoint_path] if args.checkpoint_path else []
    batches = 2 if args.quick else None
    transcript_args = ["--prompts", "2", "--max-new-tokens", "32"] if args.quick else []

    sections = [
        run_section("Depth curve", lambda: section_depth_curve(
            MODEL_ARCH, fwd_args, batches, args.section_timeout)),
        run_section("Fixed-prompt transcripts", lambda: section_transcripts(
            fwd_args, transcript_args, args.section_timeout)),
        run_section("Held-out validation CE", lambda: section_val_ce(checkpoint_path)),
    ]

    lines = [
        f"# Milestone report — checkpoint step {step}",
        "",
        f"- generated: {datetime.datetime.now().astimezone().isoformat()}",
        f"- arch: {MODEL_ARCH}",
        f"- checkpoint: {checkpoint_path}",
        f"- commit: {git_commit()}",
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
