"""Time machine — revive any stored weight in the exact world that trained it.

New code is not compatible with old weights: param-tree renames, config drift, and
orbax/flax checkpoint-format changes all break a naive `load`. So reviving a weight
means reconstructing the *world* that made it, not just pointing new code at an old
checkpoint. This tool rebuilds that world — the model/config code at the training
commit, the uncommitted edits that were live at launch, and the pinned Python libs —
then aims it at the SAME weights on disk and lets you fork a new training lineage or
run inference.

Why worktree + venv, not Docker: the compatibility surface is exactly (code, libs).
CUDA userspace is pinned *inside* `env_freeze.txt` (the `nvidia-cu12` wheels) and the
driver is a shared host passthrough — so a git worktree at the commit plus a venv
built from the freeze reproduces the whole surface. Docker would only wrap that in an
OS/driver jail on top, for 5-8GB an image on a tight SSD. The snapshot is structured
so a `build-image` step could wrap it later if that jail is ever wanted; see
docs/plans/timemachine.md.

The load-bearing fact: `runs/` is gitignored, so a worktree checked out at any old
commit shares the *same* `runs/<id>/checkpoints` on disk. The time machine reverts
code; the weights folder never moves. "Same folder" is free.

Usage:
    python -m instruments.timemachine list
    python -m instruments.timemachine reconstruct <run-id> [--for infer|train] [--no-venv]
    python -m instruments.timemachine fork <run-id> <new-name> [--no-venv]
"""

import os
import sys
import json
import math
import shutil
import hashlib
import argparse
import subprocess

from instruments import runlog
from trm.runtime.layout import LOG_REAL_STEPS  # standard library only
from instruments._common import REPO_ROOT as _REPO_ROOT, module_env

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # reconstructs a run's world; prints paths and provenance, no quantities

REPO_ROOT = str(_REPO_ROOT)
TM_ROOT = os.path.join(REPO_ROOT, ".timemachine")
RUNS_ROOT = os.path.join(REPO_ROOT, "runs")


# --- reading a run's snapshot ------------------------------------------------

def run_dir(run_id):
    return os.path.join(RUNS_ROOT, run_id)


def load_meta(run_id):
    """The run's metadata; a revival cannot proceed without it, so none is a SystemExit."""
    meta = runlog.read_metadata(run_dir(run_id))
    if not meta:
        path = os.path.join(run_dir(run_id), runlog.METADATA_FILENAME)
        raise SystemExit(f"No readable run_metadata.json for {run_id} (looked in {path}).")
    return meta


def resolve_arch(run_id, meta):
    """Which arch built the checkpoint's param tree, or None if unknowable.

    The two arches are not checkpoint-compatible, so guessing wrong rebuilds the
    WRONG skeleton and the restore corrupts silently. We therefore never default:
    prefer the machine-readable record (run_metadata parameters, or the
    system_snapshot line for runs after that capture landed), fall back to the
    older free-text 'arm=<x>', and otherwise return None so the caller demands an
    explicit --arch rather than gambling on the project default."""
    arch = meta.get("parameters", {}).get("MODEL_ARCH")
    if arch:
        return arch
    snap = os.path.join(run_dir(run_id), "system_snapshot.txt")
    if os.path.exists(snap):
        text = open(snap).read()
        for marker in ("MODEL_ARCH ", "arm="):  # new structured line, then legacy
            if marker in text:
                return text.split(marker, 1)[1].split()[0].strip("();,")
    return None


def has_file(run_id, name):
    p = os.path.join(run_dir(run_id), name)
    return os.path.exists(p) and os.path.getsize(p) > 0


# --- reconstructing the world ------------------------------------------------

def ensure_worktree(run_id, commit):
    """A detached worktree at `commit` with the run's dirty edits re-applied.
    Idempotent: an existing worktree is reused (patch applied only once, tracked
    by a sentinel so a re-run never double-applies)."""
    wt = os.path.join(TM_ROOT, "wt", run_id)
    applied = os.path.join(wt, ".tm_applied")
    if os.path.exists(applied):
        print(f"  worktree reused: {os.path.relpath(wt, REPO_ROOT)}")
        return wt

    os.makedirs(os.path.dirname(wt), exist_ok=True)
    if not os.path.exists(wt):
        subprocess.run(["git", "worktree", "add", "--detach", wt, commit],
                       cwd=REPO_ROOT, check=True)
    else:
        # Worktree exists but the sentinel doesn't — a prior run died mid-setup,
        # leaving a possibly half-patched tree. Reset it clean before re-applying so
        # the patch lands on the pristine commit, not on top of a partial apply.
        subprocess.run(["git", "reset", "--hard", commit], cwd=wt, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=wt, check=True)

    patch = os.path.join(run_dir(run_id), "worktree.patch")
    if has_file(run_id, "worktree.patch"):
        # --3way falls back to a merge if context drifted; a clean run applies flat.
        subprocess.run(["git", "apply", "--3way", patch], cwd=wt, check=True)
        print(f"  applied worktree.patch ({os.path.getsize(patch)} bytes)")
    else:
        print("  no worktree.patch — reconstructed to the commit only "
              "(pre-snapshot run; dirty edits at launch are not recoverable)")

    untracked = os.path.join(run_dir(run_id), "worktree.untracked.txt")
    if os.path.exists(untracked):
        n = len([line for line in open(untracked) if line.strip()])
        if n:
            print(f"  note: {n} untracked non-ignored files existed at launch "
                  f"(not restored; see {os.path.relpath(untracked, REPO_ROOT)})")

    open(applied, "w").close()
    return wt


def venv_key(run_id):
    """A venv is shared by every lineage with a byte-identical env_freeze.txt."""
    freeze = os.path.join(run_dir(run_id), "env_freeze.txt")
    if not os.path.exists(freeze):
        return None, None
    digest = hashlib.sha256(open(freeze, "rb").read()).hexdigest()[:16]
    return digest, freeze


def ensure_venv(run_id, build=True):
    """A venv matching the run's pinned freeze, cached by its hash. Building
    installs ~5GB of wheels (incl. the vendored CUDA libs) and needs the network,
    so it is the slow step — but it happens once per distinct environment."""
    digest, freeze = venv_key(run_id)
    if digest is None:
        print("  no env_freeze.txt — cannot pin libs (using the caller's venv)")
        return None
    venv = os.path.join(TM_ROOT, "venvs", digest)
    py = os.path.join(venv, "bin", "python")
    if os.path.exists(py):
        print(f"  venv reused: .timemachine/venvs/{digest}")
        return venv
    if not build:
        print(f"  venv NOT built (--no-venv): would install {freeze} -> "
              f".timemachine/venvs/{digest}")
        return venv
    print(f"  building venv .timemachine/venvs/{digest} from {os.path.basename(freeze)} "
          f"(installs ~5GB, one-off)...")
    os.makedirs(os.path.dirname(venv), exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([py, "-m", "pip", "install", "-r", freeze], check=True)
    return venv


def reconstruct(run_id, for_mode="infer", build_venv=True, arch_override=None):
    meta = load_meta(run_id)
    commit = meta.get("git_commit", "unknown")
    arch = arch_override or resolve_arch(run_id, meta)
    if commit in (None, "unknown"):
        raise SystemExit(f"{run_id} has no recorded commit — cannot reconstruct.")
    if arch is None:
        raise SystemExit(
            f"{run_id} does not record its MODEL_ARCH (a pre-capture run), and the "
            f"arches are not checkpoint-compatible. Re-run with --arch "
            f"plain|refiner|reasoner to say which skeleton to rebuild.")

    print(f"Reconstructing world for {run_id}  (commit {commit[:10]}, arch {arch})")
    wt = ensure_worktree(run_id, commit)
    venv = ensure_venv(run_id, build=build_venv)
    ckpt = os.path.join(run_dir(run_id), "checkpoints")
    py = os.path.join(venv, "bin", "python") if venv else sys.executable

    print(f"\nWORLD READY: {run_id}")
    print(f"  worktree : {wt}")
    print(f"  venv     : {venv or '(caller venv — libs NOT pinned)'}")
    print(f"  weights  : {ckpt}")
    print(f"  arch     : {arch}")
    _print_commands(wt, py, ckpt, arch, run_id, for_mode)
    return {"worktree": wt, "python": py, "checkpoints": ckpt, "arch": arch}


def _print_commands(wt, py, ckpt, arch, run_id, for_mode):
    if for_mode == "eval":
        return  # evaluate() prints its own eval command + verdict
    print("\nRun inside the reconstructed world:")
    if for_mode == "infer":
        print(f"  cd {wt}")
        print(f"  PYTHONPATH=. MODEL_ARCH={arch} CHECKPOINT_ROOT={ckpt} \\")
        print(f"    {py} -m trm.infer")
        print("  (a worktree older than the trm/ package has no trm.infer; "
              "run that commit's own infer script instead)")
    else:
        print(f"  # fork a NEW lineage (original {run_id} preserved):")
        print(f"  python -m instruments.timemachine fork {run_id} <new-name>")
        print("  # ...on the 6GB card, also export the run's memory knobs "
              "(XLA_PYTHON_CLIENT_MEM_FRACTION etc.) from its "
              "launch env / system_snapshot.txt, or it will OOM.")


# --- closing the loop: reproduce the metric (#44 DoD) ------------------------

def recorded_val_ces(run_id):
    """{opt step: val CE} the run logged, finite readings only, parsed once through
    runlog (replayed and torn rows dropped). {} when the run has no metrics.csv."""
    try:
        log = runlog.load(run_dir(run_id))
    except FileNotFoundError:
        return {}
    # At the probe's own step when the run recorded it (#351), else the logged row's.
    return {s: v for s, v in zip(*log.val_readings()) if math.isfinite(v)}


def checkpoint_opt_step(checkpoint_step, accumulation_steps):
    """The opt step of an orbax checkpoint. Orbax names a checkpoint by MICRO-step and the
    trainer resumes at n+1, so opt step = (n + 1) // ACCUMULATION_STEPS, exactly as
    trm/runtime/rewind.py computes it: run_287's opt step 184 is checkpoint 23551."""
    return (int(checkpoint_step) + 1) // int(accumulation_steps)


def val_ce_for_checkpoint(val_ces, checkpoint_step, accumulation_steps, val_every, log_every=LOG_REAL_STEPS):
    """(val CE or None, how it was found) for the checkpoint the yardstick scored.

    The run's own held-out val CE is the number a faithful revival must reproduce; it
    is what the #17 noise floor is stated in (2σ = 0.06 nats). Two offsets stand
    between a checkpoint and its row in metrics.csv:

    * orbax names the checkpoint by MICRO-step and metrics.csv by opt step (#348), so
      the checkpoint is converted with the run's RECORDED accumulation, never today's;
    * the trainer does not write a probe's val CE on the probe's own opt step. It holds
      the latest probe's value and writes it on the next logging row, every
      LOG_REAL_STEPS opt steps (#351), so a row holds the LAST probe in (row - 5, row].

    That row is the checkpoint's own probe only when both hold, and both are checked:
    the checkpoint's opt step N is a probe step (N % VAL_EVERY_OPT_STEPS == 0, from the
    run's recorded value), and probes are at least LOG_REAL_STEPS apart, so no later
    probe can overwrite N's value before the row is written. Then the value is on the
    first val row in [N, N + LOG_REAL_STEPS). On run_287 (probes every 8): checkpoint
    23551 is opt 184, whose value is on row 185, not the earlier probe's row 180.

    Anything else returns None with the reason: a missing recording, a checkpoint that
    is not a probe step, probes too close to tell apart (the fit gate probes every
    step), or no row in the window. LOG_REAL_STEPS is not recorded per run (#351); this
    uses today's value from trm/runtime/layout.py, which every run so far has used."""
    if not val_ces:
        return None, "no val CE in metrics.csv"
    if not accumulation_steps:
        return None, (f"run_metadata.json records no ACCUMULATION_STEPS, so checkpoint "
                      f"{checkpoint_step} cannot be placed on the opt-step axis metrics.csv uses")
    if not val_every:
        return None, ("run_metadata.json records no VAL_EVERY_OPT_STEPS, so no val row can be "
                      "tied to one probe")
    val_every = int(val_every)
    opt_step = checkpoint_opt_step(checkpoint_step, accumulation_steps)
    if opt_step % val_every:
        return None, (f"checkpoint {checkpoint_step} is opt step {opt_step}, which is not a probe step "
                      f"(the run probed every {val_every}), so no val CE was measured on these weights")
    if val_every < log_every:
        return None, (f"the run probed every {val_every} opt steps, closer than its {log_every}-step "
                      f"logging, so each row holds only the last of several probes and cannot be "
                      f"tied to opt step {opt_step}")
    window = range(opt_step, opt_step + log_every)
    rows = sorted(s for s in val_ces if s in window)
    if not rows:
        return None, (f"checkpoint {checkpoint_step} is opt step {opt_step}, and no val row lies in "
                      f"[{window.start}, {window.stop}), where its probe's value would be logged")
    return val_ces[rows[0]], (f"checkpoint {checkpoint_step} is opt step {opt_step}; its probe's value "
                              f"is logged on row {rows[0]}")


def reproduction_verdict(expected, measured, tolerance):
    """(exit code, verdict text) for a revival. 0: reproduced within the noise floor.
    1: drifted beyond it. 2: nothing honest to compare against, so it cannot be
    verified — never 0, or an un-checkable run would read as reproduced."""
    if expected is None:
        return 2, "NO recorded metric to compare against — cannot verify (measured value logged above)."
    delta = abs(measured - expected)
    if delta <= tolerance:
        return 0, f"|measured - recorded| {delta:.4f} ≤ {tolerance}: ✅ REPRODUCED (within noise floor)"
    return 1, f"|measured - recorded| {delta:.4f} > {tolerance}: ❌ DRIFTED — reconstruction is not faithful"


def evaluate(run_id, arch_override=None, build_venv=True, limit=None,
             tolerance=0.06, cpu=False, dry_run=False):
    """#44's definition of done: from the snapshot alone, rebuild the world, run
    that run's own yardstick (instruments.yardstick.eval_yardstick at its commit) on its weights,
    and confirm the held-out val CE reproduces the recorded value within the noise
    floor. A faithful time machine closes this loop; a broken one won't."""
    world = reconstruct(run_id, for_mode="eval", build_venv=build_venv,
                        arch_override=arch_override)
    wt, py, ckpt, arch = (world["worktree"], world["python"],
                          world["checkpoints"], world["arch"])

    # The worktree is checked out at that run's commit, which may predate the #143
    # package move — so the yardstick sits at one of two paths depending on how old
    # the revived world is. Probe both; a time machine that only knows today's
    # layout can't reach the runs worth reviving.
    yard_args = None
    for rel, invocation in (
        ("instruments/yardstick/eval_yardstick.py", ["-m", "instruments.yardstick.eval_yardstick"]),
        ("instruments.yardstick.eval_yardstick", ["instruments.yardstick.eval_yardstick"]),
    ):
        if os.path.exists(os.path.join(wt, rel)):
            yard_args = invocation
            break
    if yard_args is None:
        raise SystemExit(
            f"no eval_yardstick at {run_id}'s commit (looked for the post-#143 package "
            f"path and the historical tools/ path) — that run predates the yardstick "
            f"(#48); the metric loop can't be closed for it.")

    val_ces = recorded_val_ces(run_id)  # read once; matched below to the step actually scored
    params = runlog.recorded_params(runlog.read_metadata(run_dir(run_id)))
    accumulation, val_every = params.get("ACCUMULATION_STEPS"), params.get("VAL_EVERY_OPT_STEPS")
    out_json = os.path.join(TM_ROOT, "eval", f"{run_id}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    cmd = [py, *yard_args, "--arch", arch,
           "--checkpoint-path", ckpt, "--json-out", out_json]
    if limit:
        cmd += ["--limit", str(limit)]

    # The revived worktree's code, not this checkout's. CPU XLA can't lower the f16
    # matmuls, hence FORCE_F32_COMPUTE; note the fidelity caveat below.
    env = module_env(wt, MODEL_ARCH=arch, **({"FORCE_F32_COMPUTE": "1"} if cpu else {}))

    print(f"\nClosing the metric loop for {run_id}:")
    print(f"  recorded val CE rows     : {len(val_ces)}"
          + (f" (opt steps {min(val_ces)}..{max(val_ces)}); matched to the scored checkpoint below"
             if val_ces else " — none in metrics.csv"))
    print(f"  tolerance (2σ, #17)      : ±{tolerance}")
    print(f"  eval command             : (cwd {os.path.relpath(wt, REPO_ROOT)}) "
          + " ".join(os.path.relpath(c, REPO_ROOT) if c.startswith(REPO_ROOT) else c for c in cmd))
    if dry_run:
        print("  --dry-run: not executing (wiring check only).")
        return
    if cpu:
        print("  ⚠️ --cpu uses the f32 path — numbers drift slightly from the f16 "
              "training path; the authoritative check runs on the GPU (f16).")

    subprocess.run(cmd, cwd=wt, env=env, check=True)
    with open(out_json) as f:
        row = json.load(f)
    measured = (row.get("heldout") or {}).get("val_ce")
    if measured is None:
        raise SystemExit("Yardstick produced no held-out val CE (DATA_ROOT unset?) "
                         "— cannot compare. See the JSON at " + out_json)

    # Compare against the CE logged for the checkpoint the yardstick scored.
    step = (row.get("checkpoint") or {}).get("step")
    if step is None:
        expect, chosen = None, "the yardstick reported no checkpoint step"
    else:
        expect, chosen = val_ce_for_checkpoint(val_ces, step, accumulation, val_every)

    print(f"\n  measured held-out val CE : {measured:.4f}")
    print(f"  recorded held-out val CE : {'—' if expect is None else f'{expect:.4f}'}  ({chosen})")
    # Only a value tied to this checkpoint's own probe reaches the comparison, so
    # REPRODUCED (exit 0) is never read off another probe's number.
    code, verdict = reproduction_verdict(expect, measured, tolerance)
    print(f"  VERDICT: {verdict}")
    return code


# --- forking a lineage -------------------------------------------------------

def fork(run_id, new_name, build_venv=True, arch_override=None):
    """Branch a new training lineage off an old checkpoint. The original run is
    left untouched: we copy its checkpoints into a fresh run dir and resume there,
    so continued training extends the fork, not the ancestor."""
    src_ckpt = os.path.join(run_dir(run_id), "checkpoints")
    if not os.path.isdir(src_ckpt):
        raise SystemExit(f"{run_id} has no checkpoints/ to fork from.")

    dst_dir = run_dir(new_name)
    dst_ckpt = os.path.join(dst_dir, "checkpoints")
    if os.path.exists(dst_dir):
        raise SystemExit(f"{dst_dir} already exists — pick another fork name.")

    world = reconstruct(run_id, for_mode="train", build_venv=build_venv,
                        arch_override=arch_override)

    os.makedirs(dst_dir, exist_ok=True)
    print(f"\nForking checkpoints {run_id} -> {new_name} (copying, original preserved)...")
    shutil.copytree(src_ckpt, dst_ckpt)
    # Carry the snapshot forward so the fork is itself revivable.
    for f in ("run_metadata.json", "env_freeze.txt", "system_snapshot.txt",
              "worktree.patch", "worktree.untracked.txt"):
        s = os.path.join(run_dir(run_id), f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, f))

    print("\nResume the fork inside its reconstructed world:")
    print(f"  cd {world['worktree']}")
    print(f"  PYTHONPATH=. MODEL_ARCH={world['arch']} DATA_ROOT=$DATA_ROOT \\")
    print(f"    {world['python']} -m trm.train.start --checkpoint-path {dst_ckpt}")
    print("  (add the run's GPU memory knobs before launching on the 2060.)")


# --- listing -----------------------------------------------------------------

def list_runs():
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit("No runs/ directory.")
    rows = []
    for name in sorted(os.listdir(RUNS_ROOT)):
        if not name.startswith("run_"):
            continue
        if not os.path.exists(os.path.join(RUNS_ROOT, name, "run_metadata.json")):
            continue
        try:
            meta = load_meta(name)
        except SystemExit:
            continue
        rows.append((
            name,
            (meta.get("git_commit") or "unknown")[:10],
            resolve_arch(name, meta) or "UNKNOWN",
            "patch" if has_file(name, "worktree.patch") else "commit-only",
            "freeze" if has_file(name, "env_freeze.txt") else "NO-freeze",
            "ckpt" if os.path.isdir(os.path.join(RUNS_ROOT, name, "checkpoints")) else "NO-ckpt",
        ))
    if not rows:
        print("No runs found under runs/.")
        return
    w = max(len(r[0]) for r in rows)
    print(f"{'run-id':<{w}}  commit      arch      fidelity     libs       weights")
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<10}  {r[2]:<8}  {r[3]:<11}  {r[4]:<9}  {r[5]}")


def main():
    # The shared arch list, imported here: it pulls in jax, which the module-level
    # helpers (and their tests) never need.
    from instruments.arch import ARCHES

    p = argparse.ArgumentParser(description="revive a stored weight in its training world")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="what's revivable and at what fidelity")

    r = sub.add_parser("reconstruct", help="rebuild a run's world; print the run command")
    r.add_argument("run_id")
    r.add_argument("--for", dest="for_mode", choices=["infer", "train"], default="infer")
    r.add_argument("--no-venv", action="store_true", help="skip the ~5GB venv build (dry)")
    r.add_argument("--arch", choices=ARCHES, default=None,
                   help="override arch for pre-capture runs that don't record it")

    f = sub.add_parser("fork", help="branch a new training lineage off an old checkpoint")
    f.add_argument("run_id")
    f.add_argument("new_name")
    f.add_argument("--no-venv", action="store_true")
    f.add_argument("--arch", choices=ARCHES, default=None,
                   help="override arch for pre-capture runs that don't record it")

    e = sub.add_parser("eval", help="close the loop: reproduce a run's metric within the noise floor (#44 DoD)")
    e.add_argument("run_id")
    e.add_argument("--arch", choices=ARCHES, default=None,
                   help="override arch for pre-capture runs that don't record it")
    e.add_argument("--no-venv", action="store_true")
    e.add_argument("--limit", type=int, default=None, help="LAMBADA sample cap (speed)")
    e.add_argument("--tolerance", type=float, default=0.06,
                   help="max |measured-recorded| held-out val CE; default 2σ from #17")
    e.add_argument("--cpu", action="store_true", help="f32 CPU path (slow, drifts from f16)")
    e.add_argument("--dry-run", action="store_true", help="print the wiring, don't execute")

    args = p.parse_args()
    if args.cmd == "list":
        list_runs()
    elif args.cmd == "reconstruct":
        reconstruct(args.run_id, args.for_mode, build_venv=not args.no_venv,
                    arch_override=args.arch)
    elif args.cmd == "fork":
        fork(args.run_id, args.new_name, build_venv=not args.no_venv,
             arch_override=args.arch)
    elif args.cmd == "eval":
        code = evaluate(args.run_id, arch_override=args.arch, build_venv=not args.no_venv,
                        limit=args.limit, tolerance=args.tolerance, cpu=args.cpu,
                        dry_run=args.dry_run)
        sys.exit(0 if code is None else code)  # None: --dry-run, nothing was judged


if __name__ == "__main__":
    main()
