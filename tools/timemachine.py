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
    python tools/timemachine.py list
    python tools/timemachine.py reconstruct <run-id> [--for infer|train] [--no-venv]
    python tools/timemachine.py fork <run-id> <new-name> [--no-venv]
"""

import os
import sys
import csv
import json
import shutil
import hashlib
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM_ROOT = os.path.join(REPO_ROOT, ".timemachine")
RUNS_ROOT = os.path.join(REPO_ROOT, "runs")


# --- reading a run's snapshot ------------------------------------------------

def run_dir(run_id):
    return os.path.join(RUNS_ROOT, run_id)


def load_meta(run_id):
    path = os.path.join(run_dir(run_id), "run_metadata.json")
    if not os.path.exists(path):
        raise SystemExit(f"No run_metadata.json for {run_id} (looked in {path}).")
    with open(path) as f:
        return json.load(f)


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
        n = len([l for l in open(untracked) if l.strip()])
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
            f"two arches are not checkpoint-compatible. Re-run with --arch "
            f"refiner|reasoner to say which skeleton to rebuild.")

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
        print(f"  PYTHONPATH=. MODEL_ARCH={arch} DATA_ROOT=$DATA_ROOT \\")
        print(f"    {py} tools/depth_playground.py --arch {arch} \\")
        print(f"    --checkpoint-path {ckpt} --prompt 'The capital of France is'")
        print("  (older worktrees may not have depth_playground's --arch flag; "
              "fall back to that commit's infer_local.py)")
    else:
        print(f"  # fork a NEW lineage (original {run_id} preserved):")
        print(f"  python tools/timemachine.py fork {run_id} <new-name>")
        print("  # ...on the 6GB card, also export the run's memory knobs "
              "(XLA_PYTHON_CLIENT_MEM_FRACTION, CHUNKED_ATTENTION, etc.) from its "
              "launch env / system_snapshot.txt, or it will OOM.")


# --- closing the loop: reproduce the metric (#44 DoD) ------------------------

def recorded_val_ce(run_id, step=None):
    """The run's own held-out val CE (metrics.csv) — the number a faithful revival
    must reproduce. Held-out CE is what the #17 noise floor is stated in (2σ = 0.06
    nats), so it is the natural apples-to-apples target.

    The yardstick scores whatever checkpoint is on disk, whose step need not be the
    last *logged* row (val cadence and checkpoint cadence can differ). Pass `step`
    (from the yardstick's reported checkpoint step) to compare like-for-like; we fall
    back to the last logged CE only when no row matches."""
    path = os.path.join(run_dir(run_id), "metrics.csv")
    if not os.path.exists(path):
        return None
    last = matched = None
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get("val_ce", "")
            if v in ("", "nan", None):
                continue
            try:
                last = float(v)
            except ValueError:
                continue
            if step is not None:
                try:
                    if int(float(row.get("step", "nan"))) == int(step):
                        matched = last
                except ValueError:
                    pass
    return matched if matched is not None else last


def evaluate(run_id, arch_override=None, build_venv=True, limit=None,
             tolerance=0.06, cpu=False, dry_run=False):
    """#44's definition of done: from the snapshot alone, rebuild the world, run
    that run's own yardstick (tools/eval_yardstick.py at its commit) on its weights,
    and confirm the held-out val CE reproduces the recorded value within the noise
    floor. A faithful time machine closes this loop; a broken one won't."""
    world = reconstruct(run_id, for_mode="eval", build_venv=build_venv,
                        arch_override=arch_override)
    wt, py, ckpt, arch = (world["worktree"], world["python"],
                          world["checkpoints"], world["arch"])

    yard = os.path.join(wt, "tools", "eval_yardstick.py")
    if not os.path.exists(yard):
        raise SystemExit(
            f"tools/eval_yardstick.py does not exist at {run_id}'s commit — that "
            f"run predates the yardstick (#48); the metric loop can't be closed for it.")

    expect = recorded_val_ce(run_id)
    out_json = os.path.join(TM_ROOT, "eval", f"{run_id}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    cmd = [py, "tools/eval_yardstick.py", "--arch", arch,
           "--checkpoint-path", ckpt, "--json-out", out_json]
    if limit:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()
    env["PYTHONPATH"] = wt
    env["MODEL_ARCH"] = arch
    if cpu:  # CPU XLA can't lower the f16 matmuls; note the fidelity caveat below.
        env["FORCE_F32_COMPUTE"] = "1"

    print(f"\nClosing the metric loop for {run_id}:")
    print(f"  recorded held-out val CE : {expect if expect is not None else '(none in metrics.csv)'}")
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

    # Compare against the CE logged for the exact checkpoint step the yardstick scored.
    step = (row.get("checkpoint") or {}).get("step")
    expect = recorded_val_ce(run_id, step=step) if step is not None else expect

    print(f"\n  measured held-out val CE : {measured:.4f}")
    if expect is None:
        # A DoD gate that can't find the recorded metric must not exit success —
        # otherwise an un-checkable run reads as "reproduced". Distinct code 2.
        print("  verdict: NO recorded metric to compare against — cannot verify "
              "(measured value logged above).")
        raise SystemExit(2)
    delta = abs(measured - expect)
    ok = delta <= tolerance
    print(f"  |measured - recorded|    : {delta:.4f}  "
          f"({'≤' if ok else '>'} {tolerance})")
    print(f"  VERDICT: {'✅ REPRODUCED (within noise floor)' if ok else '❌ DRIFTED — reconstruction is not faithful'}")
    return ok


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

    print(f"\nResume the fork inside its reconstructed world:")
    print(f"  cd {world['worktree']}")
    print(f"  PYTHONPATH=. MODEL_ARCH={world['arch']} DATA_ROOT=$DATA_ROOT \\")
    print(f"    {world['python']} start_training.py --checkpoint-path {dst_ckpt}")
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
    p = argparse.ArgumentParser(description="revive a stored weight in its training world")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="what's revivable and at what fidelity")

    r = sub.add_parser("reconstruct", help="rebuild a run's world; print the run command")
    r.add_argument("run_id")
    r.add_argument("--for", dest="for_mode", choices=["infer", "train"], default="infer")
    r.add_argument("--no-venv", action="store_true", help="skip the ~5GB venv build (dry)")
    r.add_argument("--arch", choices=["refiner", "reasoner"], default=None,
                   help="override arch for pre-capture runs that don't record it")

    f = sub.add_parser("fork", help="branch a new training lineage off an old checkpoint")
    f.add_argument("run_id")
    f.add_argument("new_name")
    f.add_argument("--no-venv", action="store_true")
    f.add_argument("--arch", choices=["refiner", "reasoner"], default=None,
                   help="override arch for pre-capture runs that don't record it")

    e = sub.add_parser("eval", help="close the loop: reproduce a run's metric within the noise floor (#44 DoD)")
    e.add_argument("run_id")
    e.add_argument("--arch", choices=["refiner", "reasoner"], default=None,
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
        ok = evaluate(args.run_id, arch_override=args.arch, build_venv=not args.no_venv,
                      limit=args.limit, tolerance=args.tolerance, cpu=args.cpu,
                      dry_run=args.dry_run)
        sys.exit(0 if ok or ok is None else 1)


if __name__ == "__main__":
    main()
