"""One arm of a mixture pair branched from a finished run (#489).

    python -m experiments.mix.branch_decay --arm general_web --seed 1 \\
        --mixture pretrain/fineweb-edu=0.35,pretrain/fineweb=0.25,pretrain/codeparrot=0.25,pretrain/finemath=0.15

Copies the branch checkpoint (a full-state step dir: weights, optimizer, data state)
into a fresh run dir. It resumes the real trainer there with DATA_BRANCH=1 on the arm's
mixture (only from that copy: a relaunch resumes the arm's own stream), and re-runs a WSD decay from --decay-start to --stop-step. Then it scores the
endpoint weights on the yardstick. Every arm starts from the same weights and optimizer
state, so the mixture is the only thing that differs; each seed reads its own rows (see
DATA_BRANCH in trm/config.py).

The readout is the endpoint, not tokens-to-target. The schedule completes in-run, and
the question is what the weights score once annealed on this mix, which is the question
the base run's gate asks (LAMBADA at completion).

After scoring, only the final step's checkpoint is kept, so six arms fit on the SSD.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

from instruments import results

REPO = pathlib.Path(__file__).resolve().parents[2]

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "lambada_acc": ("measured", "LAMBADA last-word accuracy, the full 5,153 examples, on the endpoint weights"),
    "fineweb_ce": ("sampled", "open-web FineWeb val CE, the speedrun shard's first --fineweb-tokens targets"),
    "heldout_ce": ("sampled", "our FineWeb-Edu held-out CE (EVAL_ROWS rows), the run's own val probe"),
    "code_ce": ("sampled", "codeparrot held-out CE from the run's last per-source probe (contaminated, #485)"),
    "math_ce": ("sampled", "finemath held-out CE from the run's last per-source probe"),
}

# The pre-decay checkpoint of run_20260920_191351 (opt step 30,528), mirrored to the HDD.
DEFAULT_BRANCH = "/mnt/d_drive/TRM_cold/run_20260920_191351/predecay_optstep30528"


def step_name(opt_step: int, accumulation: int) -> str:
    """The checkpoint dir name of an opt step: the trainer names it by micro-step - 1."""
    return str(opt_step * accumulation - 1)


def last_by_source(metrics_csv: pathlib.Path) -> dict[str, float]:
    """The last per-source held-out reading the run logged, {bucket: CE}."""
    last = {}
    with metrics_csv.open(errors="replace") as f:
        for row in csv.DictReader(f):
            if (row.get("val_by_source") or "").strip():
                last = dict(kv.split("=") for kv in row["val_by_source"].split(";") if "=" in kv)
    return {k: float(v) for k, v in last.items()}


def latest_step(ckpts: pathlib.Path) -> int | None:
    """The step dir the trainer would resume from, or None for a run with no checkpoint."""
    steps = [int(d.name) for d in ckpts.iterdir() if d.is_dir() and d.name.isdigit()] if ckpts.is_dir() else []
    return max(steps, default=None)


def resume_env(ckpts: pathlib.Path, copy: pathlib.Path, env: dict) -> dict:
    """`env` plus DATA_BRANCH=1 only when the trainer resumes from the branch copy itself.

    The trainer re-branches the data stream on every restore under DATA_BRANCH, so a
    relaunch from a later checkpoint would replay the rows the arm already trained on and
    unmatch the pair; that checkpoint already holds the arm's own stream (#489).
    """
    return {**env, "DATA_BRANCH": "1"} if latest_step(ckpts) == int(copy.name) else env


def train(run_dir: pathlib.Path, env: dict, final: pathlib.Path) -> None:
    """Resume the trainer in `run_dir` and stop it once the final step's checkpoint exists."""
    log = (run_dir / "train.log").open("a")
    proc = subprocess.Popen([sys.executable, "-m", "trm.train.start", "--checkpoint-path",
                             str(run_dir / "checkpoints")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"{run_dir.name}: trainer pid {proc.pid}, until {final.name}", flush=True)
    try:
        while proc.poll() is None and not final.is_dir():
            time.sleep(30)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)  # the trainer waits for pending checkpoint writes
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
    if not final.is_dir():
        raise SystemExit(f"{run_dir.name}: trainer ended (exit {proc.returncode}) before {final.name}; "
                         f"see {run_dir / 'train.log'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, help="run dirs are runs/run_<tag>_<arm>_s<seed>")
    ap.add_argument("--mixture", required=True, help="DATA_MIXTURE for this arm")
    ap.add_argument("--seed", type=int, required=True, help="DATA_SEED: which rows this seed reads")
    ap.add_argument("--branch-from", default=DEFAULT_BRANCH, help="a full-state checkpoint step dir")
    ap.add_argument("--branch-step", type=int, default=30528, help="the opt step --branch-from holds")
    ap.add_argument("--decay-start", type=int, default=30528, help="WSD_DECAY_START")
    ap.add_argument("--stop-step", type=int, default=32576, help="the decay's horizon and the arm's endpoint")
    ap.add_argument("--fineweb-tokens", type=int, default=262144)
    ap.add_argument("--tag", default="489")
    args = ap.parse_args(argv)

    from trm.settings import CONFIG
    run_dir = REPO / "runs" / f"run_{args.tag}_{args.arm}_s{args.seed}"
    ckpts = run_dir / "checkpoints"
    final = ckpts / step_name(args.stop_step, CONFIG.ACCUMULATION_STEPS)
    scored = run_dir / "yardstick_endpoint.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",  # train.log is read live; a ~10 h arm must not buffer it
        "DATA_MIXTURE": args.mixture,
        "DATA_SEED": str(args.seed), "MODEL_SEED": str(args.seed),
        "TRAIN_TOKEN_BUDGET": str(args.stop_step * CONFIG.TOKENS_PER_OPT_STEP),
        "LR_SCHEDULE": "wsd",
        "WSD_DECAY_START": str(args.decay_start),
        "MILESTONE_FIRST_TOKENS": "0",
    }

    if not scored.exists():
        if not final.is_dir():
            # Orbax finds a step by its dir name, so the copy is named by the step it holds.
            copy = ckpts / step_name(args.branch_step, CONFIG.ACCUMULATION_STEPS)
            if latest_step(ckpts) is None:
                ckpts.mkdir(parents=True, exist_ok=True)
                shutil.copytree(args.branch_from, copy)
            train(run_dir, resume_env(ckpts, copy, env), final)
        subprocess.run([sys.executable, "-m", "instruments.yardstick.eval_yardstick", "--arch", "plain",
                        "--checkpoint-path", str(ckpts), "--step", final.name,
                        "--fineweb-tokens", str(args.fineweb_tokens), "--json-out", str(scored)],
                       cwd=REPO, env=env, check=True)
        for older in ckpts.iterdir():
            if older.is_dir() and older.name.isdigit() and older != final:
                shutil.rmtree(older)

    row = json.loads(scored.read_text())
    by_source = last_by_source(run_dir / "metrics.csv")
    nan = float("nan")
    results.emit("run", lambada_acc=row["lambada"]["lambada_acc"], lambada_ppl=row["lambada"]["lambada_ppl"],
                 fineweb_ce=(row.get("fineweb_val") or {}).get("val_ce", nan),
                 heldout_ce=(row.get("heldout") or {}).get("val_ce", nan),
                 code_ce=by_source.get("codeparrot", nan), math_ce=by_source.get("finemath", nan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
