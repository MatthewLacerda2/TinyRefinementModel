"""One arm of a real-scale recipe pair: train the shipped config briefly, report
how many tokens it needed to reach a target held-out CE (#26 stage 2).

    python -m experiments.recipe.tokens_to_ce --optimizer muon --lr-mult 50 --seed 0 \\
        --target-ce 5.5 --opt-steps 512

Launches the real trainer (`trm.train.start`) as a subprocess with the arm's
environment — optimizer, Muon LR multiplier, matched seeds, a token budget that
makes the LR schedule complete inside the run, warmup shortened to fit, and the
validation probe every 16 opt steps for resolution — stops it at the step cap,
and reads metrics.csv. Prints one RESULT line per run for the referee.

Why tokens-to-CE and not CE-at-fixed-tokens: a 1.3x token-efficiency claim is a
~30% difference on the tokens axis, far outside seed noise; the same effect read
as CE at fixed steps lands in the 0.001-0.005 nats band CLAUDE.md warns about.
The final CE is still recorded, as a readout.

The metric saturates at the cap: an arm that never reaches the target reports
the cap, so "both arms at the cap" reads as a tie the referee will call
INCONCLUSIVE, never as a win.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import signal
import subprocess
import sys
import time

from instruments import results

REPO = pathlib.Path(__file__).resolve().parents[2]

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "tokens_to_target_M": ("measured", "opt step of the first validation row at or below --target-ce, "
                                       "x TOKENS_PER_OPT_STEP; the cap when never reached"),
    "final_val_ce": ("sampled", "the validation probe's last reading before the cap (EVAL_ROWS held-out rows)"),
}


def tokens_to_target(metrics_csv: pathlib.Path, target_ce: float, cap_steps: int, tokens_per_opt_step: int):
    """(tokens in millions to first reach target, final val CE, reached, probe_aligned).

    Read at the probe's own step (`val_step`, #351) when the run recorded it; older
    runs only have the logged row's step, which is late by up to LOG_REAL_STEPS - 1
    opt steps. `probe_aligned` says which, so a pair that mixes the two readings
    (a reused old control against a new treatment) is visible in its RESULT lines."""
    final, hit, aligned = None, None, False
    with metrics_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                val = row.get("val_ce") or ""
                probe = row.get("val_step") or ""
                aligned = aligned or bool(probe)
                step = int(probe) if probe else int(row["step"])
            except (KeyError, ValueError):
                continue
            if step > cap_steps or not val:
                continue
            final = float(val)
            if hit is None and final <= target_ce:
                hit = step
    steps = hit if hit is not None else cap_steps
    return steps * tokens_per_opt_step / 1e6, final, hit is not None, aligned


def last_step(metrics_csv: pathlib.Path) -> int:
    try:
        with metrics_csv.open() as f:
            rows = [int(r["step"]) for r in csv.DictReader(f) if r.get("step", "").isdigit()]
        return max(rows) if rows else 0
    except OSError:
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--optimizer", required=True, choices=["adamw", "muon"])
    ap.add_argument("--lr-mult", type=float, default=None, help="MUON_LR_MULT for the muon arm")
    ap.add_argument("--seed", type=int, required=True, help="MODEL_SEED and DATA_SEED, both — matched pairs share it")
    ap.add_argument("--target-ce", type=float, required=True)
    ap.add_argument("--opt-steps", type=int, default=512, help="cap; also the LR schedule's horizon")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--peak-lr", type=float, default=None,
                    help="PEAK_LR: the value the cosine warms up to (#287). The whole schedule "
                         "scales with it (init = peak/10, end = peak/100), so a sweep moves one "
                         "variable — the LR scale — not three. Unset leaves the shipped 1e-4.")
    ap.add_argument("--val-every", type=int, default=16,
                    help="VAL_EVERY_OPT_STEPS. The metric cannot resolve finer than this: every "
                         "seed inside one probe interval reports the same token count, which is "
                         "how #26 stage 2 came out with sigma_pooled exactly 0.")
    ap.add_argument("--tag", default="026", help="run dirs are runs/run_<tag>_<optimizer>[_m<mult>]_s<seed>")
    args = ap.parse_args(argv)

    from trm.config import TOKENS_PER_OPT_STEP
    name = (f"run_{args.tag}_{args.optimizer}"
            + (f"_m{args.lr_mult:g}" if args.lr_mult is not None else "")
            + (f"_lr{args.peak_lr:g}" if args.peak_lr is not None else "")
            + f"_s{args.seed}")
    run_dir = REPO / "runs" / name
    metrics = run_dir / "metrics.csv"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "TRM_OPTIMIZER": args.optimizer,
        "MODEL_SEED": str(args.seed), "DATA_SEED": str(args.seed),
        "TRAIN_TOKEN_BUDGET": str(args.opt_steps * TOKENS_PER_OPT_STEP),
        "WARMUP_STEPS": str(args.warmup),
        "VAL_EVERY_OPT_STEPS": str(args.val_every),
        "CHECKPOINT_EVERY_OPT_STEPS": "256",
        "MILESTONE_EVERY_TOKENS": "0",
    }
    if args.lr_mult is not None:
        env["MUON_LR_MULT"] = str(args.lr_mult)
    if args.peak_lr is not None:
        env["PEAK_LR"] = repr(args.peak_lr)

    if last_step(metrics) < args.opt_steps:
        run_dir.mkdir(parents=True, exist_ok=True)
        log = (run_dir / "train.log").open("a")
        proc = subprocess.Popen([sys.executable, "-m", "trm.train.start", "--checkpoint-path",
                                 str(run_dir / "checkpoints")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: trainer pid {proc.pid}, to opt step {args.opt_steps}", flush=True)
        try:
            while proc.poll() is None and last_step(metrics) < args.opt_steps:
                time.sleep(30)
        finally:
            if proc.poll() is None:
                print(f"{name}: stopping trainer pid {proc.pid} at opt step {last_step(metrics)} "
                      f"(cap {args.opt_steps}; harness exiting)", flush=True)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if last_step(metrics) < args.opt_steps:
            raise SystemExit(f"{name}: trainer ended at step {last_step(metrics)} < {args.opt_steps} "
                             f"(exit {proc.returncode}); see {run_dir / 'train.log'}")
    else:
        print(f"{name}: already at the cap, reading the recorded run", flush=True)

    tokens_m, final, reached, aligned = tokens_to_target(metrics, args.target_ce, args.opt_steps,
                                                         TOKENS_PER_OPT_STEP)
    print(f"{name}: target {args.target_ce} {'reached' if reached else 'NOT reached (cap)'} at "
          f"{tokens_m:.1f}M tokens; final val CE {final}", flush=True)
    results.emit("run", tokens_to_target_M=tokens_m, final_val_ce=final if final is not None else float("nan"),
                 reached=float(reached), probe_aligned=float(aligned))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
