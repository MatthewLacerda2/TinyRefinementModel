"""Launch a supervised base run from one number: its token budget (#169).

    python -m trm.runtime.launch --budget 4e9 [--issue 157] [--dry-run]
    make launch BUDGET=4e9

Launching #157 took four pieces of knowledge assembled by hand, each a way to lose
days:
  * `--stop-step`, derived from the budget by hand — and the checkpoint it stops
    on must cover the budget, or the last stretch is trained and thrown away
    (the 4B run's final checkpoint landed 56 steps short);
  * `--checkpoint-path`, and specifically NOT `--new-run`: the supervisor replays
    the trainer's arguments on every crash relaunch, and `--new-run` makes day six's
    relaunch start a brand-new run from scratch;
  * `TRAIN_TOKEN_BUDGET` in the environment, matching that stop step;
  * a new run directory whose name the checkpoint path must carry.

The repo knows the incantation now, not the session. It refuses rather than
guesses: no budget, no launch; a card someone holds, no launch; an existing run
directory, no launch.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

from trm.runtime.layout import CHECKPOINT_EVERY_OPT_STEPS
from trm.runtime.run_budget import BUDGET_ENV
from trm.runtime.supervisor import RUNS_DIR, GpuLock, _pid_alive


def stop_step_for(budget_tokens: int, tokens_per_opt_step: int,
                  checkpoint_every: int = CHECKPOINT_EVERY_OPT_STEPS) -> int:
    """The first checkpoint boundary at or past the budget.

    The supervisor stops the run once metrics.csv reaches this step, and the
    checkpoint at exactly this step is written before that row is logged — so the
    kept weights cover every budgeted token."""
    if budget_tokens <= 0:
        raise SystemExit(f"budget must be positive, got {budget_tokens}")
    return math.ceil(math.ceil(budget_tokens / tokens_per_opt_step) / checkpoint_every) * checkpoint_every


@dataclass
class Plan:
    run_id: str
    run_dir: pathlib.Path
    stop_step: int
    argv: list[str]
    env: dict[str, str]
    stdout_path: pathlib.Path


def plan(budget_tokens: int, tokens_per_opt_step: int, *, run_id: str, issue: int | None = None,
         spec: pathlib.Path | None = None,
         runs_dir: pathlib.Path = RUNS_DIR, python: str = sys.executable) -> Plan:
    run_dir = runs_dir / run_id
    stop = stop_step_for(budget_tokens, tokens_per_opt_step)
    argv = [python, "-m", "trm.runtime.supervisor",
            "--stop-step", str(stop),
            "--run-dir", str(run_dir),
            "--log", str(runs_dir / f"{run_id}.log"),
            *(["--issue", str(issue)] if issue is not None else []),
            *(["--spec", str(spec)] if spec is not None else []),
            "--", "--checkpoint-path", str(run_dir / "checkpoints")]
    assert "--new-run" not in argv, "a relaunch would replay it and restart the run from scratch"
    return Plan(run_id, run_dir, stop, argv, {BUDGET_ENV: str(budget_tokens)},
                runs_dir / f"{run_id}.supervisor.out")


def spec_refusal(path, repo: pathlib.Path = RUNS_DIR.parent) -> str | None:
    """Why a launch must refuse this spec, or None: it must be committed and clean.
    A pre-registration that is not in git registers nothing."""
    rel = os.path.relpath(pathlib.Path(path).resolve(), repo)
    if subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, capture_output=True).returncode != 0:
        return f"{rel} is not committed — a pre-registration that is not in git registers nothing"
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo).returncode != 0:
        return f"{rel} has uncommitted changes — commit the spec before launching against it"
    return None


def spec_budget_tokens(path) -> int:
    import tomllib
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    try:
        return int(spec["protocol"]["budget_tokens"])
    except KeyError:
        raise SystemExit(f"{path}: a base-run spec declares [protocol] budget_tokens (#294)") from None


def refusal(p: Plan) -> str | None:
    """Why this launch must not happen, or None."""
    if p.run_dir.exists():
        return f"{p.run_dir} already exists — a launch never reuses a run directory"
    holder = GpuLock().holder()
    if holder and _pid_alive(holder[0]):
        return f"the GPU is held by pid {holder[0]} ({holder[1] or 'unlabelled'})"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=None, help="token budget, e.g. 4e9 (required)")
    ap.add_argument("--issue", type=int, default=None, help="pinned issue the supervisor heartbeats into")
    ap.add_argument("--spec", type=pathlib.Path, default=None,
                    help="the pre-registered base-run spec (required): must be committed and clean, "
                         "and its budget_tokens must equal --budget")
    ap.add_argument("--dry-run", action="store_true", help="print the launch, do nothing")
    args = ap.parse_args(argv)
    if args.budget is None:
        raise SystemExit("no BUDGET, no launch: pass --budget (make launch BUDGET=4e9)")
    if args.spec is None:
        raise SystemExit("no SPEC, no launch: a base run is pre-registered like everything else (#294) — "
                         "make launch SPEC=experiments/base/specs/<id>.toml BUDGET=…")
    why = spec_refusal(args.spec)
    if why:
        raise SystemExit(f"refusing to launch: {why}")
    spec_budget = spec_budget_tokens(args.spec)
    if spec_budget != int(args.budget):
        raise SystemExit(f"refusing to launch: --budget {int(args.budget):,} disagrees with the spec's "
                         f"budget_tokens {spec_budget:,}; change one, commit, relaunch")

    from trm.config import TOKENS_PER_OPT_STEP
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    p = plan(int(args.budget), TOKENS_PER_OPT_STEP, run_id=run_id, issue=args.issue, spec=args.spec.resolve())
    print(f"run:        {p.run_dir}")
    print(f"budget:     {int(args.budget):,} tokens -> stop at opt step {p.stop_step:,} "
          f"({p.stop_step * TOKENS_PER_OPT_STEP:,} tokens, a checkpoint boundary)")
    print(f"command:    {' '.join(f'{k}={v}' for k, v in p.env.items())} {' '.join(p.argv)}")
    print(f"supervisor: stdout -> {p.stdout_path}; heartbeats -> runs/{run_id}.supervisor.log")
    if args.dry_run:
        return 0
    why = refusal(p)
    if why:
        raise SystemExit(f"refusing to launch: {why}")

    p.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with p.stdout_path.open("a") as out:
        proc = subprocess.Popen(p.argv, env={**os.environ, **p.env}, stdout=out, stderr=subprocess.STDOUT,
                                cwd=RUNS_DIR.parent, start_new_session=True)
    print(f"launched:   supervisor pid {proc.pid}, detached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
