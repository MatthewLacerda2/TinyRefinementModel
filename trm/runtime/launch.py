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

**Surviving a power cut (#384).** The supervisor relaunches a trainer that crashes,
but a power cut takes the supervisor with it. So a launch records itself in
`runs/.active_base_run.json`, and `python -m trm.runtime.launch --resume` (`make
resume`) brings that supervisor back: same command, same run directory, and the
trainer resumes from its newest checkpoint as it does after any crash. It does
nothing when there is no active run, when the run already ended on purpose
(completed, or killed for a reason a relaunch would repeat), when it reached its stop
step, or when something holds the card. On this machine a systemd user unit runs it
at boot (user lingering is on, so it runs without a login):

    # ~/.config/systemd/user/trm-base-run-resume.service
    [Unit]
    Description=Resume the TinyRefinementModel base run after a reboot (#384)
    After=default.target
    [Service]
    Type=oneshot
    WorkingDirectory=/home/lendacerda/Desktop/Repos/TinyRefinementModel
    ExecStartPre=/bin/sleep 90
    ExecStart=/home/lendacerda/Desktop/Repos/TinyRefinementModel/venv/bin/python -m trm.runtime.launch --resume
    [Install]
    WantedBy=default.target

The data position after a resume is an estimate (fine for a base run, not for a
matched pair), and what a cut costs is the steps since the last checkpoint,
CHECKPOINT_EVERY_OPT_STEPS at most.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

from trm.runtime.layout import CHECKPOINT_EVERY_OPT_STEPS
from trm.runtime.run_budget import BUDGET_ENV
from trm.runtime.gpu_lock import GpuLock, _pid_alive
from trm.runtime.supervisor import DELIBERATE, GAVE_UP, RUNS_DIR, read_progress

# The base run a launch started and has not seen end, for --resume after a reboot.
ACTIVE_RUN = RUNS_DIR / ".active_base_run.json"
# Outcomes a relaunch would only repeat: a finished run, or one stopped on purpose.
TERMINAL = (*DELIBERATE, GAVE_UP)
_OUTCOME_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ([A-Z_]+):")


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


def last_outcome(heartbeat_text: str) -> str | None:
    """The last supervisor outcome in its heartbeat log (`<stamp> OUTCOME: reason`),
    skipping margin alarms and relaunch notes; None when it logged none."""
    outcome = None
    for line in heartbeat_text.splitlines():
        match = _OUTCOME_LINE.match(line)
        if match:
            outcome = match.group(1)
    return outcome


def resume_refusal(state: dict, outcome: str | None, step: int, card_holder):
    """(why the recorded run must not be resumed, whether it is over for good), or
    (None, False) to resume it. Pure. A held card is a reason to wait, not an end."""
    if outcome in TERMINAL:
        return f"{state['run_id']} already ended ({outcome}) — a relaunch would repeat it", True
    if step >= state["stop_step"]:
        return f"{state['run_id']} reached its stop step {state['stop_step']:,}", True
    if card_holder is not None:
        return f"the card is held by pid {card_holder[0]} ({card_holder[1] or 'unlabelled'})", False
    return None, False


def resume(active: pathlib.Path = ACTIVE_RUN) -> int:
    """Bring back the supervisor of the base run a power cut stopped (#384)."""
    if not active.exists():
        print("resume: no active base run")
        return 0
    state = json.loads(active.read_text())
    run_dir = pathlib.Path(state["run_dir"])
    heartbeat = run_dir.parent / f"{state['run_id']}.supervisor.log"
    text = heartbeat.read_text(errors="replace") if heartbeat.exists() else ""
    step, _ = read_progress(run_dir / "metrics.csv")
    holder = GpuLock().holder()
    why, over = resume_refusal(state, last_outcome(text), step,
                               holder if holder and _pid_alive(holder[0]) else None)
    if why:
        print(f"resume: not resuming — {why}")
        if over:
            active.unlink()  # finished: a later boot has nothing to bring back
        return 0
    stamp = datetime.datetime.now().strftime("%F %T")
    with heartbeat.open("a") as fh:
        fh.write(f"{stamp} resumed after a reboot at opt step {step:,} (#384)\n")
    with pathlib.Path(state["stdout_path"]).open("a") as out:
        proc = subprocess.Popen(state["argv"], env={**os.environ, **state["env"]}, stdout=out,
                                stderr=subprocess.STDOUT, cwd=RUNS_DIR.parent, start_new_session=True)
    print(f"resume: {state['run_id']} from opt step {step:,}, supervisor pid {proc.pid}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=None, help="token budget, e.g. 4e9 (required)")
    ap.add_argument("--issue", type=int, default=None, help="pinned issue the supervisor heartbeats into")
    ap.add_argument("--spec", type=pathlib.Path, default=None,
                    help="the pre-registered base-run spec (required): must be committed and clean, "
                         "and its budget_tokens must equal --budget")
    ap.add_argument("--dry-run", action="store_true", help="print the launch, do nothing")
    ap.add_argument("--resume", action="store_true",
                    help="bring back the recorded base run's supervisor after a reboot (#384)")
    args = ap.parse_args(argv)
    if args.resume:
        return resume()
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
    ACTIVE_RUN.write_text(json.dumps({
        "run_id": p.run_id, "run_dir": str(p.run_dir), "stop_step": p.stop_step,
        "argv": p.argv, "env": p.env, "stdout_path": str(p.stdout_path),
        "launched_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
