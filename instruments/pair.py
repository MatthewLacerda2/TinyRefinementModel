"""Start, pause, resume and watch a pre-registered sweep on the card (#565).

    make pair SPEC=experiments/recipe/specs/<id>.toml    # python -m instruments.pair start <spec>
    make pair-pause                                     # python -m instruments.pair pause
    make pair-resume                                    # python -m instruments.pair resume
    make pair-status                                    # python -m instruments.pair status

`start` runs `instruments.experiment <spec>` detached, as the user unit `trm-pair`,
at oom_score_adj 100 (#524). Its output is appended to `runs/<spec id>.log`. It also
leaves the "▶ started" comment on the spec's issue, because the card is a serial
queue and it must be visible what holds it. Which spec is on the card is written to
`runs/pair.current`, so `resume` needs no argument.

`pause` stops the unit. The stop goes to the runner alone, which stops its arm
(#519), which stops its trainer. Each layer unwinds in order, and the next start
rewinds the arm to its last checkpoint (#557). `resume` starts the same spec with
`--no-gate`. The journal skips every arm already recorded.

One unit, because there is one card.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib

from instruments._common import REPO_ROOT
from instruments.experiment import journal_path, load_execution, read_journal
from instruments.verdict import load_spec
from trm.runtime.oom import systemd_run

REPORTS = {
    "opt step": ("measured", "the last row of the running arm's metrics.csv"),
    "disk free": ("measured", "shutil.disk_usage on runs/"),
}

UNIT = "trm-pair.service"
RUNS = REPO_ROOT / "runs"
CURRENT = RUNS / "pair.current"
# The runner stops its arm, and the harness gives its trainer 120 s to exit (#519),
# so a stop needs longer than systemd's default 90 s before it escalates to SIGKILL.
STOP_TIMEOUT_S = 600
# What a tokens_to_ce arm prints when it launches its trainer.
TRAINER_LINE = re.compile(r"^(run_\S+): trainer pid \d+, to opt step (\d+)")
GIB = 1024 ** 3


def unit_state() -> str:
    out = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def log_path(spec_id: str):
    return RUNS / f"{spec_id}.log"


def launch_command(spec_path, spec_id: str, *, gate: bool) -> list[str]:
    argv = [sys.executable, "-m", "instruments.experiment", str(spec_path), *([] if gate else ["--no-gate"])]
    return systemd_run(argv, dict(os.environ), REPO_ROOT, UNIT, stdout=log_path(spec_id),
                       properties=(f"TimeoutStopSec={STOP_TIMEOUT_S}",))


def launch(spec_path, *, gate: bool) -> str:
    """Start the sweep as UNIT; the spec's id. Refuses while the unit is running."""
    if (state := unit_state()) in ("active", "activating", "deactivating"):
        raise SystemExit(f"{UNIT} is {state}: one sweep at a time on the card (make pair-status)")
    spec = load_spec(spec_path)
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT], capture_output=True)
    RUNS.mkdir(exist_ok=True)
    subprocess.run(launch_command(spec_path, spec.id, gate=gate), check=True)
    CURRENT.write_text(json.dumps({"spec": str(spec_path)}))
    print(f"▶ {spec.id} on {UNIT}; log {log_path(spec.id)}")
    return spec.id


def announce(spec_path, spec_id: str) -> None:
    """The "▶ started" comment on the spec's issue, when the spec names one."""
    with open(spec_path, "rb") as f:
        issue = tomllib.load(f)["experiment"].get("issue")
    if issue is None or shutil.which("gh") is None:
        return
    body = f"▶ started: `{spec_id}` on the card (`make pair`), log `runs/{spec_id}.log`"
    subprocess.run(["gh", "issue", "comment", str(issue), "--body", body], cwd=REPO_ROOT, check=False)


def current_spec():
    if not CURRENT.exists():
        raise SystemExit(f"no sweep recorded in {CURRENT}: start one with make pair SPEC=...")
    return json.loads(CURRENT.read_text())["spec"]


def pause() -> None:
    print(f"stopping {UNIT} (the runner stops its arm, the arm its trainer)…", flush=True)
    subprocess.run(["systemctl", "--user", "stop", UNIT], check=False)
    print(f"{UNIT}: {unit_state()}. make pair-resume continues from each arm's last checkpoint")


def running_arm(log_text: str):
    """(run name, cap) of the last trainer the log shows launched, or None."""
    found = None
    for line in log_text.splitlines():
        if match := TRAINER_LINE.match(line):
            found = match.group(1), int(match.group(2))
    return found


def last_step(metrics) -> int | None:
    if not metrics.exists():
        return None
    lines = metrics.read_text().splitlines()
    return int(lines[-1].split(",", 1)[0]) if len(lines) > 1 else None


def status() -> str:
    spec_path = current_spec()
    spec = load_spec(spec_path)
    planned = len(list(load_execution(spec).runs()))
    done = len({(r["arm"], r["seed"]) for r in read_journal(journal_path(spec.id))})
    lines = [f"{spec.id}: {UNIT} {unit_state()} | {done}/{planned} arm-seeds recorded"]
    log = log_path(spec.id)
    if log.exists() and (arm := running_arm(log.read_text(errors="replace"))):
        name, cap = arm
        lines.append(f"current arm {name}: opt step {last_step(RUNS / name / 'metrics.csv')} of {cap}")
    lines.append(f"disk free on runs/: {shutil.disk_usage(RUNS).free / GIB:.1f} GiB")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start", help="start a spec's sweep on the card")
    start.add_argument("spec")
    start.add_argument("--no-gate", action="store_true", help="skip the tests/core gate")
    sub.add_parser("pause", help="stop the sweep; resume continues it")
    sub.add_parser("resume", help="restart the paused spec without the gate")
    sub.add_parser("status", help="unit state, arms recorded, the running arm's step, disk free")
    args = ap.parse_args(argv)

    if args.action == "start":
        announce(args.spec, launch(args.spec, gate=not args.no_gate))
    elif args.action == "pause":
        pause()
    elif args.action == "resume":
        launch(current_spec(), gate=False)
    else:
        print(status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
