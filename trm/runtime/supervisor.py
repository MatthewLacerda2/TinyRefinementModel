"""Unattended training supervision — the #16 saga's aux scripts, promoted (#40 build 4).

Six launch attempts on the base run produced three bash scripts: a watchdog that
stopped the trainer at its token budget and killed it if the CE-plateau detector
tried to flip a pretrain run into SFT, a sentinel that relaunched it after a
crash, and a divergence guard added when the warm restart risked blowing up.
They worked, and one of them shipped a bug that is worth keeping in mind
forever.

**The bug, and why this is one process.** The watchdog and the sentinel were
separate processes that communicated by writing and grepping marker lines in a
log. So the sentinel's question — "the trainer is gone; did it crash, or did the
watchdog stop it?" — was answered by reading a file the watchdog had not
finished writing. A clean finish lands its marker ~60s after the process dies,
and in that window the sentinel read a completed run as a crash and relaunched
it. The production fix was to sleep 90 seconds and re-check, which narrows the
window without closing it.

Here there is no window. One process owns the child, so "did I stop it?" is a
fact it already has, and the answer comes from `decide()` — a pure function over
an observation — rather than from the filesystem. The ordering that removes the
race is one line of that function: **guards are evaluated before liveness**, so a
run that reached its budget is complete whether or not the process is still up.

Nothing here is clever. The point is that the decisions are now *stated*, in a
function that can be tested against the exact situation that fooled the original.

    python -m trm.runtime.supervisor --stop-step 2289 --issue 16 -- --new-run
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
GPU_LOCK = RUNS_DIR / ".gpu.lock"

# What the supervisor decided to do about the child.
CONTINUE, STOP, KILL, RELAUNCH, GIVE_UP = "CONTINUE", "STOP", "KILL", "RELAUNCH", "GIVE_UP"
RESTART = "RESTART"  # kill a wedged child, then relaunch it

# How the whole thing ended. RUNNING is the only non-terminal one.
RUNNING = "RUNNING"
BUDGET_COMPLETE = "BUDGET_COMPLETE"
WALLCLOCK_COMPLETE = "WALLCLOCK_COMPLETE"
KILLED_PLATEAU = "KILLED_PLATEAU"
KILLED_DIVERGENCE = "KILLED_DIVERGENCE"
CRASHED = "CRASHED"
STALLED = "STALLED"
GAVE_UP = "GAVE_UP"

# A terminal outcome the supervisor chose. Anything else that stops the child is
# a crash, and a crash is the only thing worth relaunching.
DELIBERATE = (BUDGET_COMPLETE, WALLCLOCK_COMPLETE, KILLED_PLATEAU, KILLED_DIVERGENCE)


@dataclass(frozen=True)
class Limits:
    """Every bound the supervisor enforces, in one place.

    `stop_step` exists because the trainer has no hard token stop — the budget
    is enforced from outside, at the opt step whose checkpoint covers it.
    """

    stop_step: int
    max_ce: float = 6.5
    divergence_checks: int = 2  # consecutive bad readings before killing
    # Polls with no step progress before the run counts as wedged. Generous by
    # default: startup compilation is minutes long and logs nothing, and the
    # motivating failure was three DAYS of silence, not three polls.
    stall_polls: int = 12
    max_retries: int = 2
    max_hours: float | None = None
    min_free_gb: float = 20.0


@dataclass(frozen=True)
class Observation:
    """One look at the run: what the metrics say, what the log says, is it alive."""

    step: int
    ce: float | None
    plateau_detected: bool
    alive: bool
    elapsed_hours: float = 0.0


@dataclass
class State:
    """What the supervisor remembers between looks."""

    divergence_streak: int = 0
    retries_used: int = 0
    last_step: int = -1
    stalled_polls: int = 0


@dataclass(frozen=True)
class Decision:
    action: str
    outcome: str
    reason: str


def _is_diverged(ce: float | None, max_ce: float) -> bool:
    """A missing CE is not divergence — it is a run that has not logged yet."""
    if ce is None:
        return False
    return math.isnan(ce) or math.isinf(ce) or ce > max_ce


def decide(obs: Observation, limits: Limits, state: State) -> Decision:
    """What to do about the run right now. Pure: no I/O, no signals, no clock.

    The order is the design. Guards come first and liveness comes last, so a run
    that hit its budget reads as complete even if the process is already gone —
    which is precisely the case the original sentinel got wrong when a clean stop
    landed inside its polling window. There is no timing assumption to tune here;
    a finished run is finished because the numbers say so.

    Plateau outranks everything: the CE-plateau detector flipping a pretrain run
    into SFT would silently contaminate the very thing the run exists to produce,
    so it is worth killing a run that is otherwise perfectly healthy.
    """
    if obs.step > state.last_step:
        state.last_step = obs.step
        state.stalled_polls = 0
    elif obs.alive:
        state.stalled_polls += 1

    if obs.plateau_detected:
        return Decision(KILL, KILLED_PLATEAU,
                        "CE-plateau SFT auto-flip detected; killing to protect the pretrain run")

    if _is_diverged(obs.ce, limits.max_ce):
        streak = state.divergence_streak + 1
        state.divergence_streak = streak
        if streak >= limits.divergence_checks:
            return Decision(KILL, KILLED_DIVERGENCE,
                            f"CE={obs.ce} outside the sane band for {streak} consecutive checks")
    else:
        state.divergence_streak = 0

    if obs.step >= limits.stop_step:
        return Decision(STOP, BUDGET_COMPLETE,
                        f"reached opt step {obs.step} >= budget stop {limits.stop_step}")

    if limits.max_hours is not None and obs.elapsed_hours >= limits.max_hours:
        return Decision(STOP, WALLCLOCK_COMPLETE,
                        f"ran {obs.elapsed_hours:.1f}h >= the {limits.max_hours}h cap")

    # A wedged run is the hazard that motivated all of this: a detached job once
    # stalled for three days unnoticed, because a process that is alive and doing
    # nothing looks exactly like a process that is working. It is alive, so it
    # has to be killed before it can be relaunched.
    if obs.alive and state.stalled_polls >= limits.stall_polls:
        if state.retries_used >= limits.max_retries:
            return Decision(GIVE_UP, GAVE_UP,
                            f"wedged at step {obs.step} and {limits.max_retries} relaunches were used")
        state.retries_used += 1
        state.stalled_polls = 0
        return Decision(RESTART, STALLED,
                        f"no progress past step {obs.step} for {limits.stall_polls} polls "
                        f"(restart {state.retries_used}/{limits.max_retries})")

    if not obs.alive:
        if state.retries_used >= limits.max_retries:
            return Decision(GIVE_UP, GAVE_UP,
                            f"died before budget and {limits.max_retries} relaunches were used")
        state.retries_used += 1
        return Decision(RELAUNCH, CRASHED,
                        f"died at step {obs.step} before budget "
                        f"(relaunch {state.retries_used}/{limits.max_retries})")

    return Decision(CONTINUE, RUNNING, f"step {obs.step}/{limits.stop_step}")


# --- preflight ----------------------------------------------------------------

class Preflight(Exception):
    """A reason not to start. Raised before anything expensive happens."""


def check_disk_headroom(path: pathlib.Path, min_free_gb: float) -> float:
    """Refuse to launch onto a nearly-full disk.

    The root filesystem on this box runs close to full, and a long run that
    fills it dies partway with a corrupt final checkpoint — the worst failure
    mode available, because it costs the compute AND the artifact.
    """
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        raise Preflight(
            f"only {free_gb:.1f}GB free at {path}, need {min_free_gb}GB — "
            f"archive to the HDD before launching (never delete runs/data/)")
    return free_gb


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class GpuLock:
    """The single RTX 2060 is a serial queue; this is the queue.

    A stale lock — the holder died without releasing — is taken over rather than
    respected. A lock nobody can clear is worse than no lock: it turns one crash
    into a card that stays idle until a human notices.
    """

    def __init__(self, path: pathlib.Path = GPU_LOCK, label: str = ""):
        self.path = path
        self.label = label
        self.held = False

    def holder(self) -> tuple[int, str] | None:
        if not self.path.exists():
            return None
        try:
            pid_text, _, label = self.path.read_text().strip().partition(" ")
            return int(pid_text), label
        except ValueError:
            return None  # unreadable lock is a stale lock

    def acquire(self) -> None:
        current = self.holder()
        if current and _pid_alive(current[0]):
            raise Preflight(
                f"the GPU is held by pid {current[0]} ({current[1] or 'unlabelled'}) — "
                f"one run at a time on this card")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()} {self.label}\n")
        self.held = True

    def release(self) -> None:
        """Only ever removes our own lock — a supervisor that took over a stale
        lock must not delete whatever replaced it."""
        current = self.holder()
        if self.held and current and current[0] == os.getpid():
            self.path.unlink(missing_ok=True)
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# --- reading the run ----------------------------------------------------------

def read_progress(metrics_csv: pathlib.Path) -> tuple[int, float | None]:
    """Last (step, ce) in the trainer's metrics CSV; (0, None) if it has none yet.

    Tolerant on purpose: this file is being appended to by another process, so a
    torn final line is normal and must not read as a crashed run.
    """
    try:
        with metrics_csv.open() as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return 0, None
    for row in reversed(rows):
        try:
            step = int(float(row["step"]))
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ce = float(row["ce"])
        except (KeyError, TypeError, ValueError):
            ce = None
        return step, ce
    return 0, None


def plateau_in(log_path: pathlib.Path) -> bool:
    try:
        return "CE Plateau Detected" in log_path.read_text(errors="replace")
    except OSError:
        return False


# --- the loop -----------------------------------------------------------------

@dataclass
class Supervisor:
    """Launches the trainer, watches it, and stops it for a stated reason."""

    command: tuple[str, ...]
    limits: Limits
    log_path: pathlib.Path
    metrics_csv: pathlib.Path
    poll_seconds: float = 300.0
    heartbeat_every: int = 12  # polls between status reports
    report: object = print  # callable(str); the heartbeat's destination
    env: dict = field(default_factory=dict)

    def launch(self) -> subprocess.Popen:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1", **self.env}
        log = self.log_path.open("a")
        return subprocess.Popen(self.command, cwd=REPO_ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)

    def stop(self, proc: subprocess.Popen, grace: float = 60.0) -> None:
        """TERM, then KILL. The grace period is for the trainer to finish writing
        a checkpoint — killing it mid-write is how a run loses its last hour."""
        if proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    def observe(self, proc: subprocess.Popen, started: float) -> Observation:
        step, ce = read_progress(self.metrics_csv)
        return Observation(
            step=step, ce=ce,
            plateau_detected=plateau_in(self.log_path),
            alive=proc.poll() is None,
            elapsed_hours=(time.time() - started) / 3600.0,
        )

    def run(self) -> str:
        """Supervise to a terminal outcome, relaunching after real crashes only."""
        state = State()
        proc = self.launch()
        started = time.time()
        self.report(f"▶ supervising pid {proc.pid}: {' '.join(self.command)}")
        polls = 0

        while True:
            time.sleep(self.poll_seconds)
            polls += 1
            obs = self.observe(proc, started)
            decision = decide(obs, self.limits, state)

            if polls % self.heartbeat_every == 0 or decision.action != CONTINUE:
                self.report(f"{_stamp()} {decision.outcome}: {decision.reason}"
                            + (f" (ce={obs.ce})" if obs.ce is not None else ""))

            if decision.action == CONTINUE:
                continue
            if decision.action in (STOP, KILL):
                self.stop(proc)
                return decision.outcome
            if decision.action == GIVE_UP:
                return decision.outcome
            if decision.action in (RELAUNCH, RESTART):
                if decision.action == RESTART:
                    self.stop(proc)  # wedged, so it is still up and has to go
                proc = self.launch()
                started = time.time()
                self.report(f"{_stamp()} relaunched as pid {proc.pid}")


def _stamp() -> str:
    return datetime.datetime.now().strftime("%F %T")


def github_reporter(issue: int):
    """Heartbeat into a pinned issue, so state lives where CLAUDE.md says it does.

    Never fatal: a supervisor that dies because GitHub was unreachable would
    abandon a running job over a network blip.
    """
    def report(message: str) -> None:
        print(message, flush=True)
        try:
            subprocess.run(["gh", "issue", "comment", str(issue), "--body", message],
                           check=False, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"(heartbeat to #{issue} failed: {exc})", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stop-step", type=int, required=True,
                    help="opt step whose checkpoint covers the token budget (the trainer has no hard stop)")
    ap.add_argument("--run-dir", type=pathlib.Path, required=True,
                    help="the run folder holding metrics.csv")
    ap.add_argument("--log", type=pathlib.Path, required=True, help="where trainer stdout goes")
    ap.add_argument("--issue", type=int, default=None, help="pinned issue to heartbeat into")
    ap.add_argument("--max-ce", type=float, default=Limits.max_ce)
    ap.add_argument("--max-retries", type=int, default=Limits.max_retries)
    ap.add_argument("--stall-polls", type=int, default=Limits.stall_polls,
                    help="polls with no step progress before the run counts as wedged")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=Limits.min_free_gb)
    ap.add_argument("--poll-seconds", type=float, default=300.0)
    ap.add_argument("--no-gpu-lock", action="store_true",
                    help="for a CPU run; the lock exists for the single card")
    ap.add_argument("trainer_args", nargs="*",
                    help="passed through to trm.train.start (put them after --)")
    args = ap.parse_args(argv)

    limits = Limits(stop_step=args.stop_step, max_ce=args.max_ce,
                    stall_polls=args.stall_polls, max_retries=args.max_retries,
                    max_hours=args.max_hours, min_free_gb=args.min_free_gb)

    try:
        free = check_disk_headroom(RUNS_DIR if RUNS_DIR.exists() else REPO_ROOT, limits.min_free_gb)
    except Preflight as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 1
    print(f"preflight: {free:.1f}GB free")

    supervisor = Supervisor(
        command=(sys.executable, "-m", "trm.train.start", *args.trainer_args),
        limits=limits,
        log_path=args.log,
        metrics_csv=args.run_dir / "metrics.csv",
        poll_seconds=args.poll_seconds,
        report=github_reporter(args.issue) if args.issue else print,
    )

    lock = GpuLock(label=f"stop_step={args.stop_step}")
    try:
        if not args.no_gpu_lock:
            lock.acquire()
        outcome = supervisor.run()
    except Preflight as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 1
    finally:
        lock.release()

    print(f"outcome: {outcome}")
    return 0 if outcome in DELIBERATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
