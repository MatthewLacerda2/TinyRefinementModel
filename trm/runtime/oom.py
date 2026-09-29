"""Start long runs where the kernel's OOM killer takes them last, not first (#524).

A Claude shell runs at `oom_score_adj` 200, every child inherits it, and an
unprivileged process can raise its value but never lower it. So a run started from
that shell is born the kernel's preferred victim: #489's trainer was the one killed
when a peer session's build ran the box out of memory. The user manager can grant a
unit it starts anything down to its own value, 100, so a long run is started through
`systemd-run --user -p OOMScoreAdjust=100`. Where there is no user manager (a
container, a box without one running), the run starts as it always did and says
loudly what it inherited.

Stdlib only, like `gpu_lock.py`: the launcher and the experiment runner import it
before anything heavy.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import signal
import subprocess
import time

# The user manager's own oom_score_adj: the lowest it may grant a unit it starts.
OOM_SCORE_ADJ = 100
# Set inside a protected rerun, so the rerun never asks for another one.
RERUN_MARKER = "TRM_OOM_PROTECTED"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# What systemd sets for the unit itself; carrying the launcher's copies over would lie.
_SYSTEMD_OWNED = frozenset({"INVOCATION_ID", "JOURNAL_STREAM", "NOTIFY_SOCKET", "SYSTEMD_EXEC_PID",
                            "MANAGERPID", "LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES",
                            "WATCHDOG_PID", "WATCHDOG_USEC"})


def oom_score_adj(pid: int | str = "self") -> int | None:
    """/proc/<pid>/oom_score_adj, or None when the process is gone or there is no /proc."""
    try:
        return int(pathlib.Path(f"/proc/{pid}/oom_score_adj").read_text())
    except (OSError, ValueError):
        return None


def user_manager_problem() -> str | None:
    """Why `systemd-run --user` cannot start a unit here, or None when it can."""
    if shutil.which("systemd-run") is None:
        return "there is no systemd-run"
    try:
        state = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"systemctl --user did not answer ({exc})"
    # "starting" too: the boot-time resume (#384) runs inside a oneshot whose own start
    # job holds the manager there, and it can start units all the same.
    return None if state in ("running", "degraded", "starting") else f"systemd --user is {state or 'unreachable'}"


def systemd_run(argv: list[str], env: dict[str, str], cwd: pathlib.Path, unit: str, *,
                stdout: pathlib.Path | None = None) -> list[str]:
    """The `systemd-run --user` command that starts `argv` as unit `unit` at OOM_SCORE_ADJ,
    in `cwd` with `env`. With `stdout`, detached: it returns once the process has
    started, output appended to that file, and only the main process is stopped with
    the unit (a milestone scorer outlives the supervisor, as it did before). Without
    it, in the foreground: stdio piped through and the exit code returned."""
    how = (["-p", f"StandardOutput=append:{stdout}", "-p", "StandardError=inherit", "-p", "KillMode=process"]
           if stdout is not None else ["--pipe", "--wait"])
    return ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}", "-p", "Type=exec",
            "-p", f"OOMScoreAdjust={OOM_SCORE_ADJ}", f"--working-directory={cwd}", *how,
            *(f"--setenv={k}={v}" for k, v in sorted(env.items())
              if _ENV_NAME.fullmatch(k) and k not in _SYSTEMD_OWNED),
            "--", *argv]


def unit_name(label: str) -> str:
    """A unit name no earlier launch holds: `trm-<label>-<epoch seconds>`."""
    return f"trm-{re.sub(r'[^A-Za-z0-9_.-]', '-', label)}-{int(time.time())}.service"


def warn_unprotected(why: str) -> None:
    current = oom_score_adj()
    risk = (f" — above {OOM_SCORE_ADJ}, the kernel's OOM killer may take it first (#524)"
            if current is not None and current > OOM_SCORE_ADJ else "")
    print(f"⚠ OOM: {why}, so this run keeps the oom_score_adj it inherited ({current}){risk}", flush=True)


def detach(argv: list[str], env: dict[str, str], cwd: pathlib.Path, stdout: pathlib.Path,
           label: str) -> int | None:
    """Start `argv` detached, output appended to `stdout`; its pid (None if it is gone
    already). Through the user manager where there is one, else as a new session."""
    why = user_manager_problem()
    if why is None:
        unit = unit_name(label)
        proc = subprocess.run(systemd_run(argv, env, cwd, unit, stdout=stdout), capture_output=True, text=True)
        if proc.returncode == 0:
            shown = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                                   capture_output=True, text=True).stdout.strip()
            return int(shown) if shown.isdigit() and int(shown) > 0 else None
        why = f"systemd-run failed (exit {proc.returncode}: {proc.stderr.strip()})"
    warn_unprotected(why)
    with stdout.open("a") as out:
        return subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=cwd,
                                start_new_session=True).pid


def rerun_protected(argv: list[str], env: dict[str, str], cwd: pathlib.Path, label: str) -> int | None:
    """Rerun this process's own `argv` in the foreground under the user manager at
    OOM_SCORE_ADJ, and return its exit code; None when this process should just go
    on: it is that rerun, it already sits at or below OOM_SCORE_ADJ, or there is no
    user manager (said loudly)."""
    current = oom_score_adj()
    if env.get(RERUN_MARKER) or current is None or current <= OOM_SCORE_ADJ:
        return None
    why = user_manager_problem()
    if why:
        warn_unprotected(why)
        return None
    print(f"OOM: oom_score_adj {current} inherited; rerunning under systemd --user at {OOM_SCORE_ADJ} (#524)",
          flush=True)
    unit = unit_name(label)
    proc = subprocess.Popen(systemd_run(argv, {**env, RERUN_MARKER: "1"}, cwd, unit))

    # A TERM or Ctrl-C meant for this runner stops the unit, so the rerun's own TERM
    # handling (#519: stop the arm, then free the card) still runs; killing only this
    # waiting process would leave the sweep on the card.
    def stop_unit(*_):
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)

    previous = signal.signal(signal.SIGTERM, stop_unit)
    try:
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                stop_unit()
    finally:
        signal.signal(signal.SIGTERM, previous)


def banner(pid: int | None) -> str:
    """The launch banner's oom_score_adj line; loud when the run sits above OOM_SCORE_ADJ."""
    value = None if pid is None else oom_score_adj(pid)
    if value is None:
        return f"oom:        oom_score_adj unknown (pid {pid} not readable — exited already?)"
    if value > OOM_SCORE_ADJ:
        return (f"oom:        ⚠ oom_score_adj {value} (pid {pid}) is above {OOM_SCORE_ADJ}: the kernel's "
                f"OOM killer will prefer this run over most of the box (#524)")
    return f"oom:        oom_score_adj {value} (pid {pid})"
