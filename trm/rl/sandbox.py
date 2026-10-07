"""Run a candidate program against its tests and say what happened.

This is the half of the world that makes it *verifiable*: no model scores another
model, no judge decides whether an answer is good enough. A test either passed or
it did not, and the reason it did not is one of a handful of named things.

The work happens in `_sandbox_child.py`, in its own process, under kernel limits —
read that file for what is actually guaranteed. Here we only start it, feed it, and
translate however it died into a status.

Costs: a process per attempt, so roughly 40-60 ms of interpreter startup each. That
is nothing next to generating the attempt, and it buys the isolation. Verifying a
batch in parallel is a thread pool away when it starts to matter.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CHILD = Path(__file__).with_name("_sandbox_child.py")

DEFAULT_TIMEOUT_S = 6.0
# Comfortably above what the interpreter has already mapped when the limit goes on,
# and far below what a runaway allocation wants. A task that needs more than this is
# not one of ours.
DEFAULT_MEMORY_MB = 512

# What the child reports, plus the two only the parent can see.
STATUSES = ("ok", "wrong_answer", "error", "syntax_error", "forbidden", "memory",
            "timeout", "crash")


@dataclass(frozen=True)
class Outcome:
    """One attempt's verdict. `passed`/`total` are per test; `status` is the attempt."""
    status: str
    passed: int
    total: int
    detail: str = ""

    @property
    def solved(self) -> bool:
        return self.status == "ok"


def verify(program: str, tests, *, timeout_s: float = DEFAULT_TIMEOUT_S,
           memory_mb: int = DEFAULT_MEMORY_MB) -> Outcome:
    """Run `program`, then each statement in `tests`, in a throwaway process."""
    tests = list(tests)
    # The candidate is free to print, and printing a plausible report is one of the
    # few ways it could lie about its own result. The nonce makes the answer
    # unforgeable without having to reason about what a program might do.
    nonce = secrets.token_hex(8)
    request = json.dumps({"program": program, "tests": tests, "nonce": nonce,
                          "memory_bytes": memory_mb * 1024 * 1024,
                          "cpu_seconds": max(1, int(timeout_s))})
    with tempfile.TemporaryDirectory(prefix="trm-sandbox-") as workdir:
        try:
            done = subprocess.run(
                [sys.executable, str(CHILD)], input=request, cwd=workdir,
                capture_output=True, text=True, timeout=timeout_s,
                # A bare environment: nothing about this box, no proxy, no home, and
                # no bytecode written beside a file the program might have made.
                env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PATH": ""},
            )
        except subprocess.TimeoutExpired:
            return Outcome("timeout", 0, len(tests), f"no answer within {timeout_s:g}s")

    report = _last_report(done.stdout, nonce)
    if report is not None:
        return Outcome(report["status"], int(report["passed"]), int(report["total"]),
                       report.get("detail", ""))
    # The child never answered. Either the kernel stopped it — SIGXCPU when the CPU
    # allowance ran out, SIGKILL when something larger did — or it broke in a way it
    # could not report, which is a bug in the verifier and must not read as a failed
    # attempt by the model.
    if done.returncode in (-24, -14):
        return Outcome("timeout", 0, len(tests), "killed on its CPU allowance")
    if done.returncode == -9:
        return Outcome("memory", 0, len(tests), "killed from outside, most likely memory")
    return Outcome("crash", 0, len(tests),
                   f"verifier returned {done.returncode}: {done.stderr.strip()[-400:]}")


def verify_task(program: str, task, **limits) -> Outcome:
    """`verify` against a `tasks.Task`, which is what every caller actually has."""
    return verify(program, task.tests, **limits)


def _last_report(stdout: str, nonce: str):
    """The child's answer is the JSON object on stdout carrying this call's nonce.
    The candidate is free to print whatever it likes around it, and often does."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(report, dict) and report.get("nonce") == nonce
                and report.get("status") in STATUSES):
            return report
    return None
