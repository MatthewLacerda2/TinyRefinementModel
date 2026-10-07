"""The process a candidate program actually runs in. Never imported — run by path.

`sandbox.verify` starts this file with a plain `python <path>`, so it belongs to no
package and imports nothing from the repo. That matters twice over: the program it
is about to execute must not be able to reach the trainer's modules, and a mistake
in here must not be able to take the parent down with it.

It reads one JSON object on stdin — the program, its tests, the limits and a nonce —
and writes one JSON object back on stdout, carrying that nonce. Everything else on
stdout is noise the parent ignores, which is deliberate: a candidate program that
prints is normal, and one that prints something shaped like a report cannot be
mistaken for this one.

**What is actually guaranteed, and what is only discouraged.** The kernel limits are
the real containment and hold no matter what the program does to Python:

- `RLIMIT_FSIZE = 0` — no file can be written. (Pipes are not regular files, so the
  answer still gets out.)
- `RLIMIT_AS` — the address space is capped, so a memory bomb raises `MemoryError`
  instead of swapping the box to death.
- `RLIMIT_CPU` — a spin loop burns its allowance and dies, with the parent's
  wall-clock timeout behind it as the backstop that always fires.

The import block below is the *discouraged* half: it stops `import os` and its
neighbours, which is what a language model writes when it writes something
dangerous. It is not a jail. A program that goes looking — down `__subclasses__`,
say — can still reach a hidden reference to something nasty, and then finds it
cannot write, cannot grow and cannot run for long. Do not read this file as
sandboxing hostile code; read it as making a careless completion harmless.

Network access is blocked by the import block alone, which is to say by the weaker
half. If this world is ever pointed at code that was not generated locally, the
answer is a container, not another name in the list.
"""

import builtins
import json
import resource
import signal
import sys

# Anything that reaches the operating system, the network, or another process. The
# list is short because it only has to cover what a model plausibly emits; it is not
# trying to be exhaustive, and it could not be.
BLOCKED = frozenset({
    "os", "posix", "nt", "subprocess", "_posixsubprocess", "multiprocessing",
    "socket", "_socket", "ssl", "http", "urllib", "urllib3", "requests", "ftplib",
    "telnetlib", "smtplib", "shutil", "pathlib", "tempfile", "glob", "fileinput",
    "ctypes", "mmap", "pty", "fcntl", "signal", "pickle", "marshal", "importlib",
    "runpy", "webbrowser", "site", "sysconfig",
})

# Imported for the candidate before the door shuts. Several of these pull in `os` on
# their way up (`random` does), so they have to be resolved while that is still
# allowed — and a solution that reaches for `collections.Counter` is a good solution,
# not a suspicious one. Anything outside both lists can still be imported if it
# happens to need nothing blocked.
ALLOWED = ("math", "itertools", "functools", "collections", "collections.abc", "re",
           "string", "heapq", "bisect", "operator", "copy", "random", "statistics",
           "fractions", "decimal", "datetime", "textwrap", "unicodedata", "typing",
           "dataclasses", "enum", "array", "numbers", "json")

_real_import = builtins.__import__


def _blocked_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in BLOCKED:
        raise ImportError(f"{root!r} is not available in the verifier")
    return _real_import(name, *args, **kwargs)


def lock_down(memory_bytes, cpu_seconds):
    """Apply the limits, then take the dangerous names away.

    Order matters: the limits come from `resource` and the candidate's stdlib comes
    from `ALLOWED`, and the import block would stop us reaching either one if it
    were installed first.
    """
    for name in ALLOWED:
        try:
            _real_import(name)
        except ImportError:
            pass  # a stdlib that moved or was trimmed; the candidate loses one name

    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    # Writing past RLIMIT_FSIZE raises SIGXFSZ, which kills by default. Ignored, the
    # write returns an error instead and the program sees an ordinary OSError — a
    # failed attempt, which is what it is, rather than a dead process the parent has
    # to guess about.
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

    for name in BLOCKED:
        sys.modules.pop(name, None)
    builtins.__import__ = _blocked_import
    # `open` is the other door to the filesystem and no solution to these tasks
    # needs it. Reads are harmless; taking it away costs nothing and says clearly
    # that this program is not here to touch files.
    del builtins.open


def _status_of(error):
    if isinstance(error, AssertionError):
        return "wrong_answer"
    if isinstance(error, ImportError):
        return "forbidden"
    if isinstance(error, MemoryError):
        return "memory"
    return "error"


def run(program, tests):
    """Execute the program, then each test, and report what happened.

    A test is a statement, so a passing one produces nothing and a failing one
    raises. The first failure names the whole attempt — the readout only ever asks
    "did this pass", and a human reading a transcript wants the first thing that
    went wrong, not the last.
    """
    namespace = {"__name__": "__candidate__"}
    try:
        compile(program, "<candidate>", "exec")
    except SyntaxError as err:
        return {"status": "syntax_error", "passed": 0, "total": len(tests),
                "detail": f"line {err.lineno}: {err.msg}"}
    try:
        exec(program, namespace)  # executing the candidate is the whole job
    except BaseException as err:  # a candidate may raise literally anything
        return {"status": _status_of(err), "passed": 0, "total": len(tests),
                "detail": f"{type(err).__name__}: {err}"}

    passed, status, detail = 0, "ok", ""
    for test in tests:
        try:
            exec(test, namespace)
            passed += 1
        except BaseException as err:
            if status == "ok":
                status = _status_of(err)
                detail = f"{test}  ->  {type(err).__name__}: {err}".strip()
    return {"status": status, "passed": passed, "total": len(tests), "detail": detail}


def main():
    request = json.loads(sys.stdin.read())
    lock_down(int(request["memory_bytes"]), int(request["cpu_seconds"]))
    report = run(request["program"], list(request["tests"]))
    report["nonce"] = request["nonce"]
    # Straight to the real stdout: the program may have replaced sys.stdout, and a
    # candidate that prints must not be able to corrupt the answer.
    sys.__stdout__.write("\n" + json.dumps(report) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
