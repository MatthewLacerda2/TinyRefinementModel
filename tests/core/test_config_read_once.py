"""Config is read from the environment in one place (#475).

Knobs used to be read with `os.environ.get` at import time in eleven files, each with
its own parsing and its own default; a value frozen at one module's import could not
be told apart from a value read later, and nothing recorded the ones nobody listed.
`trm.settings.Config.from_env` is now the only reader of a knob. This scans `trm/`
for any other read of the environment, so a new `os.environ.get("SOME_KNOB", ...)`
fails here instead of becoming a hidden default on the hot path.

Two uses are not reads and pass anywhere: handing the whole environment to a child
process (`{**os.environ, ...}`) and setting a variable JAX reads
(`os.environ.setdefault`). Anything else outside trm/settings.py is named below
with its reason.
"""

import ast
import pathlib

import pytest

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the lint job (#325).
pytestmark = pytest.mark.jaxfree

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
READER = "trm/settings.py"

# (file, enclosing function) -> why this read is not a knob read.
NOT_KNOBS = {
    ("trm/config.py", "<module>"):
        "JAX_COMPILATION_CACHE_DIR is JAX's own variable: when set, JAX's value wins over ours",
    ("trm/runtime/supervisor.py", "preflight_fit"):
        "PYTEST_CURRENT_TEST refuses to launch the real trainer from inside a test",
}


def _environment_reads(source):
    """(enclosing function, line) of every use of os.environ / os.getenv that is not
    a whole-environment hand-off or a setdefault."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def enclosing(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return "<module>"

    def is_environment(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            return node.attr in ("environ", "getenv")
        return False

    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            reads += [(enclosing(node), node.lineno) for a in node.names if a.name in ("environ", "getenv")]
        if not is_environment(node):
            continue
        parent = parents.get(node)
        handed_off = isinstance(parent, ast.Dict) and any(
            key is None and value is node for key, value in zip(parent.keys, parent.values))
        setting = isinstance(parent, ast.Attribute) and parent.attr == "setdefault"
        if not (handed_off or setting):
            reads.append((enclosing(node), node.lineno))
    return reads


def _library_files():
    return sorted(p for p in (REPO_ROOT / "trm").rglob("*.py") if "__pycache__" not in p.parts)


def test_only_settings_reads_a_knob_from_the_environment():
    stray = []
    for path in _library_files():
        name = str(path.relative_to(REPO_ROOT))
        if name == READER:
            continue
        for function, line in _environment_reads(path.read_text()):
            if (name, function) not in NOT_KNOBS:
                stray.append(f"{name}:{line} (in {function})")
    assert not stray, (
        "these read the environment outside trm/settings.py:\n  " + "\n  ".join(stray)
        + "\nA knob is a field of trm.settings.Config, read by Config.from_env; a path is "
          "trm.settings.location. Anything else goes in NOT_KNOBS with its reason.")


def test_every_exemption_is_still_used():
    """An exemption whose read is gone would quietly excuse the next one."""
    used = {(str(p.relative_to(REPO_ROOT)), function)
            for p in _library_files() for function, _ in _environment_reads(p.read_text())}
    assert not set(NOT_KNOBS) - used, f"stale exemptions: {sorted(set(NOT_KNOBS) - used)}"


def test_the_scan_sees_a_read_and_passes_a_hand_off():
    """The guard is only as good as its detector: a read must register, the two
    non-reads must not."""
    assert _environment_reads('import os\nX = int(os.environ.get("X", "1"))\n') == [("<module>", 2)]
    assert _environment_reads('import os\ndef f():\n    return os.getenv("X")\n') == [("f", 3)]
    assert _environment_reads('from os import environ\n') == [("<module>", 1)]
    assert _environment_reads('import os\nenv = {**os.environ, "A": "1"}\n') == []
    assert _environment_reads('import os\nos.environ.setdefault("XLA_FLAGS", "")\n') == []
