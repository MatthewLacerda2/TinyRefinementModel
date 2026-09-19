"""An instrument may not silently run in a different environment than production (#166).

`vram_headroom_smoke` set XLA_PYTHON_CLIENT_ALLOCATOR=platform — an allocator that
frees exactly and so cannot fragment — while production ran BFC, then cuda_async.
Fragmentation is what killed every base run (docs/findings/2026-08-14-...). The
instrument had configured away the failure it existed to predict, and was cited as
clearance for a launch anyway.

Divergence is often legitimate. What this forbids is divergence nobody can see at
the moment they read the number: an instrument that sets a key production sets, to
a different value, must declare it —

    ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_ALLOCATOR": "why"}
"""

import ast
import pathlib

import pytest

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the seconds-long
# lint job instead of the jax-heavy pytest job (#325).
pytestmark = pytest.mark.jaxfree

ROOT = pathlib.Path(__file__).resolve().parents[2]




def _env_writes(tree):
    """(key, value, line) for os.environ[K] = V and os.environ.setdefault(K, V)."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                and ast.unparse(node.targets[0].value) == "os.environ"
                and isinstance(node.targets[0].slice, ast.Constant) and isinstance(node.value, ast.Constant)):
            yield node.targets[0].slice.value, node.value.value, node.lineno
        elif (isinstance(node, ast.Call) and ast.unparse(node.func) == "os.environ.setdefault"
              and len(node.args) == 2 and all(isinstance(a, ast.Constant) for a in node.args)):
            yield node.args[0].value, node.args[1].value, node.lineno


def undeclared_divergences(source, production):
    tree = ast.parse(source)
    declared = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "ENV_DIVERGENCES":
            declared = {k: v for k, v in ast.literal_eval(node.value).items() if str(v).strip()}
    return [(key, value, line) for key, value, line in _env_writes(tree)
            if key in production and str(value) != production[key] and key not in declared]


def production_env():
    """What trm/train/start.py sets, read from its own lines — not restated here."""
    env = {key: str(value) for key, value, _ in _env_writes(ast.parse((ROOT / "trm/train/start.py").read_text()))}
    assert env, "trm/train/start.py no longer sets any environment — this lint would check nothing"
    return env


INSTRUMENTS = sorted(p for p in (ROOT / "instruments").rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", INSTRUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_instrument_environment_matches_production_or_says_why(path):
    production = production_env()
    bad = undeclared_divergences(path.read_text(), production)
    assert not bad, "\n".join(
        f"{path.relative_to(ROOT)}:{line} sets {key}={value}; production (trm/train/start.py) "
        f"uses {production[key]} — match it, or declare it in ENV_DIVERGENCES with a reason"
        for key, value, line in bad)


def test_the_lint_catches_the_allocator_that_hid_fragmentation():
    prod = {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async"}
    src = 'import os\nos.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"\n'
    assert undeclared_divergences(src, prod) == [("XLA_PYTHON_CLIENT_ALLOCATOR", "platform", 2)]
    assert undeclared_divergences(src.replace("[", ".setdefault(").replace('"] = ', '", ')
                                  .rstrip() + ")\n", prod)
    assert not undeclared_divergences(src.replace("platform", "cuda_async"), prod)
    assert not undeclared_divergences(
        'ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_ALLOCATOR": "why"}\n' + src, prod)
    assert undeclared_divergences('ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_ALLOCATOR": " "}\n' + src, prod)


def test_production_env_is_read_from_start():
    assert production_env()["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
