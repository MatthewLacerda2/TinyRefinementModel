"""The tree's dependency direction, held by a check rather than by good manners (#143).

The folders are a declaration of *kind*, and kind implies a direction:

    trm/          the library and its operations — lives forever
    experiments/  one folder per research line — dies as a unit when tombstoned
    instruments/  permanent measuring gear — smokes, benches, the yardstick

`trm/` must not import from either leaf tree, or the library acquires a
dependency on something CLAUDE.md rule 6 will one day delete in a single diff.
And one research line must not import another, or "kill the folder, kill the
line" stops being a local operation.

Prose conventions rot. This is the same species as the tier guard in
tests/conftest.py: the folder is the declaration, and the machine holds the line.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TREES = ("trm", "experiments", "instruments")


def _module_files(tree):
    return sorted(p for p in (REPO_ROOT / tree).rglob("*.py") if "__pycache__" not in p.parts)


def _imported_roots(path):
    """Top-level dotted prefixes this file imports, e.g. {'trm.model', 'instruments'}."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _all_module_files():
    return [(tree, p) for tree in TREES for p in _module_files(tree)]


def test_trees_exist():
    """If a rename silently drops a tree, every check below would pass vacuously."""
    for tree in TREES:
        assert (REPO_ROOT / tree).is_dir(), f"{tree}/ is missing"
        assert _module_files(tree), f"{tree}/ has no modules — the guard would be vacuous"


@pytest.mark.parametrize(
    "path", _module_files("trm"), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_library_never_imports_from_the_leaf_trees(path):
    leaked = sorted(
        name for name in _imported_roots(path)
        if name.split(".")[0] in ("experiments", "instruments")
    )
    assert not leaked, (
        f"{path.relative_to(REPO_ROOT)} imports {leaked}. trm/ must not depend on a "
        f"tree that gets deleted with its research line — move the shared piece into trm/."
    )


def test_research_lines_do_not_import_each_other():
    lines_dir = REPO_ROOT / "experiments"
    lines = sorted(p.name for p in lines_dir.iterdir()
                   if p.is_dir() and p.name != "__pycache__")
    offences = []
    for line in lines:
        for path in (p for p in (lines_dir / line).rglob("*.py")
                     if "__pycache__" not in p.parts):
            for name in _imported_roots(path):
                parts = name.split(".")
                if parts[0] == "experiments" and len(parts) > 1 and parts[1] != line:
                    offences.append(f"{path.relative_to(REPO_ROOT)} -> {name}")
    assert not offences, (
        "research lines must be independent, so tombstoning one is a single "
        "folder deletion:\n  " + "\n  ".join(offences))


def test_repo_root_holds_no_python():
    """Entry points are `python -m trm.train.start` and friends, so a `.py` in the
    root has no declared home — which is the condition #143 exists to end. Ignored
    scratch (`aux*`) doesn't count; this walks tracked files only."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.split()
    in_root = sorted(f for f in tracked if "/" not in f)
    assert not in_root, (
        f"tracked Python in the repo root: {in_root}. Every module belongs to one of "
        f"{TREES} or tests/; entry points are run with `python -m <pkg>.<module>`."
    )
