"""Which tests can a change reach? For a tight local loop, never for CI (#169).

    python tests/affected.py            # pytest arguments for what changed vs main
    make test-affected

CI runs core + apparatus on every push, and that is right for CI. Locally, and for
any runner that iterates, it is minutes per edit. This follows imports instead:
a test is selected when it imports — directly or through any chain of repo
modules — a module that changed, or when it names the changed file's tree or
filename in a string (the lints that read source by path rather than importing
it: package layout, instrument defaults and environment, the findings gate).

It fails OPEN, never closed. A missed test is worse than a slow one, so anything
it cannot reason about selects the whole suite: a change outside Python and docs
(config files, specs, CI, the Makefile), conftest.py, a file that does not parse,
or a changed module nothing imports by name (it may be run as `python -m`).
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TREES = ("trm", "instruments", "experiments")
SUITE = ["tests/core", "tests/apparatus"]
DOC_SUFFIXES = {".md", ".txt", ".svg", ".png"}


def module_name(path: pathlib.Path) -> str:
    parts = list(path.with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path: pathlib.Path) -> set[str] | None:
    """Every module name this file imports, with `from a import b` counted as both
    `a` and `a.b` (b may be a submodule). None when the file does not parse."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def strings_in(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _test_files(repo):
    return [f for tier in ("core", "apparatus") for f in (repo / "tests" / tier).glob("test_*.py")]


def named_by_path(path: pathlib.PurePosixPath, repo: pathlib.Path) -> set[str]:
    """Tests that refer to this file's tree or its name in a string literal."""
    top, name = path.parts[0], path.name
    hits = set()
    for file in _test_files(repo):
        for text in strings_in(file):
            words = text.replace("\\", "/").split("/")
            if top in words or name in words or text == name:
                hits.add(file.relative_to(repo).as_posix())
                break
    return hits


def select(changed: list[str], repo: pathlib.Path = REPO) -> list[str]:
    """pytest arguments covering every test the changed paths can reach."""
    tests, modules = set(), set()
    for rel in changed:
        path = pathlib.PurePosixPath(rel)
        tests |= named_by_path(path, repo)
        if path.suffix in DOC_SUFFIXES or (path.parts and path.parts[0] == "docs"):
            continue
        if path.suffix != ".py":
            return SUITE
        if path.parts[0] == "tests":
            if path.name == "conftest.py" or len(path.parts) < 3:
                return SUITE
            if path.parts[1] in ("core", "apparatus") and path.name.startswith("test_"):
                tests.add(rel)
                continue
            return SUITE
        if path.parts[0] not in TREES:
            return SUITE
        modules.add(module_name(pathlib.Path(rel)))

    if modules:
        graph = {}
        for tree in TREES + ("tests",):
            for file in (repo / tree).rglob("*.py"):
                if "__pycache__" in file.parts:
                    continue
                found = imports_of(file)
                if found is None:
                    return SUITE
                graph[file.relative_to(repo).as_posix()] = found
        reached, frontier = set(modules), set(modules)
        while frontier:
            frontier = {module_name(pathlib.Path(f)) for f, imps in graph.items()
                        if not f.startswith("tests/") and imps & frontier} - reached
            reached |= frontier
        hits = {f for f, imps in graph.items()
                if f.startswith(("tests/core/", "tests/apparatus/")) and pathlib.Path(f).name.startswith("test_")
                and imps & reached}
        if not hits:
            return SUITE  # nothing imports it by name: it may be run as a module, so assume everything
        tests |= hits
    return sorted(tests)


def changed_vs_main(repo: pathlib.Path = REPO) -> list[str]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    base = git("merge-base", "HEAD", "main")[0]
    return sorted(set(git("diff", "--name-only", base)) | set(git("ls-files", "--others", "--exclude-standard")))


if __name__ == "__main__":
    changed = changed_vs_main()
    selection = select(changed) if changed else []
    print(" ".join(selection))
    print(f"# {len(changed)} changed path(s) -> {'full suite' if selection == SUITE else f'{len(selection)} test file(s)'}",
          file=sys.stderr)
