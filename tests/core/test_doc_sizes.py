"""The doctrine stays small enough to read whole: a line limit per file, held by CI.

A CLAUDE.md is loaded into every session and a skill into every task that calls it,
so a file past the size anyone reads whole is a file whose rules quietly stop being
followed. CLAUDE.md's own remedy is to split a piece into a skill, not to grow the
file; this makes "too big" a red build instead of a judgment call nobody makes.

The limits are the owner's, carried over from other projects where they held.
It runs in CI's lint job, which runs on markdown-only changes too.
"""

import pathlib
import subprocess

import pytest

# Needs neither jax, numpy nor tests/conftest.py: runs in the lint job (#325).
pytestmark = pytest.mark.jaxfree

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILLS = pathlib.Path(".claude/skills")

CLAUDE_MD_MAX_LINES = 300  # every CLAUDE.md, at any depth
SKILL_MD_MAX_LINES = 500  # every .md under .claude/skills/


def _tracked_markdown():
    """Tracked .md files only: a local worktree or run folder is not the doctrine."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [pathlib.Path(p) for p in out.split("\0") if p]


def _limited_files():
    """(path, limit) for every file the limits cover."""
    files = []
    for path in _tracked_markdown():
        if path.name == "CLAUDE.md":
            files.append((path, CLAUDE_MD_MAX_LINES))
        elif path.is_relative_to(SKILLS):
            files.append((path, SKILL_MD_MAX_LINES))
    return files


def test_the_guard_is_not_vacuous():
    """If a rename moved CLAUDE.md or the skills, every size check would pass on nothing."""
    paths = [path for path, _ in _limited_files()]
    assert pathlib.Path("CLAUDE.md") in paths, "the root CLAUDE.md is not tracked"
    assert any(p.is_relative_to(SKILLS) for p in paths), f"no .md tracked under {SKILLS}/"


@pytest.mark.parametrize(("path", "limit"), _limited_files(), ids=lambda v: str(v))
def test_doctrine_file_fits_its_line_limit(path, limit):
    lines = len((REPO_ROOT / path).read_text().splitlines())
    assert lines <= limit, (
        f"{path} is {lines} lines, over its limit of {limit}. Split a piece into a skill "
        f"(.claude/skills/<name>/SKILL.md) or a doc it points at; don't raise the limit "
        f"to fit."
    )
