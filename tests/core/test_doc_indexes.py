"""The two indexes a session reads first stay complete, so neither goes stale in silence.

`docs/findings/README.md` is where a session starts digging into what we learned, and
the `instruments` skill is where it looks up which instrument answers a question. An
entry missing from either is knowledge nobody finds. Both used to be kept by a line of
prose asking for it; one finding and two entry points had slipped past.
"""

import ast
import pathlib
import re

import pytest

# Needs neither jax, numpy nor tests/conftest.py: runs in the lint job (#325).
pytestmark = pytest.mark.jaxfree

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FINDINGS = REPO_ROOT / "docs" / "findings"
INSTRUMENTS = REPO_ROOT / "instruments"
INSTRUMENTS_SKILL = REPO_ROOT / ".claude" / "skills" / "instruments" / "SKILL.md"


def _findings():
    return sorted(p.name for p in FINDINGS.glob("*.md") if p.name != "README.md")


def _entry_points():
    """Dotted module names under instruments/ that run as `python -m instruments.<name>`."""
    names = []
    for path in sorted(INSTRUMENTS.rglob("*.py")):
        if "__pycache__" in path.parts or '__name__ == "__main__"' not in path.read_text():
            continue
        names.append(".".join(path.relative_to(INSTRUMENTS).with_suffix("").parts))
    return names


def test_the_indexes_have_something_to_check():
    assert _findings(), f"no findings under {FINDINGS}"
    assert _entry_points(), f"no entry points under {INSTRUMENTS}"


@pytest.mark.parametrize("name", _findings())
def test_every_finding_is_indexed(name):
    assert f"`{name}`" in (FINDINGS / "README.md").read_text(), (
        f"docs/findings/{name} is not in docs/findings/README.md: add a line "
        f"\"- `{name}` — <the claim in a dozen words>\" under Entries."
    )


@pytest.mark.parametrize("name", _entry_points())
def test_every_instrument_is_in_the_skill_index(name):
    named = re.search(rf"`(?:make \w+ )?{re.escape(name)}[`\s\[]", INSTRUMENTS_SKILL.read_text())
    assert named, (
        f"instruments/{name.replace('.', '/')}.py runs as a module but the instruments "
        f"skill's index never names it: add a row with the question it answers."
    )


@pytest.mark.parametrize("name", _entry_points())
def test_every_instrument_docstring_carries_its_usage_line(name):
    path = INSTRUMENTS / f"{name.replace('.', '/')}.py"
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    assert f"-m instruments.{name}" in doc, (
        f"{path.relative_to(REPO_ROOT)}'s module docstring is its manual: it opens with the "
        f"question it answers and a usage line, `python -m instruments.{name} ...`."
    )
