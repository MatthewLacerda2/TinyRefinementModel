"""A finding published after the referee became usable must say where it came from.

`instruments/verdict.py` and `instruments/experiment.py` were built in #150/#151 and
ran for the first time in September 2026 — the machine existed for months and judged
nothing. #231's premise was that it was avoided out of discipline. It was not: the
first two things anyone would type (a `command` as a string, a bare `python`) both
failed, one of them silently, and six defects in total surfaced the moment it was
actually used.

Now that it is walkable, publication is what gets gated — not exploration. Poking at
a model with a scratch script is how hypotheses are born and needs no ceremony. What
needs a verdict is a result becoming *repo knowledge*: a `docs/findings/` entry.

So an entry dated on or after the cutoff must carry either

    Spec: experiments/<line>/specs/<id>-<slug>.toml

naming a file that exists, or

    Evidence: observational — <why no spec applies>

Observational is a real category, not a loophole: #229's all-NaN logits were found by
accident while running something else, and there is no control arm for "this
checkpoint is broken". Saying so is honest; leaving it unsaid is how a chat log
becomes a citation.

Everything before the cutoff is grandfathered. Those findings predate the format and
rewriting their provenance now would be inventing it.
"""

import datetime
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FINDINGS = REPO / "docs" / "findings"

# The referee's first real use. Entries from this day forward carry provenance.
CUTOFF = datetime.date(2026, 9, 12)

DATED = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-")
SPEC_LINE = re.compile(r"^Spec:\s*(.+)$", re.M)
OBSERVATIONAL = re.compile(r"^Evidence:\s*observational\b\s*[—-]\s*\S+", re.M)


def _entries():
    for path in sorted(FINDINGS.glob("*.md")):
        m = DATED.match(path.name)
        if not m:
            continue          # README and plots
        date = datetime.date(*(int(g) for g in m.groups()))
        yield pytest.param(path, date, id=path.stem)


def test_there_are_findings_to_check():
    assert list(_entries()), "no dated findings found — has the folder moved?"


@pytest.mark.parametrize("path,date", _entries())
def test_a_finding_names_its_spec_or_declares_itself_observational(path, date):
    if date < CUTOFF:
        return                # grandfathered: predates the format

    text = path.read_text()
    cited = [c.strip() for line in SPEC_LINE.findall(text)
             for c in line.split(",") if c.strip()]
    if cited:
        # A finding may rest on several. Every one must resolve: a citation that
        # cannot be followed is worse than none, because it looks like provenance.
        missing = [c for c in cited if not (REPO / c).exists()]
        assert not missing, f"{path.name} cites specs that do not exist: {missing}"
        return

    assert OBSERVATIONAL.search(text), (
        f"{path.name} is dated {date} and carries neither a `Spec:` line nor an "
        f"`Evidence: observational — <why>` declaration.\n\n"
        f"A result becomes repo knowledge here. Say which pre-registered spec judged "
        f"it, or say plainly that it was observational and why no control applies "
        f"(as #229's accidental NaN discovery was). Exploration needs no ceremony; "
        f"publication does.")
