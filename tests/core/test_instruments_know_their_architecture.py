"""An instrument that builds a model must know which architecture it is building.

Four checks were found aimed slightly to one side of what they guard, all in one
week, all the same shape:

- `overfit_smoke` — the pre-launch gate CI runs before every training run — built
  `UniversalReasoner` unconditionally. That is the CONTROL. The refiner was the live
  bet for months and never once passed through it.
- `smoke_refiner_gpu` tested f16 overflow on random tokens, on the written grounds
  that "text content is irrelevant to numerical health". The overflow is
  corpus-specific: 65,120 on code against 47.7 on prose (#235).
- `vram_headroom_smoke` — the tool that sizes a run to the card — hardcoded
  `RefinerForTraining`, so after the 2026-09-12 default change it would have sized
  the wrong architecture.
- two determinism tests compared NaN to NaN and passed, because NaN is deterministic
  (#229).

A file that *constructs* a model has made an architecture choice. This requires that
choice to be visible: either it reads `MODEL_ARCH`, or it says in one line that it is
deliberately arch-specific and why. The point is not that arch-specific tools are
wrong — several are correct — it is that the decision must be written down, so the
next default change surfaces them instead of silently pointing them at the past.
"""

import pathlib
import re

import pytest

INSTRUMENTS = pathlib.Path(__file__).resolve().parents[2] / "instruments"
BUILDS_A_MODEL = re.compile(
    r"\b(RefinerForTraining|UniversalReasoner|PlainTransformer|CausalRefiner)\s*\(")
# The declaration a deliberately single-architecture instrument writes instead.
EXEMPTION = re.compile(r"#\s*ARCH-SPECIFIC:\s*\S+")


def _model_building_instruments():
    for path in sorted(INSTRUMENTS.glob("*.py")):
        text = path.read_text()
        if BUILDS_A_MODEL.search(text):
            yield pytest.param(path, id=path.name)


def test_the_shared_selector_is_itself_arch_aware():
    """Most instruments now go through `instruments/arch.py` instead of naming a
    class, which is the fix — but it concentrates the risk in one file. If THAT
    stops reading MODEL_ARCH, every caller silently points at a fixed architecture
    again and the per-file guard below sees nothing to complain about."""
    selector = (INSTRUMENTS / "arch.py").read_text()
    assert "MODEL_ARCH" in selector
    assert all(a in selector for a in ("plain", "refiner", "reasoner"))


def test_the_scan_still_matches_something():
    """A guard that matches nothing passes forever. If the constructor names change,
    this fails before the per-file checks quietly stop looking."""
    found = list(_model_building_instruments())
    assert found, "the constructor pattern matches no instrument — has it gone stale?"


@pytest.mark.parametrize("path", _model_building_instruments())
def test_it_reads_MODEL_ARCH_or_declares_why_not(path):
    text = path.read_text()
    if "MODEL_ARCH" in text:
        return
    assert EXEMPTION.search(text), (
        f"{path.name} constructs a model but neither reads MODEL_ARCH nor declares "
        f"itself arch-specific. Add `--arch` defaulting to MODEL_ARCH, or write a "
        f"line `# ARCH-SPECIFIC: <why>` if one architecture is genuinely the point. "
        f"Silently building a fixed architecture is how the pre-launch gate spent "
        f"months smoking the control instead of the live bet.")


# --- reaching into a retired architecture's internals (#314) -------------------
#
# The constructor scan above cannot see the other half of the bug. An instrument can
# build through `instruments.arch.build` (arch-aware, passes) and then walk into
# `model.refiner.encoder` or `model.encoder_stack` — attributes only a retired arch
# has — and crash with AttributeError on `plain`, the default. `smoke_refiner_gpu`
# and `bench_train_step` both did exactly this.

RETIRED_ARCH_ATTRIBUTES = {"refiner", "encoder_stack", "decoder_stack", "reasoning_stack", "hunch_cache"}
# A whole file declares itself with the marker at the start of a line; a multi-arch
# file declares each branch into a retired arch with the marker on that line.
FILE_EXEMPTION = re.compile(r"^#\s*ARCH-SPECIFIC:\s*\S+", re.M)


def retired_attribute_accesses(source):
    """Sorted (line, attribute) for every undeclared access to a retired arch's internals."""
    if FILE_EXEMPTION.search(source):
        return []
    import ast
    lines = source.splitlines()
    return sorted({(node.lineno, node.attr) for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Attribute) and node.attr in RETIRED_ARCH_ATTRIBUTES
                   and not EXEMPTION.search(lines[node.lineno - 1])})


@pytest.mark.parametrize("path", sorted(INSTRUMENTS.rglob("*.py")), ids=lambda p: str(p.relative_to(INSTRUMENTS)))
def test_it_does_not_reach_into_a_retired_arch_undeclared(path):
    hits = retired_attribute_accesses(path.read_text())
    assert not hits, (
        f"{path.name} reads {', '.join(f'.{a} (line {n})' for n, a in hits)}, attributes only a "
        f"retired architecture has, so it crashes on plain. Branch on the arch and mark that "
        f"line `# ARCH-SPECIFIC: <why>`, refuse the arch by name, or mark the whole file.")


def test_the_attribute_scan_catches_what_it_was_written_for():
    """The two lines #314 found, verbatim, must be flagged; the declared forms and an
    import of the refiner module must not. A scan that flags nothing passes forever."""
    assert retired_attribute_accesses("r = model.refiner\n") == [(1, "refiner")]
    assert retired_attribute_accesses("model.encoder_stack.use_remat = False\n") == [(1, "encoder_stack")]
    assert retired_attribute_accesses(
        "for blk in (*model.refiner.encoder, model.refiner.refine_block):\n    pass\n") == [(1, "refiner")]
    assert retired_attribute_accesses("r = model.refiner  # ARCH-SPECIFIC: refiner branch\n") == []
    assert retired_attribute_accesses("# ARCH-SPECIFIC: reasoner only\nmodel.hunch_cache\n") == []
    assert retired_attribute_accesses("from trm.model.refiner import CausalRefiner\n") == []
