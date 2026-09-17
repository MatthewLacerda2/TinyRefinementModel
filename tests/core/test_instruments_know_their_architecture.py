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

A file that *constructs* a model has made an architecture choice, and a class name
in the middle of a file is where that choice went unseen. So instruments build through
`instruments.arch.build` (which defaults to MODEL_ARCH) or `trm.model.build_model`,
never by class name — unless one line declares the file deliberately arch-specific and
why. The point is not that arch-specific tools are wrong — several are correct — it is
that the decision must be written down, so the next default change surfaces them
instead of silently pointing them at the past.

Known limitations, accepted: both scans read syntax. `build("refiner")` with a literal
arch, or `getattr(model, "refiner")`, walks past them. They exist to catch the accident,
not to outwit a determined author.
"""

import ast
import io
import os
import pathlib
import re
import tokenize

import pytest

INSTRUMENTS = pathlib.Path(__file__).resolve().parents[2] / "instruments"
ARCH_CLASSES = {"RefinerForTraining", "UniversalReasoner", "PlainTransformer", "CausalRefiner"}
# The declaration a deliberately single-architecture instrument writes instead.
EXEMPTION = re.compile(r"#\s*ARCH-SPECIFIC:\s*\S+")
ALL_INSTRUMENTS = sorted(INSTRUMENTS.rglob("*.py"))


def _arch_specific_comments(source):
    """{line: column} of every real `# ARCH-SPECIFIC: <why>` comment.

    Read from the tokenizer, not the text: the same words inside a docstring are a
    string, and a regex over the file let a quotation exempt the whole file (#331 and
    #334 reviews). A comment at column 0 declares the file; an inline one, its line."""
    return {tok.start[0]: tok.start[1]
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT and EXEMPTION.match(tok.string)}


def _undeclared(source, nodes):
    """(line, name) for each (node, name) not covered by an ARCH-SPECIFIC comment."""
    marked = _arch_specific_comments(source)
    if 0 in marked.values():
        return []
    return sorted({(node.lineno, name) for node, name in nodes if node.lineno not in marked})


def direct_constructions(source):
    """Sorted (line, class) for every undeclared call that builds an arch by class name,
    including through an import alias (`from trm.model.plain import PlainTransformer as P`)."""
    tree = ast.parse(source)
    names = {cls: cls for cls in ARCH_CLASSES}
    names.update({alias.asname: alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                  for alias in node.names if alias.name in ARCH_CLASSES and alias.asname})
    calls = ((node, getattr(node.func, "id", getattr(node.func, "attr", None)))
             for node in ast.walk(tree) if isinstance(node, ast.Call))
    return _undeclared(source, ((node, names[name]) for node, name in calls if name in names))


def scan(path):
    """Both scans over one real file. The path is recorded only after its source parsed
    and both scans ran, so `scan_all` reports what was actually read, not what was asked."""
    source = path.read_text()
    found = direct_constructions(source), retired_attribute_accesses(source)
    return path, found


def scan_all(paths):
    """{path: (constructions, attribute accesses)} for every file the scanner read."""
    return dict(scan(path) for path in paths)


@pytest.mark.parametrize("path", ALL_INSTRUMENTS, ids=lambda p: str(p.relative_to(INSTRUMENTS)))
def test_it_builds_through_the_selector_or_declares_why_not(path):
    _, (hits, _) = scan(path)
    assert not hits, (
        f"{path.name} constructs {', '.join(f'{c} (line {n})' for n, c in hits)} by class name. "
        f"Build through `instruments.arch.build` with `--arch` defaulting to MODEL_ARCH, or "
        f"write `# ARCH-SPECIFIC: <why>` if one architecture is genuinely the point. Silently "
        f"building a fixed architecture is how the pre-launch gate spent months smoking the "
        f"control instead of the live bet.")


def test_the_shared_selector_is_itself_arch_aware():
    """Instruments go through `instruments/arch.py` instead of naming a class, which is
    the fix — but it concentrates the risk in one file. If THAT stops reading
    MODEL_ARCH, every caller silently points at a fixed architecture again and the
    per-file guard sees nothing to complain about."""
    selector = (INSTRUMENTS / "arch.py").read_text()
    assert "MODEL_ARCH" in selector
    assert all(a in selector for a in ("plain", "refiner", "reasoner"))


def test_the_constructor_scan_catches_what_it_was_written_for():
    """A guard that matches nothing passes forever. The shape overfit_smoke had must be
    flagged; the selector, a declared file and a docstring quotation must not exempt
    or be flagged wrongly."""
    assert direct_constructions("m = UniversalReasoner(dim, rngs)\n") == [(1, "UniversalReasoner")]
    assert direct_constructions("m = plain.PlainTransformer(dim, rngs)\n") == [(1, "PlainTransformer")]
    assert direct_constructions("m = build(args.arch, dim=dim)\n") == []
    assert direct_constructions("# ARCH-SPECIFIC: refiner only\nm = RefinerForTraining(d, r)\n") == []
    quoted = '"""Notes.\n# ARCH-SPECIFIC: only a quotation\n"""\nm = RefinerForTraining(d, r)\n'
    assert direct_constructions(quoted) == [(4, "RefinerForTraining")]
    aliased = "from trm.model.plain import PlainTransformer as P\nm = P(dim, rngs)\n"
    assert direct_constructions(aliased) == [(2, "PlainTransformer")], "an import alias hides nothing"


def test_the_scans_read_every_real_instrument():
    """The self-checks above run on strings. This one ties the scans to the files: the
    .py files counted by an independent walk of instruments/ (os.walk, not the rglob the
    scan's input comes from) must all come back from the scanner, which records a path
    only once it has parsed and scanned it. A file dropped from the input, or one the
    scanner skipped, fails here."""
    on_disk = {pathlib.Path(root) / name
               for root, _dirs, files in os.walk(INSTRUMENTS)
               if "__pycache__" not in root for name in files if name.endswith(".py")}
    scanned = set(scan_all(ALL_INSTRUMENTS))
    assert scanned == on_disk, f"not scanned: {sorted(map(str, on_disk - scanned))}"
    assert len(scanned) > 20 and INSTRUMENTS / "yardstick" / "eval_yardstick.py" in scanned


# --- reaching into a retired architecture's internals (#314) -------------------
#
# The constructor scan above cannot see the other half of the bug. An instrument can
# build through `instruments.arch.build` (arch-aware, passes) and then walk into
# `model.refiner.encoder` or `model.encoder_stack` — attributes only a retired arch
# has — and crash with AttributeError on `plain`, the default. `smoke_refiner_gpu`
# and `bench_train_step` both did exactly this.

RETIRED_ARCH_ATTRIBUTES = {"refiner", "encoder_stack", "decoder_stack", "reasoning_stack", "hunch_cache"}


def retired_attribute_accesses(source):
    """Sorted (line, attribute) for every undeclared access to a retired arch's internals.

    A whole file declares itself with the comment on a line of its own at column 0; a
    multi-arch file declares each branch into a retired arch with the comment on that line."""
    return _undeclared(source, ((node, node.attr) for node in ast.walk(ast.parse(source))
                                if isinstance(node, ast.Attribute) and node.attr in RETIRED_ARCH_ATTRIBUTES))


@pytest.mark.parametrize("path", ALL_INSTRUMENTS, ids=lambda p: str(p.relative_to(INSTRUMENTS)))
def test_it_does_not_reach_into_a_retired_arch_undeclared(path):
    _, (_, hits) = scan(path)
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
    # The marker's words inside a string are not a declaration, at file or line level.
    docstring = '"""Notes.\n# ARCH-SPECIFIC: only a quotation\n"""\nmodel.refiner\n'
    assert retired_attribute_accesses(docstring) == [(4, "refiner")]
    assert retired_attribute_accesses('x = (model.refiner, "# ARCH-SPECIFIC: in a string")\n') == [(1, "refiner")]
