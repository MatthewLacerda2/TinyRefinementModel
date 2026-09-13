"""Every instrument says what kind of number it prints (#175).

Two instruments misled decisions without computing anything wrong: `mem_profile`'s
table was cumulative bytes read as peak (37 GiB against a real 0.58 GiB), and
`vram_headroom_smoke`'s "peak" was a 150ms nvidia-smi poll (batch 2 read cheaper
than batch 1). Precision was never distinguished from fidelity.

So an entry point declares, next to the code, each headline number's kind:

    REPORTS = {"arena peak": ("measured", "memory_stats() peak_bytes_in_use")}

The machine check is small on purpose. The value is the moment an author writes
("cumulative", ...) beside a field they were about to call "peak".
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
KINDS = {"measured", "sampled", "estimated", "cumulative"}


def is_entry_point(tree):
    has_main = any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
    guarded = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)
    return has_main or guarded


def declaration(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "REPORTS":
            return ast.literal_eval(node.value)
    return None


def problems(source):
    tree = ast.parse(source)
    if not is_entry_point(tree):
        return []
    reports = declaration(tree)
    if reports is None:
        return ["no module-level REPORTS: declare each headline number's kind, or REPORTS = {} "
                "with a comment if it prints none"]
    found = []
    for name, value in reports.items():
        if not (isinstance(value, tuple) and len(value) == 2):
            found.append(f"{name!r}: expected (kind, how)")
            continue
        kind, how = value
        if kind not in KINDS:
            found.append(f"{name!r}: kind {kind!r} is not one of {sorted(KINDS)}")
        if not str(how).strip():
            found.append(f"{name!r}: say how it was obtained")
        if "peak" in name.lower() and kind == "cumulative":
            found.append(f"{name!r}: a cumulative quantity is not a peak (#66)")
    return found


INSTRUMENTS = sorted(p for p in (ROOT / "instruments").rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", INSTRUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_entry_point_declares_what_it_reports(path):
    found = problems(path.read_text())
    assert not found, f"{path.relative_to(ROOT)}: " + "; ".join(found)


def test_the_check_fires_on_the_failures_that_motivated_it():
    assert problems("def main(): pass") != []
    assert problems('REPORTS = {"peak": ("cumulative", "size x count")}\ndef main(): pass') != []
    assert problems('REPORTS = {"peak": ("exact-ish", "poll")}\ndef main(): pass') != []
    assert problems('REPORTS = {"peak": ("sampled", " ")}\ndef main(): pass') != []
    assert problems('REPORTS = {"peak": ("measured", "memory_stats")}\ndef main(): pass') == []
    assert problems("def helper(): pass") == [], "a library module with no entry point needs nothing"
