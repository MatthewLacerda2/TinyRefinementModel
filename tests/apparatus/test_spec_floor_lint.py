"""A parity criterion is only meaningful if both arms are doing the task.

#246 build 1 registered chance as 1/7 = 0.143 on a mod-7 arithmetic task.
Multiplication mod 7 makes zero absorbing, so 34% of nest-4 expressions evaluate to
0 and the real majority-class floor was 0.340. Both arms scored 0.334-0.341 — they
had learned "always guess the mode" and nothing else — and the parity criterion
called it a pass.

**Two arms that both fail are trivially "within 2 sigma" of each other.** That is not
a subtle statistical point; it is the default outcome whenever a task is too hard,
and a parity bar cannot tell it apart from success.

So a spec whose KEEP depends on parity must either declare an absolute floor (a
`constant = true` arm with a `beats` criterion against it) or say in `[protocol]
floor_note` why a floor is unnecessary here. The note is not a loophole: for a
difference metric the null really is zero and a floor is meaningless, and for two
arms that visibly clear chance by 4x it is ceremony. The requirement is that
somebody *thought about it*, in writing, before the run.

Audited when written: all six specs then in the repo had a parity criterion in
`keep_if` and none had a floor. One of them (#79) recorded an arm at 0.1323 against
a chance of 0.143 — below chance — which is exactly the shape this guards.
"""

import pathlib
import tomllib

import pytest

SPECS = sorted((pathlib.Path(__file__).resolve().parents[2] / "experiments")
               .glob("*/specs/*.toml"))


def _load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _specs():
    for path in SPECS:
        yield pytest.param(path, id=path.stem)


def test_there_are_specs_to_lint():
    """A lint that walks an empty directory passes forever."""
    assert SPECS, "no experiment specs found — has the layout moved?"


@pytest.mark.parametrize("path", _specs())
def test_a_parity_keep_declares_a_floor_or_says_why_not(path):
    spec = _load(path)
    criteria = spec.get("criteria", {})
    keep_if = spec.get("verdict", {}).get("keep_if", [])
    constants = {name for name, body in spec.get("arms", {}).items()
                 if body.get("constant")}

    parity = [k for k in keep_if if criteria.get(k, {}).get("rule") == "within"]
    if not parity:
        return

    has_floor = any(
        criteria.get(k, {}).get("rule") == "beats"
        and criteria[k].get("control") in constants
        for k in keep_if)
    if has_floor:
        return

    note = spec.get("protocol", {}).get("floor_note", "").strip()
    assert note, (
        f"{path.name}: KEEP depends on the parity criterion {parity}, but the spec "
        f"declares no floor and gives no reason.\n\n"
        f"Two arms that both FAIL a task are trivially 'within 2 sigma' of each "
        f"other — #246 build 1 passed a parity criterion with both arms sitting "
        f"exactly on the majority-class baseline.\n\n"
        f"Either add a `constant = true` arm with the floor MEASURED from the task "
        f"generator plus a `beats` criterion against it, or write "
        f"`[protocol] floor_note = \"...\"` explaining why a floor does not apply "
        f"(a difference metric whose null is zero, say).")


@pytest.mark.parametrize("path", _specs())
def test_a_declared_floor_is_not_left_empty(path):
    """A constant arm with no recorded value is a floor that silently is not one —
    `evaluate` would fail on it only after the whole sweep had finished, which is
    how #246 lost 18 runs."""
    spec = _load(path)
    results = spec.get("results", {})
    for name, body in spec.get("arms", {}).items():
        if not body.get("constant"):
            continue
        declared = [pt for pt, arms in results.items() if name in arms]
        assert declared, (
            f"{path.name}: arm '{name}' is constant = true but has no value in "
            f"[results]. A constant arm is never run, so its value can only come "
            f"from the spec; without one, every criterion naming it fails at "
            f"judging time — after the sweep has already been spent.")


@pytest.mark.parametrize("path", _specs())
def test_no_spec_still_carries_a_scaffold_placeholder(path):
    """`--new` writes TODOs on purpose: an unfilled spec should fail loudly rather
    than run with a placeholder hypothesis nobody meant. A spec is committed BEFORE
    it runs, so a TODO that reached the repo is one that was never filled in."""
    text = path.read_text()
    assert "TODO" not in text, (
        f"{path.name} still carries a scaffold placeholder. Fill every TODO before "
        f"committing — the criteria are only criteria because git records that they "
        f"preceded the numbers, and a placeholder preceded nothing.")
