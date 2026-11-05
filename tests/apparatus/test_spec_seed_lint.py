"""A spec that registers a seed list registers at least three seeds, checked when it is written.

Rule 1: a single-seed number is not a result, and two seeds give a noise floor that is
itself noise (a 2σ gap is p 0.09 at two seeds a side, 0.035 at three: `docs/design/
experiment-spec.md`). The audit's `holds-at-seed-count` sees this only once results
exist, after the card has spent the run; this lint reads the spec as committed, before
any arm trains, so a short sweep fails while it still costs nothing.

A spec with no seed list (the base run is one model and one data order, written as a
string) is out of scope: it compares against constants, not against its own seeds.
Every spec in the repo registered three or more when this was written.
"""

import pathlib
import tomllib

import pytest

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the lint job (#325).
pytestmark = pytest.mark.jaxfree

MIN_SEEDS = 3

SPECS = sorted((pathlib.Path(__file__).resolve().parents[2] / "experiments")
               .glob("*/specs/*.toml"))


def seed_lists(spec):
    """(where, seeds) for every seed list the spec registers, at any level."""
    execution = spec.get("execution", {})
    found = [("[protocol] seeds", spec.get("protocol", {}).get("seeds")),
             ("[execution] seeds", execution.get("seeds"))]
    found += [(f"[execution.legs.{leg}] seeds", body.get("seeds"))
              for leg, body in execution.get("legs", {}).items()]
    return [(where, seeds) for where, seeds in found if isinstance(seeds, list)]


def short_lists(spec):
    return [f"{where} has {len(seeds)}" for where, seeds in seed_lists(spec) if len(seeds) < MIN_SEEDS]


def test_there_are_specs_to_lint():
    assert SPECS, "no experiment specs found — has the layout moved?"


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_seed_list_has_at_least_three(path):
    short = short_lists(tomllib.loads(path.read_text()))
    assert not short, (
        f"{path.name}: {'; '.join(short)} — register at least {MIN_SEEDS} seeds per arm; "
        f"fewer gives a noise floor that is itself noise (rule 1)."
    )


def test_the_lint_sees_every_level_and_skips_a_seed_string():
    spec = {"protocol": {"seeds": [0, 1]},
            "execution": {"seeds": [0, 1, 2], "legs": {"short": {"seeds": [7]}}}}
    assert short_lists(spec) == ["[protocol] seeds has 2", "[execution.legs.short] seeds has 1"]
    assert short_lists({"protocol": {"seeds": "MODEL_SEED=42, DATA_SEED=42"}}) == []
