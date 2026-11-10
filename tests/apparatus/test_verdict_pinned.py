"""Every committed spec parses, and a spec without `[stages]` is judged exactly as
the referee judged it before #604 added group-sequential looks.

`fixtures/verdicts_before_604.json` was written by `pin()` below, run on the
referee at main 1ee0232, before any #604 change. It holds every number a verdict
carries (outcome, reason, each point's σ, Δ, p, seeds and holds, and the printed
`describe()`), so the comparison is `==` on floats, not approx: bit for bit.

The base runs are judged once, at completion, by `instruments.base_run.verdict_for`,
which hands the referee one LAMBADA accuracy beside the constant GPT-2-small. Their
specs record no measured result yet, so the pin also judges them at a grid of
accuracies around both of their bars (0.3256 to keep, 0.05 below it to kill) in
that same results shape. That is base run 002's verdict path, pinned before it runs.

A spec edited since the pin (its results landed, say) is skipped with its reason:
the pin guards #604's change, not every later edit. Re-pin with
`python -m tests.apparatus.test_verdict_pinned` only to add specs, never to make a
changed verdict pass.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from instruments.spec_file import SpecFile
from instruments.verdict import evaluate, load_recorded_results, load_spec

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = sorted(ROOT.glob("experiments/*/specs/*.toml"))
PIN = pathlib.Path(__file__).parent / "fixtures" / "verdicts_before_604.json"

# Accuracies a base run might score: around the kill bar (0.3256 - 0.05 = 0.2756),
# around the keep bar (0.3256), 001's real 0.2036, chance, and past SmolLM2's 0.4289.
BASE_GRID = (0.0, 0.2036, 0.2755, 0.2756, 0.2757, 0.26, 0.30, 0.3255, 0.3256, 0.3257, 0.4289)


def _judged(spec, results) -> dict:
    """Every number a verdict carries, by field name, as JSON can hold it."""
    try:
        verdict = evaluate(spec, results)
    except (KeyError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "outcome": verdict.outcome,
        "reason": verdict.reason,
        "describe": verdict.describe(),
        "criteria": {
            c.criterion.name: {
                "holds": c.holds,
                "points": [{"point": p.point, "sigmas": p.sigmas, "holds": p.holds, "delta": p.delta,
                            "p": p.p, "seeds": list(p.seeds)} for p in c.points],
            } for c in verdict.criteria},
    }


def _base_results(path, accuracy) -> dict | None:
    """The results `instruments.base_run.verdict_for` builds: the run's one score
    beside the constant reference. None for a spec that is not a base run."""
    file = SpecFile.load(path)
    runnable, constants = list(file.runnable_arms()), file.constant_arms()
    if path.parent.parent.name != "base" or len(runnable) != 1 or len(constants) != 1:
        return None
    results = load_recorded_results(path)
    results.setdefault("run", {})[runnable[0]] = [float(accuracy)]
    return results


def judge_all(path) -> dict:
    spec = load_spec(path)
    entry = {"recorded": _judged(spec, load_recorded_results(path))}
    if _base_results(path, 0.0) is not None:
        entry["base_grid"] = {repr(a): _judged(spec, _base_results(path, a)) for a in BASE_GRID}
    return entry


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin() -> None:
    PIN.write_text(json.dumps({p.relative_to(ROOT).as_posix(): {"sha256": _sha(p), **judge_all(p)}
                               for p in SPECS}, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_committed_spec_parses_through_the_referee(path):
    spec = load_spec(path)
    assert spec.criteria and spec.file is not None
    load_recorded_results(path)


PINNED = json.loads(PIN.read_text()) if PIN.exists() else {}


def test_the_pin_covers_the_live_base_run():
    assert "experiments/base/specs/002-smol-mix-base.toml" in PINNED
    assert len(PINNED["experiments/base/specs/002-smol-mix-base.toml"]["base_grid"]) == len(BASE_GRID)


@pytest.mark.parametrize("rel", sorted(PINNED), ids=lambda r: pathlib.Path(r).stem)
def test_a_spec_without_stages_is_judged_as_before_604(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} was removed since the pin")
    if _sha(path) != PINNED[rel]["sha256"]:
        pytest.skip(f"{rel} changed since the pin; its pinned verdict no longer describes it")
    assert SpecFile.load(path).stages is None, "the pin is for specs judged at a single look"
    pinned = {k: v for k, v in PINNED[rel].items() if k != "sha256"}
    assert json.loads(json.dumps(judge_all(path))) == pinned


if __name__ == "__main__":
    pin()
