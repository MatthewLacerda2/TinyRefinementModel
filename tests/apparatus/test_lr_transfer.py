"""The #483 judge: where an LR optimum sits on a grid, and whether it moves with width.

Each case is a grid of tokens-to-target numbers built to sit on one side of the
pre-registered bar (experiments/recipe/lr_transfer.py), so a change to the rule
shows up as a case changing sides.
"""

import pytest

from experiments.recipe import lr_transfer
from experiments.recipe.lr_transfer import FAILS, HOLDS, INCONCLUSIVE, arm_name, grid_from_arms, judge

GRID = (1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3)


def bowl(best, depth=10.0, spread=0.5):
    """Three seeds per LR, lowest at `best`, rising `depth` per grid step away."""
    return {lr: [depth * abs(lr_transfer.grid_steps(lr, best)) + 60 + d for d in (-spread, 0, spread)]
            for lr in GRID}


def test_the_same_optimum_at_every_width_holds():
    verdict, optima, _ = judge({192: bowl(6e-4), 384: bowl(6e-4), 576: bowl(6e-4)})
    assert verdict == HOLDS
    assert [o.best for o in optima] == [6e-4] * 3


def test_one_grid_step_away_still_holds():
    assert judge({192: bowl(1.2e-3), 576: bowl(6e-4)})[0] == HOLDS


def test_two_grid_steps_away_fails():
    verdict, _, why = judge({192: bowl(3e-4), 384: bowl(6e-4), 576: bowl(1.2e-3)})
    assert verdict == FAILS
    assert "w192" in why and "w384" not in why, "only the width two steps off fails"


def test_noise_widens_the_optimum_rather_than_picking_a_lucky_argmin():
    """A neighbour one step off the best, inside 2 sigma, is part of the optimum:
    the narrow width's argmin may be two steps from full width's and still transfer."""
    narrow = bowl(3e-4, depth=10.0)
    narrow[6e-4] = [v - 9.5 for v in narrow[6e-4]]    # 6e-4 within noise of the best
    full = bowl(1.2e-3)
    assert judge({192: narrow, 576: full})[0] == HOLDS


def test_an_optimum_on_the_grid_edge_is_not_located():
    verdict, _, why = judge({192: bowl(1.5e-4), 576: bowl(6e-4)})
    assert verdict == INCONCLUSIVE and "edge" in why


def test_a_sweep_too_noisy_to_place_the_optimum_is_not_a_pass():
    """Rule 1: two flat, noisy curves overlap everywhere. That is no optimum, not transfer."""
    flat = {lr: [60.0, 80.0, 100.0] for lr in GRID}
    flat[6e-4] = [59.0, 79.0, 99.0]
    verdict, _, why = judge({192: flat, 576: flat})
    assert verdict == INCONCLUSIVE and "cannot be told apart" in why


def test_arm_names_parse_and_a_stray_arm_is_refused():
    assert [arm_name(576, lr) for lr in GRID] == [
        "w576_lr15e-5", "w576_lr3e-4", "w576_lr6e-4", "w576_lr12e-4", "w576_lr24e-4"]
    grid = grid_from_arms({arm_name(192, 6e-4): [1.0, 2.0], arm_name(576, 1.5e-4): [3.0, 4.0]})
    assert grid == {192: {6e-4: [1.0, 2.0]}, 576: {1.5e-4: [3.0, 4.0]}}
    with pytest.raises(ValueError, match="w<width>_lr<lr>"):
        grid_from_arms({"control": [1.0]})


def test_the_cli_judges_a_specs_recorded_results(tmp_path, capsys):
    """The sweep's spec carries per-seed tokens-to-target under [results.run], as the
    runner records them; the judge reads that table and nothing else."""
    rows = [f'{arm_name(w, lr)} = [{", ".join(f"{v:g}" for v in bowl(6e-4)[lr])}]'
            for w in (192, 576) for lr in GRID]
    spec = tmp_path / "483.toml"
    spec.write_text("[results.run]\n" + "\n".join(rows) + "\n")
    assert lr_transfer.main([str(spec)]) == 0
    out = capsys.readouterr().out
    assert "HOLDS" in out and 'RESULT {"fails": 0.0, "holds": 1.0, "point": "transfer"}' in out
