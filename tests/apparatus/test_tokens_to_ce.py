"""The recipe pair's metric (#26 stage 2): tokens to a target CE, capped, never a free win."""

from experiments.recipe.tokens_to_ce import last_step, tokens_to_target

HEADER = "step,ce,val_ce\n"


def test_first_probe_at_or_below_target_decides_and_the_cap_saturates(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(HEADER + "5,9.0,\n16,8.0,7.9\n32,7.0,6.4\n48,6.2,5.4\n64,6.0,5.3\n80,5.9,5.1\n")
    tokens_m, final, reached = tokens_to_target(path, 5.5, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, final, reached) == (48.0, 5.3, True), "first probe <= 5.5 is step 48; rows past the cap ignored"
    tokens_m, final, reached = tokens_to_target(path, 4.0, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, reached) == (64.0, False), "never reached: the cap, not a win"
    assert last_step(path) == 80 and last_step(tmp_path / "missing.csv") == 0
