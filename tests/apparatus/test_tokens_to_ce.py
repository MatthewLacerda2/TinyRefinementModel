"""The recipe pair's metric (#26 stage 2): tokens to a target CE, capped, never a free win."""

from experiments.recipe.tokens_to_ce import last_step, tokens_to_target

HEADER = "step,ce,val_ce\n"


def test_first_probe_at_or_below_target_decides_and_the_cap_saturates(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(HEADER + "5,9.0,\n16,8.0,7.9\n32,7.0,6.4\n48,6.2,5.4\n64,6.0,5.3\n80,5.9,5.1\n")
    tokens_m, final, reached, aligned = tokens_to_target(path, 5.5, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, final, reached) == (48.0, 5.3, True), "first probe <= 5.5 is step 48; rows past the cap ignored"
    assert not aligned, "a run without val_step is read at the logged row, and says so"
    tokens_m, final, reached, _ = tokens_to_target(path, 4.0, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, reached) == (64.0, False), "never reached: the cap, not a win"
    assert last_step(path) == 80 and last_step(tmp_path / "missing.csv") == 0


def test_a_run_that_recorded_val_step_is_read_at_the_probe(tmp_path):
    """#351: a probe at opt step 48 lands on row 50. With val_step recorded, the
    tokens are counted at 48, where the weights were measured, not at 50."""
    path = tmp_path / "metrics.csv"
    path.write_text("step,ce,val_ce,val_step\n45,6.3,,\n50,6.2,5.4,48\n55,6.1,,\n")
    tokens_m, final, reached, aligned = tokens_to_target(path, 5.5, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, final, reached, aligned) == (48.0, 5.4, True, True)
