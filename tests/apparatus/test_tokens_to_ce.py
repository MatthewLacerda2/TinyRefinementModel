"""The recipe pair's metric (#26 stage 2): tokens to a target CE, capped, never a free win."""

from experiments.recipe.tokens_to_ce import last_step, tokens_to_target

HEADER = "step,ce,val_ce\n"


def test_first_probe_at_or_below_target_decides_and_the_cap_saturates(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(HEADER + "5,9.0,\n16,8.0,7.9\n32,7.0,6.4\n48,6.2,5.4\n64,6.0,5.3\n80,5.9,5.1\n")
    tokens_m, final, reached, aligned, interp_m = tokens_to_target(path, 5.5, cap_steps=64,
                                                                   tokens_per_opt_step=1_000_000)
    assert (tokens_m, final, reached) == (48.0, 5.3, True), "first probe <= 5.5 is step 48; rows past the cap ignored"
    assert abs(interp_m - 46.4) < 1e-9, "6.4 at 32 and 5.4 at 48 put 5.5 at 32 + 16 x 0.9 (#547)"
    assert not aligned, "a run without val_step is read at the logged row, and says so"
    tokens_m, final, reached, _, interp_m = tokens_to_target(path, 4.0, cap_steps=64, tokens_per_opt_step=1_000_000)
    assert (tokens_m, reached, interp_m) == (64.0, False, 64.0), "never reached: the cap, not a win"
    assert last_step(path) == 80 and last_step(tmp_path / "missing.csv") == 0


def test_a_run_that_recorded_val_step_is_read_at_the_probe(tmp_path):
    """#351: a probe at opt step 48 lands on row 50. With val_step recorded, the
    tokens are counted at 48, where the weights were measured, not at 50."""
    path = tmp_path / "metrics.csv"
    path.write_text("step,ce,val_ce,val_step\n45,6.3,,\n50,6.2,5.4,48\n55,6.1,,\n")
    tokens_m, final, reached, aligned, interp_m = tokens_to_target(path, 5.5, cap_steps=64,
                                                                   tokens_per_opt_step=1_000_000)
    assert (tokens_m, final, reached, aligned, interp_m) == (48.0, 5.4, True, True, 48.0), \
        "the first probe already at or below the target is its own crossing"


def test_minutes_to_target_times_from_the_first_row(tmp_path):
    """#385: wall-clock from the first logged row (after compile) to the row that
    first reached the target; the whole span when it never did."""
    from experiments.recipe.tokens_to_ce import minutes_to_target

    path = tmp_path / "metrics.csv"
    path.write_text("step,ce,val_ce,wall_clock\n"
                    "5,7.0,,2026-09-19T10:00:00Z\n"
                    "10,6.0,5.9,2026-09-19T10:02:00Z\n"
                    "15,5.8,5.7,2026-09-19T10:04:30Z\n")
    assert minutes_to_target(path, 5.85, cap_steps=64) == 4.5
    assert minutes_to_target(path, 4.0, cap_steps=64) == 4.5, "never reached: the whole span"
    assert minutes_to_target(path, 5.95, cap_steps=64) == 2.0


def test_set_passes_a_config_knob_and_refuses_an_unknown_one():
    """--set reaches the trainer's env only for a knob, a trm.settings.Config field (#359):
    a typo would otherwise train an arm identical to its control."""
    import pytest

    from experiments.recipe.tokens_to_ce import parse_knobs

    assert parse_knobs(["ADAM_B2=0.95", "WEIGHT_DECAY=0.1"]) == {"ADAM_B2": "0.95", "WEIGHT_DECAY": "0.1"}
    for bad in (["ADAM_BETA2=0.95"], ["adam_b2=0.95"], ["ADAM_B2"]):
        with pytest.raises(SystemExit):
            parse_knobs(bad)


def test_an_arm_can_set_its_mixture():
    """#439: the mixture is a knob a spec pins per arm. `--set` refuses any name
    that is not a Config field, so a mixture knob that lived anywhere else could
    not be put on an arm at all — and the `=` inside the value must survive."""
    from experiments.recipe.tokens_to_ce import parse_knobs

    knobs = parse_knobs(["DATA_MIXTURE=pretrain/fineweb-edu=0.9:0.6,pretrain/finemath=0.1:0.4",
                         "MIXTURE_RAMP_FRACTION=0.5"])
    assert knobs == {"DATA_MIXTURE": "pretrain/fineweb-edu=0.9:0.6,pretrain/finemath=0.1:0.4",
                     "MIXTURE_RAMP_FRACTION": "0.5"}


def _ckpt(directory, step):
    path = directory / str(step)
    path.mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("{}")


def test_an_interrupted_arm_sets_aside_best_checkpoints_newer_than_its_resume_point(tmp_path):
    """#554: stopped at opt step 272 with the rolling checkpoint at 256, best_val_ce held
    264 and 272, and the relaunched trainer died at its first new best."""
    from experiments.recipe.tokens_to_ce import prepare_resume

    ckpts = tmp_path / "run" / "checkpoints"
    _ckpt(ckpts, 16383)                        # opt 256 at 64 micro-steps per opt step
    for step in (16383, 16895, 17407):         # opt 256, 264, 272
        _ckpt(ckpts / "best_val_ce", step)
    note = prepare_resume(tmp_path / "run", accumulation_steps=64)
    assert "opt step 256" in note and "2 newer" in note
    assert sorted(p.name for p in (ckpts / "best_val_ce").iterdir()) == ["16383"]
    assert prepare_resume(tmp_path / "run", accumulation_steps=64) is None, "a clean arm is left alone"


def test_an_arm_stopped_before_its_first_rolling_checkpoint_starts_over(tmp_path):
    from experiments.recipe.tokens_to_ce import prepare_resume

    run = tmp_path / "run"
    _ckpt(run / "checkpoints" / "best_val_ce", 511)
    (run / "metrics.csv").write_text("step,val_ce\n8,9.0\n")
    note = prepare_resume(run, accumulation_steps=64)
    assert "starting over" in note and not run.exists()
    assert [p.name.startswith("run.set_aside_") for p in tmp_path.iterdir()] == [True]
    assert prepare_resume(tmp_path / "never_ran", accumulation_steps=64) is None
