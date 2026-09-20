"""Milestone checkpoints: kept regardless of recency, so a bad stretch cannot evict
the checkpoint recovery needs (#187); spaced by doubling and weights-only, so keeping
every one of them costs gigabytes, not tens of them (#394)."""

import ast
import inspect

import orbax.checkpoint as ocp
import jax.numpy as jnp

from trm.runtime.checkpoints import make_milestone_manager, milestone_due, milestone_thresholds
from trm.runtime.layout import (MILESTONE_FIRST_TOKENS, MILESTONE_MAX_COUNT, MILESTONE_RATIO,
                                MILESTONE_SUBDIR)

TOKENS_PER_OPT_STEP = 131_072


def crossings(marks, run_opt_steps, tokens_per_opt_step=TOKENS_PER_OPT_STEP):
    return [s for s in range(1, run_opt_steps + 1)
            if milestone_due(s, s - 1, tokens_per_opt_step, marks)]


def test_the_schedule_doubles_and_stops_at_the_cap():
    marks = milestone_thresholds()
    assert marks[0] == MILESTONE_FIRST_TOKENS and len(marks) == MILESTONE_MAX_COUNT
    assert all(b == int(a * MILESTONE_RATIO) for a, b in zip(marks, marks[1:]))


def test_disk_grows_with_the_logarithm_of_the_run():
    """The point of the doubling (#394): ten times the run buys three more saves,
    where the old fixed 500M cadence bought ten times as many."""
    marks = milestone_thresholds()
    kept = lambda tokens: sum(1 for m in marks if m <= tokens)  # noqa: E731
    assert kept(1_000_000_000) == 7      # 8M..512M
    assert kept(10_000_000_000) == 11
    assert kept(100_000_000_000) == 14


def test_a_milestone_lands_once_per_threshold_and_at_the_step_that_crosses_it():
    marks = milestone_thresholds()
    due = crossings(marks, 30_518)  # the 4B run
    assert len(due) == sum(1 for m in marks if m <= 30_518 * TOKENS_PER_OPT_STEP) == 9
    for step, mark in zip(due, marks):
        assert (step - 1) * TOKENS_PER_OPT_STEP < mark <= step * TOKENS_PER_OPT_STEP


def test_the_early_ones_are_denser_than_the_rolling_cadence():
    """Why the trainer checks every optimizer step: the first milestones fall
    between rolling boundaries, and one that waited for the next boundary would
    not be the point in training it claims to be."""
    from trm.runtime.layout import CHECKPOINT_EVERY_OPT_STEPS
    first = milestone_thresholds()[0] // TOKENS_PER_OPT_STEP
    assert first < CHECKPOINT_EVERY_OPT_STEPS


def test_a_short_pair_still_keeps_a_few():
    """The other half of #394's complaint: at 500M tokens a 67M-token pair kept none."""
    assert len(crossings(milestone_thresholds(), 512)) >= 3


def test_milestones_can_be_turned_off():
    assert milestone_thresholds(first=0) == ()
    assert not any(milestone_due(s, s - 1, TOKENS_PER_OPT_STEP, ()) for s in range(1, 10_000))


def test_nothing_evicts_a_milestone(tmp_path):
    """The rolling manager keeps 3 by recency; the milestone manager keeps all."""
    mngr = make_milestone_manager(tmp_path)
    for step in (100, 200, 300, 400, 500):
        mngr.save(step, args=ocp.args.Composite(
            model=ocp.args.StandardSave({"w": jnp.array(float(step))}),
            monitor_state=ocp.args.JsonSave({}),
            step=ocp.args.JsonSave(step)))
        mngr.wait_until_finished()
    assert list(mngr.all_steps()) == [100, 200, 300, 400, 500]
    assert (tmp_path / MILESTONE_SUBDIR).is_dir()


def test_a_saved_milestone_holds_weights_and_no_optimizer(tmp_path):
    """What makes it ~4x smaller, and what makes it not a resume point."""
    mngr = make_milestone_manager(tmp_path)
    mngr.save(100, args=ocp.args.Composite(
        model=ocp.args.StandardSave({"w": jnp.array(1.0)}),
        monitor_state=ocp.args.JsonSave({"run_id": "r"}),
        step=ocp.args.JsonSave(100)))
    mngr.wait_until_finished()
    written = {p.name for p in (tmp_path / MILESTONE_SUBDIR / "100").iterdir()}
    assert "model" in written and "optimizer" not in written


def test_the_trainer_saves_a_milestone_every_optimizer_step_it_is_due():
    from trm.train import trainer
    tree = ast.parse(inspect.getsource(trainer.train_loop))
    saves = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "save_milestone"]
    assert len(saves) == 1
    enclosing = [n for n in ast.walk(tree) if isinstance(n, ast.If) and saves[0] in list(ast.walk(n))]
    tests = " ".join(ast.unparse(n.test) for n in enclosing)
    assert "milestone_due" in tests and "CHECKPOINT_EVERY_OPT_STEPS" not in tests
    # and never the full-state save into the milestone manager
    assert not [c for c in ast.walk(tree) if isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "save_checkpoint"
                and getattr(c.args[0], "id", None) == "milestone_mngr"]


def test_rewind_sets_milestones_aside_too(tmp_path):
    """A milestone newer than the resume point would make orbax refuse the resumed
    run's milestone saves; it is moved out of the way, not deleted."""
    from trm.runtime import rewind as rw

    def ckpt(directory, opt):
        path = directory / str(opt * 128 - 1)
        path.mkdir(parents=True)
        (path / "_CHECKPOINT_METADATA").write_text("{}")

    for opt in (4928, 4992, 5056):
        ckpt(tmp_path, opt)
    ckpt(tmp_path / MILESTONE_SUBDIR, 3840)
    ckpt(tmp_path / MILESTONE_SUBDIR, 5056)
    _, moved = rw.rewind(tmp_path, 4992, 128)
    assert [c.opt_step for c in rw.checkpoints_in(tmp_path / MILESTONE_SUBDIR, 128)] == [3840]
    assert sum(p.parent.name == MILESTONE_SUBDIR for p in moved) == 1 and all(p.exists() for p in moved)


def test_rewind_says_a_weights_only_checkpoint_is_not_a_resume_point(tmp_path):
    from trm.runtime import rewind as rw

    full, weights = tmp_path / "639", tmp_path / MILESTONE_SUBDIR / "639"
    for path in (full, weights):
        (path / "model").mkdir(parents=True)
        (path / "_CHECKPOINT_METADATA").write_text("{}")
    (full / "optimizer").mkdir()
    assert "not a resume point" not in rw.checkpoints_in(tmp_path, 128)[0].describe()
    assert "not a resume point" in rw.checkpoints_in(tmp_path / MILESTONE_SUBDIR, 128)[0].describe()


def test_the_disk_guard_budgets_for_all_three_tiers_writing_at_once():
    """Rolling, best and milestone can all land on one boundary; a floor sized for
    two would let the third write tear."""
    from trm.runtime.supervisor import CHECKPOINTS_PER_WRITE, KILLED_DISK, Limits, Observation, State, decide
    assert CHECKPOINTS_PER_WRITE == 3
    obs = Observation(step=5000, ce=3.4, alive=True, free_gb=6.0, checkpoint_gb=1.7)
    assert decide(obs, Limits(stop_step=30_000, disk_margin_gb=1.0), State()).outcome == KILLED_DISK
