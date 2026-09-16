"""Milestone checkpoints: kept regardless of recency, so a bad stretch cannot evict
the checkpoint recovery needs (#187)."""

import ast
import inspect
import json

import orbax.checkpoint as ocp
import jax.numpy as jnp

from trm.runtime.checkpoints import make_milestone_manager, milestone_due
from trm.runtime.layout import MILESTONE_SUBDIR


def test_one_milestone_per_crossed_multiple_on_the_first_boundary_past_it():
    every, tokens_per_step, milestone = 64, 131_072, 500_000_000
    run_steps = 30_518  # the 4B run
    due = [s for s in range(every, run_steps + 1, every) if milestone_due(s, every, tokens_per_step, milestone)]
    assert len(due) == (run_steps // every * every * tokens_per_step) // milestone == 7
    for s in due:
        assert (s * tokens_per_step) // milestone > ((s - every) * tokens_per_step) // milestone


def test_milestones_can_be_turned_off():
    assert not any(milestone_due(s, 64, 131_072, 0) for s in range(64, 10_000, 64))


def test_nothing_evicts_a_milestone(tmp_path):
    """The rolling manager keeps 3 by recency; the milestone manager keeps all."""
    mngr = make_milestone_manager(tmp_path)
    for step in (100, 200, 300, 400, 500):
        mngr.save(step, args=ocp.args.Composite(
            model=ocp.args.StandardSave({"w": jnp.array(float(step))}),
            optimizer=ocp.args.StandardSave({"w": jnp.array(0.0)}),
            monitor_state=ocp.args.JsonSave({}),
            step=ocp.args.JsonSave(step)))
        mngr.wait_until_finished()
    assert list(mngr.all_steps()) == [100, 200, 300, 400, 500]
    assert (tmp_path / MILESTONE_SUBDIR).is_dir()


def test_the_trainer_saves_a_milestone_at_the_rolling_boundary_when_one_is_due():
    from trm.train import trainer
    tree = ast.parse(inspect.getsource(trainer.train_loop))
    saves = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "save_checkpoint"
             and getattr(c.args[0], "id", None) == "milestone_mngr"]
    assert len(saves) == 1
    enclosing = [n for n in ast.walk(tree) if isinstance(n, ast.If) and saves[0] in list(ast.walk(n))]
    tests = " ".join(ast.unparse(n.test) for n in enclosing)
    assert "CHECKPOINT_EVERY_OPT_STEPS" in tests and "milestone_due" in tests


def test_rewind_sets_milestones_aside_too(tmp_path):
    """A milestone newer than the resume point would make orbax refuse the resumed
    run's milestone saves; it is moved out of the way, not deleted."""
    from trm.runtime import rewind as rw

    def ckpt(directory, opt):
        path = directory / str(opt * 128 - 1)
        (path / "monitor_state").mkdir(parents=True)
        (path / "monitor_state" / "metadata").write_text(json.dumps({"sft_active": False}))
        (path / "_CHECKPOINT_METADATA").write_text("{}")

    for opt in (4928, 4992, 5056):
        ckpt(tmp_path, opt)
    ckpt(tmp_path / MILESTONE_SUBDIR, 3840)
    ckpt(tmp_path / MILESTONE_SUBDIR, 5056)
    _, moved = rw.rewind(tmp_path, 4992, 128)
    assert [c.opt_step for c in rw.checkpoints_in(tmp_path / MILESTONE_SUBDIR, 128)] == [3840]
    assert sum(p.parent.name == MILESTONE_SUBDIR for p in moved) == 1 and all(p.exists() for p in moved)


def test_the_disk_guard_budgets_for_all_three_tiers_writing_at_once():
    """Rolling, best and milestone can all land on one boundary; a floor sized for
    two would let the third write tear."""
    from trm.runtime.supervisor import CHECKPOINTS_PER_WRITE, KILLED_DISK, Limits, Observation, State, decide
    assert CHECKPOINTS_PER_WRITE == 3
    obs = Observation(step=5000, ce=3.4, plateau_detected=False, alive=True, free_gb=6.0, checkpoint_gb=1.7)
    assert decide(obs, Limits(stop_step=30_000, disk_margin_gb=1.0), State()).outcome == KILLED_DISK
