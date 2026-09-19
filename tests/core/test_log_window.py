"""A logging window reports the mean of what it holds, however it started (#194).

Every resume wrote one bogus row: CE 1.87 beside a real ~3.2, and depth_avg 2.76
on a uniform 1–8 draw — both scaled by ~0.6, because the first window after a
resume held fewer micro-steps than the nominal count it was divided by. The
bogus CE also entered the plateau detector's running minimum, so the damage
outlived the row.
"""

import inspect

from trm.train import trainer
from trm.train.trainer import LogWindow


def _logged_rows(start_step, accum, log_real, value_at):
    """Replay the trainer's gating from `start_step`, returning each logged mean."""
    window, rows = LogWindow(), []
    for step in range(start_step, start_step + 3 * accum * log_real):
        v = value_at(step)
        window.add(loss=v, token_loss=v, grad_norm=v, depth=v)
        if (step + 1) % (accum * log_real) == 0:
            rows.append(window.means())
            window.reset()
    return rows


def test_a_resume_off_a_window_boundary_logs_true_means():
    """The exact #157 shape: steady CE, resumed mid-window. Every row, the first
    included, must read the steady value — not a fraction of it."""
    for start in (1, 7, 13, 19):
        rows = _logged_rows(start, accum=4, log_real=5, value_at=lambda s: 3.2)
        assert rows and all(abs(r[1] - 3.2) < 1e-12 for r in rows), (start, rows[0])


def test_the_mean_is_over_the_steps_actually_held():
    window = LogWindow()
    for depth in (1, 8, 3):
        window.add(loss=0.0, token_loss=0.0, grad_norm=0.0, depth=depth)
    assert window.means()[3] == 4.0


def test_the_trainer_logs_through_the_window_not_a_nominal_divisor():
    """The replay above is only evidence if train_loop uses the same object."""
    source = inspect.getsource(trainer.train_loop)
    assert "window.means()" in source and "window.add(" in source and "window.reset()" in source
    assert "/ divisor" not in source, "a nominal divisor is the bug"


def test_a_skipped_micro_step_does_not_advance_the_step():
    """#355: the optimizer never counts a non-finite micro-step. If the trainer's `step`
    advanced past one, every boundary keyed on it (logging, the probe, checkpoints, the
    applied-gradient telemetry) drifted off the optimizer's window by one micro-step per
    skip. The branch that skips must therefore leave `step` alone."""
    import ast

    tree = ast.parse(inspect.getsource(trainer.train_loop))
    skip_branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and ast.unparse(node.test).startswith("not (math.isfinite(current_loss)")]
    assert len(skip_branches) == 1, "the non-finite branch moved; point this test at it"
    advances = [node for node in ast.walk(skip_branches[0]) if isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name) and node.target.id == "step"]
    assert not advances, "a skipped micro-step must not advance `step`"
