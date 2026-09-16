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


def test_a_quantity_the_arch_does_not_have_means_none_not_zero():
    """Plain has no depth dial (#316): its depth is None every micro-step, and the
    window's mean is None — a blank depth_avg cell, never a 0.0 that reads as a draw."""
    window = LogWindow()
    for loss in (3.0, 5.0):
        window.add(loss=loss, token_loss=loss, grad_norm=1.0, depth=None)
    assert window.means() == (4.0, 4.0, 1.0, None)
    window.reset()
    window.add(loss=1.0, token_loss=1.0, grad_norm=1.0, depth=2)
    assert window.means()[3] == 2.0, "reset forgets the None"


def test_the_trainer_logs_through_the_window_not_a_nominal_divisor():
    """The replay above is only evidence if train_loop uses the same object."""
    source = inspect.getsource(trainer.train_loop)
    assert "window.means()" in source and "window.add(" in source and "window.reset()" in source
    assert "/ divisor" not in source, "a nominal divisor is the bug"
