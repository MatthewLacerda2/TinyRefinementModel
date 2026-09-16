"""The trajectory readouts compute what they claim (#225).

`tests/core/test_trajectory_capture.py` guards the *capture* — that the states
coming back are the ones the model actually passed through. This file guards the
*arithmetic* on top of them, against synthetic trajectories whose answers are
known by construction, because a readout that is subtly wrong is worse than none:
it produces a plausible number that a research conclusion then rests on.

The non-finite tests are not defensive padding. A checkpoint 3.20B tokens into
the base run returns all-NaN logits on some code documents (#229), and one
out-of-vocab token id poisons a whole window (#233). A mean that silently
swallows those is exactly how a bad number becomes a finding.
"""

import numpy as np
import pytest

from instruments.latents import Trajectory
from instruments.results import parse


def _traj(states, gates=None):
    states = np.asarray(states, dtype=np.float32)
    return Trajectory(states=states, gates=gates, depth=states.shape[0] - 1)


def _line(points):
    """A trajectory walking a straight line in unit steps along one axis."""
    return _traj([[[[float(k), 0.0]]] for k in range(points)])


# --- step size ---------------------------------------------------------------

def test_step_size_measures_the_distance_each_pass_moved():
    traj = _traj([[[[0.0, 0.0]]], [[[3.0, 4.0]]], [[[3.0, 4.0]]]])
    assert np.allclose(traj.step_sizes(), [5.0, 0.0]), "3-4-5, then a pass that did nothing"


def test_step_size_has_one_entry_per_pass_not_per_state():
    traj = _line(5)                      # 5 states = 1 origin + 4 passes
    assert traj.step_sizes().shape == (4,)


# --- turning angle -----------------------------------------------------------

def test_walking_a_straight_line_turns_by_nothing():
    """+1 means the passes are one long stride chopped up — the reading that
    would suggest the same displacement is reachable in fewer passes."""
    assert np.allclose(_line(4).turning_angles(), [1.0, 1.0])


def test_a_pass_that_undoes_the_last_one_reads_as_a_reversal():
    traj = _traj([[[[0.0, 0.0]]], [[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    assert np.allclose(traj.turning_angles(), [-1.0])


def test_unrelated_passes_read_as_a_right_angle():
    traj = _traj([[[[0.0, 0.0]]], [[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    assert np.allclose(traj.turning_angles(), [0.0], atol=1e-6)


def test_a_zero_length_step_does_not_divide_by_zero():
    """A converged loop stops moving; the readout must survive its own success."""
    traj = _traj([[[[0.0, 0.0]]], [[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    assert np.isfinite(traj.turning_angles()).all()


# --- distance to final -------------------------------------------------------

def test_distance_to_final_is_zero_at_the_end_and_counts_down():
    d = _line(4).distance_to_final()
    assert d[-1] == 0.0
    assert np.all(np.diff(d) < 0), "each pass should end nearer the destination"


# --- non-finite refusal ------------------------------------------------------

@pytest.mark.parametrize("readout", ["step_sizes", "turning_angles", "distance_to_final"])
def test_every_readout_refuses_a_trajectory_that_blew_up(readout):
    traj = Trajectory(states=np.array([[[[0.0, np.nan]]], [[[1.0, 2.0]]]], dtype=np.float32),
                      gates=None, depth=1, nonfinite=1)
    assert not traj.ok
    with pytest.raises(ValueError, match="non-finite"):
        getattr(traj, readout)()


def test_a_healthy_trajectory_is_not_refused():
    """The counter-test: a guard that rejected everything would pass every test
    above and be useless."""
    traj = _line(3)
    assert traj.ok
    traj.step_sizes(), traj.turning_angles(), traj.distance_to_final()


# --- the machine-readable protocol -------------------------------------------

def test_results_are_emitted_one_line_per_pass(capsys):
    """`instruments/results.py` is how a spec drives this and the referee reads
    it — the whole reason these numbers can be judged rather than eyeballed."""
    traj = _traj([[[[0.0, 0.0]]], [[[1.0, 0.0]]], [[[2.0, 0.0]]], [[[3.0, 0.0]]]],
                 gates=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    traj.emit_results()

    rows = parse(capsys.readouterr().out)
    assert [r["point"] for r in rows] == ["d1", "d2", "d3"]
    assert all("step_size" in r and "gate_openness" in r for r in rows)
    assert "turning_angle" not in rows[0], "the first pass has no previous step to turn from"
    assert np.isclose(rows[2]["gate_openness"], 0.3)


@pytest.mark.parametrize("arch", ["plain", "reasoner"])
def test_an_arch_without_a_refine_loop_is_refused_before_anything_loads(monkeypatch, arch):
    import instruments.latents as latents
    monkeypatch.setattr("trm.config.MODEL_ARCH", arch)
    monkeypatch.setattr("trm.runtime.restore.restore_model",
                        lambda *a, **k: pytest.fail("restored a model it should have refused"))
    with pytest.raises(SystemExit, match=f"MODEL_ARCH='{arch}'"):
        latents._main(["--checkpoint", "nowhere"])
