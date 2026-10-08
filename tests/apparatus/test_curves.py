"""instruments.curves: one reader for a run's held-out curve, and a crossing that is
not quantized to the probe interval (#547)."""

import pytest

from instruments.curves import ValCurve

ROWS = "step,ce,val_ce,val_step\n8,9,7.0,8\n10,8,,\n16,7,6.0,16\n24,6,5.0,24\n32,5,4.5,32\n"


@pytest.fixture
def curve(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(ROWS)
    return ValCurve.read(path)


def test_the_curve_is_the_probes_at_their_own_steps(curve):
    assert curve.steps == (8, 16, 24, 32) and curve.ces == (7.0, 6.0, 5.0, 4.5) and curve.aligned
    assert curve.final == 4.5 and curve.at(20) == 6.0 and curve.at(7) is None


def test_the_crossing_interpolates_between_the_probes_that_bracket_the_target(curve):
    assert curve.first_at_or_below(5.5) == 24, "quantized: the probe interval is its resolution"
    assert curve.crossing(5.5) == 20.0, "6.0 at 16 and 5.0 at 24 put 5.5 halfway"
    assert curve.crossing(7.5) == 8.0, "already below at the first probe: that probe"
    assert curve.crossing(4.0) is None


def test_seeds_inside_one_probe_interval_no_longer_tie(tmp_path):
    """#547: three seeds that all first read below the target at the same probe had
    sigma 0, so any gap read as infinite sigmas. Interpolated, they differ."""
    crossings = set()
    for seed, before in enumerate((5.9, 5.7, 5.6)):
        path = tmp_path / f"s{seed}.csv"
        path.write_text(f"step,ce,val_ce,val_step\n8,1,{before},8\n16,1,5.4,16\n")
        curve = ValCurve.read(path)
        assert curve.first_at_or_below(5.5) == 16
        crossings.add(curve.crossing(5.5))
    assert len(crossings) == 3


def test_a_cap_drops_later_probes(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(ROWS)
    assert ValCurve.read(path, cap_steps=16).steps == (8, 16)
