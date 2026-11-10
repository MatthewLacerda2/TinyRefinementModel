"""The #447 replay measures the clip and nothing else.

Its whole claim rests on two properties: fed identical gradients, a clip that never
binds makes the two optimizers agree *exactly* — otherwise the harness is comparing
different inputs and every number it prints is contaminated — and the norm profile
it replays is fixed once the source run has passed the horizon, since that source
is usually a run still writing its metrics.csv.
"""

import json

import numpy as np

from experiments.recipe import clip_under_muon

SEQ = 16


def _world(tmp_path, rows):
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "chunk_0.npy", np.random.default_rng(0).integers(0, 50000, 4000, dtype=np.int32))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,applied_grad_norm\n" + "".join(f"{s},{n}\n" for s, n in rows))
    return data, metrics


def _results(capsys):
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")]
    return {row["point"]: row["value"] for row in (json.loads(ln[7:]) for ln in lines)}


def test_a_clip_that_never_binds_leaves_both_partitions_exactly_alone(tmp_path, capsys):
    data, metrics = _world(tmp_path, [(0, 4.0), (10, 2.0), (20, 1.5), (29, 0.8)])
    clip_under_muon.main(["--metrics", str(metrics), "--data", str(data), "--steps", "30",
                          "--dim", "32", "--layers", "1", "--vocab", "256", "--seq", str(SEQ),
                          "--clip", "1e9"])
    values = _results(capsys)
    assert values, "the harness printed no RESULT lines"
    assert all(value == 1.0 for value in values.values()), values


def test_a_binding_clip_is_seen(tmp_path, capsys):
    """The other side of the same property: with norms 2-4x the ceiling, the clipped
    optimizer's history differs, and the harness has to be able to say so."""
    data, metrics = _world(tmp_path, [(0, 4.0), (10, 2.0), (20, 3.0), (29, 1.5)])
    clip_under_muon.main(["--metrics", str(metrics), "--data", str(data), "--steps", "30",
                          "--dim", "32", "--layers", "1", "--vocab", "256", "--seq", str(SEQ),
                          "--clip", "1.0"])
    values = _results(capsys)
    assert min(v for k, v in values.items() if k.startswith(("muon_cos", "adam_cos"))) < 1.0


def test_rows_logged_after_the_horizon_do_not_move_the_replay(tmp_path):
    rows = [(s, 3.0 - s / 20) for s in range(0, 40, 5)]
    _, before = _world(tmp_path, rows)
    early = clip_under_muon.norm_profile(before, 30, seed=0)
    before.write_text(before.read_text() + "40,9.0\n45,0.1\n50,7.0\n")
    assert np.array_equal(early, clip_under_muon.norm_profile(before, 30, seed=0))
