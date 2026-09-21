"""The #460 held-out profile reports what the mean CE averages away, so each of its
readings has to be exactly what it claims: bits per byte by hand, position buckets
that partition the targets, a tail that is a tail, a calibration error that reads 0
for a calibrated model, and per-token losses whose mean is the trainer's val CE."""

import math

import numpy as np

from instruments import heldout_profile


def _flat(n, rng):
    return dict(nll=rng.exponential(2.0, n), top_prob=rng.uniform(0, 1, n), correct=rng.random(n) < 0.3,
                context_len=rng.integers(1, 513, n), nbytes=rng.integers(1, 8, n))


def test_bits_per_byte_by_hand():
    out = heldout_profile.profile(nll=[1.0, 2.0, 3.0], top_prob=[0.5] * 3, correct=[1, 0, 0],
                                  context_len=[1, 2, 3], nbytes=[1, 2, 3])
    assert math.isclose(out["bpb"], 6.0 / math.log(2) / 6)
    assert math.isclose(out["ce"], 2.0)


def test_position_buckets_partition_the_window():
    lengths = np.arange(1, 513)
    out = heldout_profile.profile(nll=lengths.astype(float), top_prob=np.full(512, 0.5),
                                  correct=np.zeros(512), context_len=lengths, nbytes=np.ones(512))
    assert list(out["ce_by_position"]) == ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127",
                                           "128-255", "256-511", "512"]
    # Each bucket's mean is the mean of the lengths it holds: every target in exactly one.
    assert out["ce_by_position"]["4-7"] == 5.5 and out["ce_by_position"]["512"] == 512.0


def test_labels_do_not_depend_on_which_lengths_survive_the_mask():
    kept = np.array([1, 300, 300])  # nothing between 256 and 511 but 300, no 512 at all
    out = heldout_profile.profile(nll=[1.0, 2.0, 2.0], top_prob=[0.5] * 3, correct=[0] * 3,
                                  context_len=kept, nbytes=[1] * 3, window=512)
    assert list(out["ce_by_position"]) == ["1", "256-511"]


def test_every_checkpoint_dir_of_a_run_writes_to_the_run():
    run = heldout_profile.pathlib.Path("/r/run_x")
    for sub in ("checkpoints", "checkpoints/milestones", "checkpoints/best_val_ce"):
        assert heldout_profile.run_dir(f"/r/run_x/{sub}") == run


def test_the_tail_is_a_tail():
    out = heldout_profile.profile(**_flat(20_000, np.random.default_rng(0)))
    assert out["worst_1%_ce"] >= out["worst_10%_ce"] >= out["ce"]
    assert out["p99_ce"] >= out["p90_ce"] >= out["p50_ce"]


def test_a_calibrated_model_reads_zero_and_an_overconfident_one_does_not():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0, 1, 200_000)
    base = _flat(confidence.size, rng)
    calibrated = heldout_profile.profile(**{**base, "top_prob": confidence,
                                            "correct": rng.random(confidence.size) < confidence})
    overconfident = heldout_profile.profile(**{**base, "top_prob": np.full(confidence.size, 0.9),
                                               "correct": rng.random(confidence.size) < 0.5})
    assert calibrated["ece"] < 0.01
    assert math.isclose(overconfident["ece"], 0.4, abs_tol=0.01)


def test_per_token_losses_average_to_the_trainers_val_ce():
    """The profile scores the same rows the trainer's probe does; if its mean drifted
    from the probe's number, every reading beside it would be about other targets."""
    from flax import nnx

    from trm.config import EOT_TOKEN_ID, MAX_SEQ_LEN
    from trm.model import build_model
    from trm.train.validation import ValidationProbe

    rng = np.random.default_rng(2)
    rows = [rng.integers(0, 300, (1, 2 * MAX_SEQ_LEN + 1)).astype(np.int32) for _ in range(2)]
    rows[0][0, 7] = EOT_TOKEN_ID  # masked out of the targets on both paths (#373)
    model = build_model("plain", 32, nnx.Rngs(0), vocab_size=50304, num_heads=2, num_layers=1)
    probe = ValidationProbe("unused")
    probe._batches = rows
    scored = heldout_profile.score_rows(model, rows)
    assert scored["nll"].size == 2 * 2 * MAX_SEQ_LEN - 1
    assert math.isclose(float(scored["nll"].mean()), probe.run(model), rel_tol=1e-5)
    assert scored["context_len"].min() == 1 and scored["context_len"].max() == MAX_SEQ_LEN
