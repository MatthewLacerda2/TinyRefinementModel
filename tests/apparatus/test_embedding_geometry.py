"""#464 draws one picture per milestone and claims the movement is the model's, not the
projection's. That rests on three properties: the token kinds are what they say, every
frame goes through the SAME projection, and the separation number only rises when tokens
of a kind actually group together."""

import numpy as np

from instruments import embedding_geometry as eg


def test_kinds_are_what_they_say():
    assert eg.kind_of(" the") == "word"
    assert eg.kind_of("the") == "other"       # no leading space: a continuation subword
    assert eg.kind_of(".") == "punctuation"
    assert eg.kind_of(", ") == "punctuation"
    assert eg.kind_of("    ") == "punctuation"  # whitespace runs, as code indentation is
    assert eg.kind_of("42") == "digits"
    assert eg.kind_of(" 7") == "digits"
    assert eg.kind_of("a1") == "other"        # mixed: not a number
    assert eg.kind_of("):") == "code"
    assert eg.kind_of(" =") == "code"
    assert eg.kind_of("café") == "other"


def test_one_projection_is_shared_by_every_frame():
    """The projection is fitted on the LAST milestone and reused, so a frame that equals
    that milestone lands exactly on its own principal axes, and an earlier frame is drawn
    in the same basis rather than in one of its own."""
    rng = np.random.default_rng(0)
    last = rng.normal(size=(40, 6))
    centre, axes = eg.projection(last)
    assert axes.shape == (2, 6)
    assert np.allclose(axes @ axes.T, np.eye(2), atol=1e-9)
    points = (last - centre) @ axes.T
    spread = points.var(0)
    assert spread[0] > spread[1]  # first direction carries more of the variance
    # An earlier frame, here the same embedding scaled down, must project to the same
    # shape scaled down — not re-fitted to fill the panel.
    early = (0.25 * (last - centre)) + centre
    assert np.allclose((early - centre) @ axes.T, 0.25 * points)


def test_separation_sees_grouping_and_not_noise():
    rng = np.random.default_rng(1)
    kinds = np.array(["word"] * 30 + ["digits"] * 30)
    scattered = rng.normal(size=(60, 8))
    grouped = scattered + np.where(kinds[:, None] == "word", 6.0, -6.0)
    assert abs(eg.readings(scattered, kinds)["separation"]) < 0.1
    assert eg.readings(grouped, kinds)["separation"] > 0.5


def test_anisotropy_and_rms_read_a_known_shape():
    rows = np.zeros((50, 4))
    rows[:, 0] = np.linspace(-1, 1, 50)  # all the variance on one axis
    reading = eg.readings(rows, np.array(["word"] * 50))
    assert reading["anisotropy"] > 0.999
    assert np.isclose(reading["rms"], np.sqrt((rows ** 2).mean()))
    # One kind only: there is no across-kind pair to compare against, and saying 0
    # would claim "measured, no grouping".
    assert np.isnan(reading["separation"])


def test_frequency_counts_come_from_the_shards(tmp_path):
    source = tmp_path / "pretrain" / "fineweb-edu"
    source.mkdir(parents=True)
    np.save(source / "chunk_0.npy", np.array([5] * 10 + [7] * 6 + [9] * 3, dtype=np.int32))
    assert eg.frequent_tokens(str(tmp_path), ["fineweb-edu"], 2, 100).tolist() == [5, 7]
