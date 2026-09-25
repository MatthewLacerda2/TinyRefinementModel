"""#464 draws one picture per milestone and claims the movement is the model's, not the
projection's. That rests on three properties: the token kinds are what they say, every
frame goes through the SAME projection, and the separation number only rises when tokens
of a kind actually group together."""

import pathlib

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


def test_the_none_of_the_above_bucket_is_not_counted_as_a_kind():
    """`other` is defined as what did not match; rows in it have no reason to cohere, and
    counting them as a kind would hand a quarter of the signal to a bucket that means
    nothing. It is excluded from the reading — moving those rows must not move it."""
    rng = np.random.default_rng(2)
    kinds = np.array(["word"] * 20 + ["digits"] * 20 + ["other"] * 20)
    rows = rng.normal(size=(60, 8)) + np.where(kinds[:, None] == "word", 5.0, 0.0)
    moved = rows.copy()
    moved[kinds == "other"] = rng.normal(size=(20, 8)) + 40.0  # pile `other` up somewhere
    assert np.isclose(eg.readings(rows, kinds)["separation"],
                      eg.readings(moved, kinds)["separation"])


def test_each_named_kind_gets_its_own_number():
    """Words are ~76% of the same-kind pairs in the real vocabulary, so the headline is
    mostly about them; a small kind that groups must be visible on its own."""
    rng = np.random.default_rng(3)
    kinds = np.array(["word"] * 40 + ["digits"] * 6 + ["code"] * 6)
    rows = rng.normal(size=(52, 8))
    rows[kinds == "digits"] += 30.0  # only the digits are piled together
    reading = eg.readings(rows, kinds)
    assert reading["separation_digits"] > 0.5
    assert abs(reading["separation_code"]) < 0.2


def test_a_padding_id_is_other_not_punctuation(monkeypatch):
    """An id past the tokenizer decodes to nothing, and nothing looks like whitespace —
    it must not join a real group."""
    import types
    fake = types.SimpleNamespace(n_vocab=10, decode_single_token_bytes=lambda tid: b" the")
    monkeypatch.setattr(eg, "token_kinds", eg.token_kinds)
    import sys
    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=lambda name: fake))
    assert eg.token_kinds([3, 11]).tolist() == ["word", "other"]


def test_anisotropy_and_rms_read_a_known_shape():
    rows = np.zeros((50, 4))
    rows[:, 0] = np.linspace(-1, 1, 50)  # all the variance on one axis
    reading = eg.readings(rows, np.array(["word"] * 50))
    assert reading["anisotropy"] > 0.999
    assert np.isclose(reading["rms"], np.sqrt((rows ** 2).mean()))
    # One kind only: there is no across-kind pair to compare against, and saying 0
    # would claim "measured, no grouping".
    assert np.isnan(reading["separation"])


def test_the_point_cloud_is_edges_and_one_object_per_kind(tmp_path):
    """The board's mesh widget draws EDGES: a bare vertex is invisible there, so every
    token must arrive as a cross of three segments, under a named object its colour can
    be pinned to."""
    points = np.array([[0.0, 0, 0], [1.0, 2, 3], [-1.0, 0, 1]])
    kinds = np.array(["word", "digits", "word"])
    text = pathlib.Path(eg.write_point_cloud(points, kinds, tmp_path / "cloud.obj")).read_text()
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("o ")] == ["digits", "word"]
    assert sum(ln.startswith("l ") for ln in text.splitlines()) == 3 * len(points)
    assert sum(ln.startswith("v ") for ln in text.splitlines()) == 6 * len(points)
    # Every segment names two vertices that exist, and each is one axis long.
    vertices = [np.array([float(x) for x in ln.split()[1:]])
                for ln in text.splitlines() if ln.startswith("v ")]
    for line in (ln for ln in text.splitlines() if ln.startswith("l ")):
        a, b = (vertices[int(i) - 1] for i in line.split()[1:])
        assert np.count_nonzero(np.abs(a - b) > 1e-9) == 1


def test_density_blurs_into_a_field_without_moving_its_middle():
    """The soft field is what makes two kinds sharing ground read as a mixed colour; it
    must not slide the mass away from where the tokens are. The band here is half a bin,
    not four: an even blur kernel shifts the whole field by 1.5 bins, which is exactly the
    defect this guards, and a loose band would pass it."""
    points = np.zeros((50, 2))
    field = eg.density(points, limit=1.0, bins=50)
    assert np.isclose(field.max(), 1.0)
    grid = np.indices(field.shape)
    centroid = [float((axis * field).sum() / field.sum()) for axis in grid]
    assert all(abs(c - 25.0) <= 0.5 for c in centroid), centroid  # a point at 0 lands in bin 25 of 50
    assert (field > 0.01).sum() > 4  # and spread over neighbouring cells, not one spike


def test_frequency_counts_come_from_the_shards(tmp_path):
    source = tmp_path / "pretrain" / "fineweb-edu"
    source.mkdir(parents=True)
    np.save(source / "chunk_0.npy", np.array([5] * 10 + [7] * 6 + [9] * 3, dtype=np.int32))
    assert eg.frequent_tokens(str(tmp_path), ["fineweb-edu"], 2, 100).tolist() == [5, 7]
