"""The #472 logit lens has to be wired to the model it reads: its last row is the
model's own next-token guess, and the path file is a glTF a viewer can load, with one
animated segment per block."""

import json
import struct

import numpy as np

from instruments import logit_lens as ll


def test_the_last_row_of_the_lens_is_the_models_own_guess(tiny_model):
    probs, normed, own = ll.lens(tiny_model, [50256, 464, 3139, 286])
    assert probs.shape[0] == len(tiny_model.blocks) + 1, "the embedding, then one row per block"
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-4)
    assert ll.gate(probs, own), "the lens at the last block picks the model's argmax"
    assert np.allclose(np.log(probs[-1]) - np.log(probs[-1]).max(), own - own.max(), atol=1e-3), \
        "not only the argmax: the whole last-row distribution is the model's"
    assert normed.shape == (len(tiny_model.blocks) + 1, tiny_model.embed.embedding.shape[1])


def test_a_lens_that_disagrees_with_the_model_fails_the_gate():
    probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert not ll.gate(probs, np.array([0.0, 1.0]))


def test_a_head_change_that_keeps_the_argmax_still_fails_the_gate():
    """A temperature on the model's logits keeps its argmax and moves every probability."""
    logits = np.array([2.0, 1.0, 0.0])
    probs = np.exp(logits) / np.exp(logits).sum()
    assert ll.gate(np.stack([probs, probs]), logits)
    assert not ll.gate(np.stack([probs, probs]), logits / 2.0)


def test_top_k_is_ordered_and_complete():
    rows = ll.top_k(np.array([[0.1, 0.6, 0.3]]), k=2)
    assert rows == [[(1, 0.6), (2, 0.3)]]


def test_the_path_is_drawn_at_the_clouds_radius_in_its_projection():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(4, 16)) * np.array([[1.0], [3.0], [9.0], [27.0]])  # the scale grows with depth
    centre, directions = np.zeros(16), np.eye(16)[:3]
    points = ll.path_points(states, centre, directions, radius=2.0)
    full = ll.path_points(states, centre, np.eye(16), radius=2.0)
    assert points.shape == (4, 3)
    assert np.allclose(np.linalg.norm(full, axis=1), 2.0), "direction only: every state at the cloud's radius"


def test_explained_variance_is_a_share():
    rows = np.diag([3.0, 2.0, 1.0, 0.5])
    share = ll.explained(rows, np.eye(4)[:2])
    assert 0.0 < share < 1.0 and ll.explained(rows, np.eye(4)) == 1.0


def test_the_path_file_is_a_glb_with_one_animated_segment_per_block():
    cloud = np.random.default_rng(1).normal(size=(30, 3))
    points = np.cumsum(np.ones((9, 3)), axis=0)
    blob = ll.path_glb(cloud, points)
    magic, version, total = struct.unpack("<4sII", blob[:12])
    assert (magic, version, total) == (b"glTF", 2, len(blob))
    length, kind = struct.unpack("<I4s", blob[12:20])
    assert kind == b"JSON" and length % 4 == 0
    doc = json.loads(blob[20:20 + length])
    names = [n["name"] for n in doc["nodes"]]
    assert names == ["cloud"] + [f"layer_{k}" for k in range(8)]
    channels = doc["animations"][0]["channels"]
    assert sorted(c["target"]["node"] for c in channels) == list(range(1, 9))
    assert all(c["target"]["path"] == "scale" for c in channels)
    bin_len, bin_kind = struct.unpack("<I4s", blob[20 + length:28 + length])
    assert bin_kind == b"BIN\0" and bin_len == doc["buffers"][0]["byteLength"] + (-doc["buffers"][0]["byteLength"] % 4)
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= bin_len
