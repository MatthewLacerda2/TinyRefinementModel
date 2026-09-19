"""A resumed data stream is the stream an uninterrupted run would have read (#424).

The old resume re-derived a per-corpus skip from the average mixture weights and
restarted every rng from the seed: each file's random start offset was dropped and
the mixer drew different corpora. The loader now snapshots its exact state with
every batch, and the checkpoint keeps the one for the last batch consumed.
"""

import json

import numpy as np
import pytest

from trm.config import MAX_SEQ_LEN

STRIDE = 2 * MAX_SEQ_LEN + 1


def _corpus(root, name, samples_per_file, files, base):
    """Shards whose every token is unique, so two streams agree only if they read
    the same tokens in the same place."""
    d = root / name
    d.mkdir(exist_ok=True)
    start = base
    for i in range(files):
        n = samples_per_file * STRIDE + 37  # a ragged tail, like a real shard
        np.save(d / f"chunk_{i}.npy", (np.arange(start, start + n) % 50000).astype(np.int32))
        start += n
    return str(d)


def _mixer(tmp_path):
    from trm.data.loaders import DataMixer, TextDataGenerator
    dirs = [_corpus(tmp_path, "web", 4, 3, 0), _corpus(tmp_path, "code", 3, 2, 20000)]
    return DataMixer([TextDataGenerator(d, rng=np.random.default_rng(7)) for d in dirs],
                     [0.6, 0.4], rng=np.random.default_rng(11))


@pytest.mark.parametrize("cut", [1, 5, 9, 13])  # mid-file, at and past file ends
def test_a_restored_mixer_serves_the_rows_an_uninterrupted_one_would(tmp_path, cut):
    reference = _mixer(tmp_path)
    expected = [reference.get_batch(1)[0] for _ in range(18)]

    first = _mixer(tmp_path)
    for _ in range(cut):
        first.get_batch(1)
    # The snapshot crosses a checkpoint as JSON; so does it here.
    snapshot = json.loads(json.dumps(first.state()))

    resumed = _mixer(tmp_path)
    resumed.load_state(snapshot)
    for want in expected[cut:]:
        got = resumed.get_batch(1)[0]
        if want is None:
            assert got is None
            break
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_a_state_from_a_different_source_list_is_refused(tmp_path):
    mixer = _mixer(tmp_path)
    state = mixer.state()
    state["sources"] = state["sources"][:1]
    with pytest.raises(ValueError, match="PRETRAIN_SOURCES"):
        _mixer(tmp_path).load_state(state)
