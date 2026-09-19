"""The prefetch thread builds batches in numpy, never JAX (#411).

A JAX op on the loader thread (dispatch, a host-to-device copy) competes with the
training thread for the GIL every micro-step; the jitted grad step moves the batch
to the card itself. This guards that the loader hands back plain numpy, and a copy:
a view of the memory-mapped shard would move its disk reads onto the training thread.
"""

import numpy as np

from trm.config import MAX_SEQ_LEN


def test_the_loader_and_the_mixer_return_numpy(tmp_path):
    from trm.data.loaders import DataMixer, TextDataGenerator

    np.save(tmp_path / "chunk_0.npy", np.full(8 * MAX_SEQ_LEN + 8, 5, dtype=np.int32))

    rows, boundary = TextDataGenerator(str(tmp_path)).get_batch(2)
    assert type(rows) is np.ndarray and rows.dtype == np.int32
    assert type(boundary) is np.ndarray

    mixer = DataMixer([TextDataGenerator(str(tmp_path))], [1.0])
    rows, boundary = mixer.get_batch(2)
    assert type(rows) is np.ndarray and type(boundary) is np.ndarray
