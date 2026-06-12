"""Data pipeline determinism.

Two generators built the same way must yield identical batch streams — this is
what makes a training run reproducible from its seed at all.

Deliberately NOT tested: that skip_count resume replays the exact same stream.
It does not, by design — fresh runs apply a random boundary-augmentation offset
that the skip path bypasses, and the trainer splits the resume total across
sources by expected mixture weights rather than actual draws. Resume is
approximate. If that ever becomes exact, add the replay test here.
"""

import os

import numpy as np
import pytest
from dotenv import load_dotenv

from config import resolve_root

load_dotenv()


def _source_dir():
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        return None
    path = f"{resolve_root(data_root)}/pretrain/fineweb-edu"
    return path if os.path.isdir(path) else None


@pytest.mark.skipif(_source_dir() is None, reason="local pretrain data not available")
def test_same_seed_yields_identical_batch_stream():
    from data_loaders import TextDataGenerator

    gen_a = TextDataGenerator(_source_dir())
    gen_b = TextDataGenerator(_source_dir())

    for _ in range(4):
        batch_a, bound_a = gen_a.get_batch(2)
        batch_b, bound_b = gen_b.get_batch(2)
        np.testing.assert_array_equal(np.asarray(batch_a), np.asarray(batch_b))
        np.testing.assert_array_equal(np.asarray(bound_a), np.asarray(bound_b))
