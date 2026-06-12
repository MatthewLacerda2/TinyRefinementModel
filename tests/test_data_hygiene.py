"""Data hygiene: the token streams must contain what the model assumes.

Catches corrupted prefill output before it costs a training run: token ids
outside the vocabulary (silent garbage through the embedding lookup or an
out-of-bounds gather), PAD flooding (a run that mostly trains on padding), and
wrong storage dtype. Samples the head and a few random slices of each chunk
via memory-mapping, so it stays cheap regardless of file size.
"""

import glob
import os

import numpy as np
import pytest
from dotenv import load_dotenv

from config import VOCAB_SIZE, PAD_TOKEN_ID, resolve_root

load_dotenv()

SAMPLE_TOKENS = 2_000_000  # per file: head slice + random slices


def _source_dirs():
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        return []
    root = f"{resolve_root(data_root)}/pretrain"
    return sorted(d for d in glob.glob(f"{root}/*/") if glob.glob(d + "*.npy"))


@pytest.mark.skipif(not _source_dirs(), reason="local pretrain data not available")
@pytest.mark.parametrize("source_dir", _source_dirs(), ids=os.path.basename)
def test_token_streams_are_sane(source_dir):
    rng = np.random.default_rng(0)
    for path in sorted(glob.glob(source_dir + "*.npy")):
        data = np.load(path, mmap_mode="r")
        assert np.issubdtype(data.dtype, np.integer), f"{path}: tokens stored as {data.dtype}"

        chunk = SAMPLE_TOKENS // 4
        starts = [0] + list(rng.integers(0, max(len(data) - chunk, 1), size=3))
        sample = np.concatenate([np.asarray(data[s:s + chunk]) for s in starts])

        assert sample.min() >= 0, f"{path}: negative token ids"
        assert sample.max() < VOCAB_SIZE, (
            f"{path}: token id {sample.max()} >= vocab size {VOCAB_SIZE}"
        )
        pad_fraction = float(np.mean(sample == PAD_TOKEN_ID))
        assert pad_fraction < 0.25, (
            f"{path}: {pad_fraction:.0%} of sampled tokens are PAD — stream is mostly padding"
        )
