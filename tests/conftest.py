"""Shared test configuration.

Tests default to CPU so the suite stays runnable while a training process owns
the GPU. Set RUN_TESTS_ON_GPU=1 to opt back in when the GPU is free.
This must happen before any test module imports jax.
"""

import os

if not os.environ.get("RUN_TESTS_ON_GPU"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # CPU XLA cannot lower the model's f16-with-f32-accumulation matmuls
    # (see config.py). GPU mode exercises the real f16 path.
    os.environ.setdefault("FORCE_F32_COMPUTE", "1")

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a free GPU; skipped unless RUN_TESTS_ON_GPU=1"
    )


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("RUN_TESTS_ON_GPU"):
        skip_gpu = pytest.mark.skip(reason="GPU busy or unavailable (set RUN_TESTS_ON_GPU=1)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def token_batch():
    """Fixed [1, 64] token batch, values clear of PAD."""
    from config import PAD_TOKEN_ID

    rng = np.random.default_rng(7)
    tokens = rng.integers(1, 5000, size=(1, 64))
    tokens[tokens == PAD_TOKEN_ID] += 1
    return tokens.astype(np.int32)


@pytest.fixture(scope="session")
def tiny_model():
    """Full UniversalReasoner (batch 1), shared across the session — construction
    dominates test time on CPU, the forwards are cheap at short sequence lengths."""
    from flax import nnx
    from config import LATENT_DIM
    from model import UniversalReasoner

    return UniversalReasoner(LATENT_DIM, nnx.Rngs(0), batch_size=1)
