"""Shared test configuration.

Tests default to CPU so the suite stays runnable while a training process owns
the GPU. Set RUN_TESTS_ON_GPU=1 to opt back in when the GPU is free.
This must happen before any test module imports jax.

Every test lives in exactly one tier folder — core/, apparatus/, or expensive/ —
and the folder IS the declaration (see tests/README.md). The guard below refuses
to collect a test file dropped straight into tests/, because "where does this go"
must have one answer, and an unfiled test is one nobody can decide to delete.
"""

import os
import pathlib

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


TIERS = ("core", "apparatus", "expensive")
_TESTS_ROOT = pathlib.Path(__file__).parent.resolve()


def pytest_collection_modifyitems(config, items):
    """Refuse to collect a test that isn't under one of the three tiers.

    Catches both ways of dodging the taxonomy: a file left in tests/ directly,
    and a fourth folder invented to avoid choosing.
    """
    unfiled = sorted({
        str(path.relative_to(_TESTS_ROOT))
        for path in (pathlib.Path(str(item.fspath)).resolve() for item in items)
        if path.is_relative_to(_TESTS_ROOT) and path.relative_to(_TESTS_ROOT).parts[0] not in TIERS
    })
    if unfiled:
        raise pytest.UsageError(
            "these test files are not in a tier folder:\n  tests/"
            + "\n  tests/".join(unfiled)
            + f"\nEvery test belongs to exactly one of {TIERS} — "
              "see tests/README.md for what each one means and when it gets deleted."
        )

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
