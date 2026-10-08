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
else:
    # The embedding backward is an atomic scatter-add on the card, so the same step
    # differs from itself run to run; the identity tests need it pinned. Tests only:
    # runs leave it off for its +13% per micro-step (#540, #556).
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()

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
    from trm.settings import CONFIG

    rng = np.random.default_rng(7)
    tokens = rng.integers(1, 5000, size=(1, 64))
    tokens[tokens == CONFIG.PAD_TOKEN_ID] += 1
    return tokens.astype(np.int32)


# Small enough that building one costs seconds on CPU, big enough for every
# property the consumers check (padding, causality, init loss, checkpoint schema).
TINY_DIM = 60
TINY_LAYERS = 2


@pytest.fixture(scope="session")
def make_tiny_model():
    """Build a small PlainTransformer at `seed`, through the factory the trainer uses.
    For tests that need a second instance of the same shape, e.g. to restore a
    checkpoint into a differently-initialized model."""
    from flax import nnx

    from trm.model import build_model
    from trm.settings import CONFIG

    def make(seed=0, **extra):
        return build_model(CONFIG, nnx.Rngs(seed), dim=TINY_DIM,
                           **{"num_layers": TINY_LAYERS, **extra})
    return make


@pytest.fixture(scope="session")
def tiny_model(make_tiny_model):
    """A small PlainTransformer, shared across the session."""
    return make_tiny_model(seed=0)


# --- helpers several test files used to copy byte for byte (#325) -------------------

# Anchored on the marker file, not on a fixed number of parent hops: a
# `dirname(dirname(__file__))` silently became `tests/` the day test files moved into
# tier folders, and a subprocess run from there failed with a bare ModuleNotFoundError.
_REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


@pytest.fixture(scope="session")
def repo_root():
    """The repository root (the directory holding pyproject.toml), as a pathlib.Path."""
    return _REPO


@pytest.fixture
def ce_batch():
    """Build a random (hidden [b, s, d], embedding [vocab, d], targets [b, s]) triple for
    checking the chunked cross-entropy against a naive full-logit computation."""
    import jax.numpy as jnp

    def make(seed=0, b=2, s=40, d=8, vocab=17):
        rng = np.random.default_rng(seed)
        hidden = jnp.asarray(rng.standard_normal((b, s, d)), dtype=jnp.float32)
        embedding = jnp.asarray(rng.standard_normal((vocab, d)), dtype=jnp.float32)
        targets = jnp.asarray(rng.integers(0, vocab, size=(b, s)), dtype=jnp.int32)
        return hidden, embedding, targets
    return make


@pytest.fixture(scope="session")
def n_params():
    """Count a model's trainable parameters: the sum of every nnx.Param leaf's size."""
    import jax
    from flax import nnx

    return lambda model: sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


@pytest.fixture(scope="session")
def in_vocab_encoder():
    """Build a tokenizer whose ids fit a toy vocabulary of size `vocab`.

    The real `r50k_base` emits ids in the tens of thousands (`generate_text` pads with
    the model's own pad, so a toy model's pad is in range). One out-of-range id makes a toy
    model return all-NaN logits for the whole window (#233), so a generation test on the
    real tokenizer sampled every token from NaN and passed only because NaN is
    deterministic. The #229 guard is what surfaced it."""
    class InVocabEncoder:
        def __init__(self, vocab):
            self.vocab = vocab

        def encode(self, text):
            return [1 + (ord(c) % (self.vocab - 2)) for c in text]

        def decode(self, ids):
            return "".join(chr(97 + (i % 26)) for i in ids)

    return InVocabEncoder


@pytest.fixture
def recorded_run(tmp_path):
    """Lay out a recorded run as `tmp_path/runs/<name>/` and return its directory: an
    excerpt of its metrics.csv and its run_metadata.json. Which runs, what was kept, and
    why: tests/apparatus/fixtures/README.md."""
    import shutil

    fixtures = pathlib.Path(__file__).parent / "apparatus" / "fixtures"

    def lay_out(name):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        shutil.copy(fixtures / name / "metrics.csv", run / "metrics.csv")
        # The champion's metadata predates this layout and is shared with test_base_run.
        metadata = (fixtures / "champion_run_metadata.json" if name == "run_20260813_214725"
                    else fixtures / name / "run_metadata.json")
        shutil.copy(metadata, run / "run_metadata.json")
        return run
    return lay_out


@pytest.fixture
def champion_run(recorded_run):
    """The recorded 4B champion run, `run_20260813_214725`, laid out in tmp_path."""
    return recorded_run("run_20260813_214725")
