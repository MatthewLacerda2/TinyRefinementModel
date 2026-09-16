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
    from trm.config import PAD_TOKEN_ID

    rng = np.random.default_rng(7)
    tokens = rng.integers(1, 5000, size=(1, 64))
    tokens[tokens == PAD_TOKEN_ID] += 1
    return tokens.astype(np.int32)


# Small enough that building one costs seconds on CPU, big enough for every
# property the consumers check (padding, causality, init loss, checkpoint schema).
# Before #322 this was the full 138M-param reasoner — the control arch — so eight
# core tests asserted general properties through an architecture nothing ships.
TINY_DIM = 60
TINY_OVERRIDES = {"plain": {"num_layers": 2}}


@pytest.fixture(scope="session")
def make_tiny_model():
    """Build a small model of the architecture a run would train (MODEL_ARCH), at
    `seed`. For tests that need a second instance of the same shape, e.g. to
    restore a checkpoint into a differently-initialized model."""
    from instruments.arch import build
    from trm.config import MODEL_ARCH

    def make(seed=0):
        return build(MODEL_ARCH, dim=TINY_DIM, seed=seed, **TINY_OVERRIDES.get(MODEL_ARCH, {}))
    return make


@pytest.fixture(scope="session")
def tiny_model(make_tiny_model):
    """A small MODEL_ARCH model (plain by default), shared across the session."""
    return make_tiny_model(seed=0)


@pytest.fixture(scope="session")
def make_reasoner_model():
    """Build a small UniversalReasoner at `seed`, for the tests that read state only
    the reasoner has: the carried hunch and the aux regularizers. Goes with #292."""
    from instruments.arch import build

    return lambda seed=0: build("reasoner", dim=TINY_DIM, seed=seed)


@pytest.fixture(scope="session")
def reasoner_model(make_reasoner_model):
    """A small UniversalReasoner, shared across the session."""
    return make_reasoner_model(seed=0)


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


_CONFIG_CHILD = r"""
import importlib, json, os, sys
cases, attrs = json.loads(sys.argv[1]), json.loads(sys.argv[2])
base = dict(os.environ)
out = []
for case in cases:
    os.environ.clear()
    os.environ.update(base)
    for key, value in case.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    # Evict every repo module, not only trm.config: anything config imports that also
    # reads the environment at load would otherwise stay cached from the first case.
    for name in [m for m in sys.modules if m == "trm" or m.startswith("trm.")]:
        del sys.modules[name]
    # "NAME" reads trm.config; "pkg.module:NAME" reads that module, imported here too
    # because importing it can refuse the same way config does.
    targets = [(a, *a.rpartition(":")[::2]) for a in attrs]
    try:
        modules = {"": importlib.import_module("trm.config")}
        for _, module, _ in targets:
            if module and module not in modules:
                modules[module] = importlib.import_module(module)
    except BaseException as exc:  # SystemExit included: that is how the guards refuse
        out.append({"ok": False, "error": str(exc)})
        continue
    # Outside the refusal try: a mistyped attribute name crashes the child, loudly.
    out.append({"ok": True, "values": {a: getattr(modules[module], name) for a, module, name in targets}})
print("CONFIG_CASES " + json.dumps(out))
"""


@pytest.fixture(scope="session")
def import_config_under(repo_root):
    """Import trm.config once per environment case, all cases in ONE fresh interpreter.

    config reads the environment and validates it at import, and the test process has
    long since imported it, so each case needs a fresh import. A separate interpreter
    per case paid a cold Python + jax start each time (#325); here one child re-executes
    the module per case instead. `cases` is a list of {VAR: value} overrides, where None
    unsets VAR. `attrs` name trm.config attributes, or "pkg.module:NAME" for a module
    that reads config at import (it is re-imported per case too). Each case returns
    {"ok": True, "values": {attr: value}} when the import
    succeeded, or {"ok": False, "error": message} when it raised. The fail-closed guards
    raise SystemExit, the same exception a bare `import trm.config` would have died of."""
    import json
    import subprocess
    import sys

    def run(cases, attrs=()):
        r = subprocess.run(
            [sys.executable, "-c", _CONFIG_CHILD, json.dumps(cases), json.dumps(list(attrs))],
            env={**os.environ, "JAX_PLATFORMS": "cpu"}, cwd=repo_root, capture_output=True, text=True)
        assert r.returncode == 0, f"the config-import child itself failed:\n{r.stderr}"
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CONFIG_CASES ")][-1]
        return json.loads(line[len("CONFIG_CASES "):])
    return run


@pytest.fixture(scope="session")
def in_vocab_encoder():
    """Build a tokenizer whose ids fit a toy vocabulary of size `vocab`.

    The real `r50k_base` emits ids in the tens of thousands, and `generate_text` pads
    with the config PAD_TOKEN_ID (50256) on top of that. One out-of-range id makes a toy
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
def champion_run(tmp_path):
    """A copy of the recorded champion run, `runs/run_20260813_214725`, in tmp_path:
    its full run_metadata.json and an excerpt of its metrics.csv. What was kept, and
    why, is tests/apparatus/fixtures/README.md. Returns the run directory."""
    import shutil

    fixtures = pathlib.Path(__file__).parent / "apparatus" / "fixtures"
    run = tmp_path / "runs" / "run_20260813_214725"
    run.mkdir(parents=True)
    shutil.copy(fixtures / "run_20260813_214725" / "metrics.csv", run / "metrics.csv")
    shutil.copy(fixtures / "champion_run_metadata.json", run / "run_metadata.json")
    return run
