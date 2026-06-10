"""Guards against the source-exhaustion bug: the trainer supplies full-length
curriculum weight lists every batch, while DataMixer drops exhausted sources.
Before set_weights existed, the external assignment clobbered the renormalized
weights and zip() silently truncated — draws could come back undersized."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

from data_loaders import DataMixer


class FakeSource:
    """Stands in for TextDataGenerator: serves `rows_left` rows, then exhausts."""

    def __init__(self, tag, rows_left):
        self.tag = tag
        self.rows_left = rows_left
        self.exhausted = False

    def get_batch(self, n):
        if self.rows_left < n:
            self.exhausted = True
            return None, None
        self.rows_left -= n
        return jnp.full((n, 4), self.tag, dtype=jnp.int32), jnp.zeros((n,), dtype=bool)


def make_mixer(rows=(10_000, 3, 10_000), weights=(0.3, 0.4, 0.3)):
    sources = [FakeSource(i, r) for i, r in enumerate(rows)]
    return sources, DataMixer(sources, list(weights), rng=np.random.default_rng(0))


def test_full_batches_and_redistribution_after_exhaustion():
    sources, mixer = make_mixer()
    seen_after_exhaustion = set()
    for _ in range(300):
        mixer.set_weights([0.2, 0.5, 0.3])  # full-length list every call, like the trainer
        batch, mask = mixer.get_batch(2)
        assert batch is not None, "mixer returned no batch while sources remain"
        assert batch.shape[0] == 2, f"undersized batch: {batch.shape}"
        if sources[1].exhausted:
            seen_after_exhaustion.update(batch[:, 0].tolist())
    assert sources[1].exhausted, "middle source never exhausted"
    assert 1 not in seen_after_exhaustion, "exhausted source kept serving rows"
    assert {0, 2} <= seen_after_exhaustion, "surviving sources did not absorb the freed mass"


def test_returns_none_when_all_sources_exhausted():
    sources, mixer = make_mixer(rows=(4, 4, 4))
    for _ in range(50):
        batch, mask = mixer.get_batch(2)
        if batch is None:
            break
    else:
        raise AssertionError("mixer never signalled exhaustion")
    assert all(s.exhausted for s in sources)


def test_set_weights_renormalizes_over_survivors():
    sources, mixer = make_mixer(rows=(10_000, 0, 10_000), weights=(0.1, 0.8, 0.1))
    for _ in range(20):  # draw until the empty middle source gets sampled and dropped
        mixer.get_batch(2)
        if sources[1].exhausted:
            break
    assert sources[1].exhausted
    mixer.set_weights([0.2, 0.5, 0.3])
    assert len(mixer.weights) == 2
    assert abs(sum(mixer.weights) - 1.0) < 1e-9
    assert abs(mixer.weights[0] - 0.4) < 1e-9  # 0.2 / (0.2 + 0.3)
    assert abs(mixer.weights[1] - 0.6) < 1e-9  # 0.3 / (0.2 + 0.3)
