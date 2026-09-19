"""Which data produces the gradient tail (#364): every micro-step's norm is filed
under the source it came from."""

import numpy as np


class _Source:
    def __init__(self, value):
        self.value, self.exhausted = value, False

    def get_batch(self, count):
        import jax.numpy as jnp

        return jnp.full((count, 3), self.value), jnp.ones((count, 3))


def test_the_mixer_names_the_source_of_a_single_source_batch():
    from trm.data.loaders import DataMixer

    mixer = DataMixer([_Source(1), _Source(2)], [0.0, 1.0], rng=np.random.default_rng(0))
    batch, _ = mixer.get_batch(1)
    assert int(batch[0, 0]) == 2 and mixer.last_source == 1


def test_a_batch_that_mixes_sources_is_filed_as_mixed():
    from trm.data.loaders import DataMixer

    mixer = DataMixer([_Source(1), _Source(2)], [0.5, 0.5], rng=np.random.default_rng(3))
    for _ in range(20):
        batch, _ = mixer.get_batch(8)
        if len(set(np.asarray(batch[:, 0]).tolist())) > 1:
            assert mixer.last_source is None
            return
    raise AssertionError("no mixed batch drawn in 20 tries")


def test_the_window_label_is_mean_max_clipped_count_per_source():
    from trm.train.trainer import SourceGrads

    grads = SourceGrads(("pretrain/fineweb-edu", "pretrain/codeparrot"))
    for norm, clipped in ((10.0, False), (30.0, True)):
        grads.add(1, norm, clipped)
    grads.add(0, 5.0, False)
    grads.add(None, 7.0, False)
    assert grads.label() == "codeparrot=20.0/30.0/1/2 fineweb-edu=5.0/5.0/0/1 mixed=7.0/7.0/0/1"
    grads.reset()
    assert grads.label() == ""
