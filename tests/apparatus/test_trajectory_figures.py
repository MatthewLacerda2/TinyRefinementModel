"""The residual-stream pictures (#391): drawn from a trajectory, on any plain model."""

import numpy as np
import pytest


def _states(blocks=4, positions=60, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(positions, dim)) + 5.0  # a shared mean, as real streams have
    steps = rng.normal(scale=0.3, size=(blocks, positions, dim))
    return np.concatenate([z0[None], z0[None] + np.cumsum(steps, axis=0)]).astype(np.float32)


def test_every_path_starts_at_the_origin_and_shares_one_plane():
    """The PCA is fitted on z_k - z_0, never on z itself, so the shared mean cannot
    eat the components and every token's path starts at (0, 0)."""
    from instruments.trajectory_figures import displacement_pca

    coords, explained = displacement_pca(_states())
    assert coords.shape == (5, 60, 2)
    assert np.allclose(coords[0], 0.0)
    assert 0.0 < explained.sum() <= 1.0


def test_relative_steps_are_scale_free():
    """Scaling the whole stream by 10 leaves every relative step where it was."""
    from instruments.trajectory_figures import relative_steps

    states = _states()
    assert relative_steps(states).shape == (4, 60)
    assert np.allclose(relative_steps(states), relative_steps(10.0 * states), rtol=1e-5)


def test_the_three_figures_are_written(tmp_path):
    from instruments.trajectory_figures import draw

    gaps = np.random.default_rng(1).uniform(size=60)
    written = draw(_states(), gaps, tmp_path, "toy · step 0")
    assert sorted(p.split("/")[-1] for p in written) == [
        "trajectory_difficulty.png", "trajectory_pca.png", "trajectory_steps.png"]
    assert all((tmp_path / name).stat().st_size > 0 for name in
               ("trajectory_difficulty.png", "trajectory_pca.png", "trajectory_steps.png"))


@pytest.mark.parametrize("layers", [2])
def test_collect_reads_a_plain_model_without_its_pad_positions(layers):
    import jax.numpy as jnp
    from flax import nnx

    from instruments.trajectory_figures import collect
    from trm.config import MAX_SEQ_LEN
    from trm.model.plain import PlainTransformer

    pad = 36
    model = PlainTransformer(32, nnx.Rngs(0), vocab_size=37, num_heads=4, num_layers=layers,
                             max_seq_len=MAX_SEQ_LEN, pad_token_id=pad)
    row = jnp.full((1, MAX_SEQ_LEN), pad, dtype=jnp.int32).at[0, :10].set(jnp.arange(1, 11))

    states, gaps = collect(model, [row], pad)
    assert states.shape == (layers + 1, 10, 32), "only the ten real tokens"
    assert gaps.shape == (10,) and np.all((gaps >= 0) & (gaps <= 1))
