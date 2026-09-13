"""Every process shares one bounded, on-SSD compilation cache (#204).

Measured when it landed (RTX 2060, plain stack, grad step + optimizer apply):
25.4s to compile cold, 5.6s warm, loss bit-identical. It must stay transparent —
a cache that changed a number would be a bug, not an optimization.
"""

import os
import pathlib

import jax

from trm import config


def test_the_cache_lives_beside_the_runs_and_is_bounded():
    repo = pathlib.Path(config.__file__).resolve().parents[1]
    assert pathlib.Path(config.COMPILATION_CACHE_DIR) == repo / "runs" / ".jax_cache"
    assert 0 < config.COMPILATION_CACHE_MAX_BYTES <= 4 * 1024**3


def test_importing_config_turns_it_on_unless_the_environment_says_otherwise():
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        assert jax.config.values["jax_compilation_cache_dir"] == os.environ["JAX_COMPILATION_CACHE_DIR"]
    else:
        assert jax.config.values["jax_compilation_cache_dir"] == config.COMPILATION_CACHE_DIR
        assert jax.config.values["jax_compilation_cache_max_size"] == config.COMPILATION_CACHE_MAX_BYTES


def test_runs_is_gitignored_so_the_cache_never_reaches_a_commit():
    repo = pathlib.Path(config.__file__).resolve().parents[1]
    assert "runs/" in (repo / ".gitignore").read_text().splitlines()
