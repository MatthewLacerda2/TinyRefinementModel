"""Rotary position embedding (RoPE) — the single copy both architectures share.

The rotation and the cos/sin table construction used to live twice, in
layers.py (RotaryAttention) and plan_a_model.py, with identical math.
Deliberately config-free: plan_a_model stays fully parametrized so the exact
same architecture runs at toy scale and at real scale.
"""

import jax
import jax.numpy as jnp


def rope_tables(max_pos, head_dim):
    """cos/sin tables for positions [0, max_pos), each [max_pos, head_dim // 2]."""
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
    freqs = jnp.outer(jnp.arange(max_pos), inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    """Rotate feature pairs of x by position (split-half convention)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jax.lax.complex(x1, x2) * jax.lax.complex(cos, sin)
    return jnp.concatenate([rotated.real, rotated.imag], axis=-1).astype(x.dtype)
