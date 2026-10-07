"""What the model weighs — counted, not built.

This module **never imports or constructs a model class**. It is arithmetic over
this process's Config (trm/settings.py) and the constants in `trm/config.py`, and
that is the point: the tool it replaces instantiated a whole model on
the way to printing a parameter count — slow, and a real hazard to run against a busy 6GB card just to answer a
question that is a sum of products.

Purity has an obvious failure mode: a formula transcribed once and then left
behind when the architecture moves. That is handled where it belongs, in the
test suite — `tests/apparatus/test_model_stats.py` builds the real model on CPU
and asserts these numbers match the instantiated tree exactly,
per group as well as in total. The instrument stays pure; the test guarantees it
is not lying.

**On VRAM.** Parameters, AdamW moments and gradients are exactly computable:
count x dtype width, and `trm/train/optimizers.py` fixes the widths (bf16 first
moment, f32 variance, plus a full f32 gradient accumulator inside
`optax.MultiSteps`). Activations are *not* exactly computable, and pretending
otherwise is the specific failure this replaces: the old estimate claimed 15.47
MB of activations and a 2.08 GB total against a real ~5.0 GB peak. What that
missed is not one tensor — it is the activations each block keeps for the
backward, XLA scratch, concurrently-alive compiled programs, and allocator overhead
(`docs/findings/2026-08-14-bfc-fragmentation-killed-every-base-run.md`).

So `vram_estimate` returns the terms it can defend and calls the sum a
**floor**. The one activation term included — each block's input state, kept for
the backward — is labelled a lower bound and is a lower bound; it is not an
activation total, and the floor is not a prediction of peak VRAM.
"""

from __future__ import annotations

from typing import NamedTuple

from trm.config import VOCAB_SIZE
from trm.settings import CONFIG

MIB = 1024 ** 2

# Dtype widths in bytes. Parameters are stored f32 (NNX's default, see trm/config.py) whatever
# the compute dtype is; the optimizer's widths are set in trm/train/optimizers.py.
F32, BF16, F16 = 4, 2, 2


class MeasuredPeak(NamedTuple):
    config: dict   # resolved-config values that must all match, plus "batch"
    gb: float
    source: str    # where the number came from
    reading: str   # what it measured: "card" (nvidia-smi, the whole card) or "arena" (the allocator's pool only)


# Training peaks measured on the card, each for one exact config — the only numbers
# the report may quote beside the floor. A config not listed has no measured peak,
# and the report says so rather than quoting a neighbour's (#317).
_PLAIN_ARENA_PEAKS_MIB = {8: 4112, 9: 4437, 10: 4762}   # trm/config.py, the PLAIN_LAYERS note
MEASURED_PEAKS = tuple(
    MeasuredPeak({"dim": 960, "num_heads": 15, "num_layers": layers, "post_norm": False, "batch": 1},
                 mib * MIB / 1e9,
                 f"instruments.vram_headroom_smoke 2026-09-13: {mib} MiB allocator arena peak "
                 f"under cuda_async (trm/config.py, PLAIN_LAYERS)", "arena")
    for layers, mib in _PLAIN_ARENA_PEAKS_MIB.items())


def _linear(in_features, out_features, bias=True):
    """nnx.Linear: a [in, out] kernel plus an [out] bias (use_bias defaults True)."""
    return in_features * out_features + (out_features if bias else 0)


def _rmsnorm(features):
    """nnx.RMSNorm: a scale vector. No bias — use_bias defaults False."""
    return features


def _embed(rows, dim):
    return rows * dim


# ── The block ─────────────────────────────────────────────────────────────────

def _mlp_hidden(dim):
    """SwiGLU width for plain.Block — 8/3 x dim rounded UP to a multiple of 64."""
    return ((int(8 * dim / 3) + 63) // 64) * 64


def _block(dim, num_heads, post_norm):
    """plain.Block: MHA (RoPE + q/k norms) + SwiGLU MLP + two RMSNorms; post-norm adds
    an RMSNorm on each residual branch."""
    head_dim = dim // num_heads
    attention = 4 * _linear(dim, dim) + 2 * _rmsnorm(head_dim)   # q, k, v, o
    hidden = _mlp_hidden(dim)
    mlp = 2 * _linear(dim, hidden) + _linear(hidden, dim)        # gate, up, down
    return attention + 2 * _rmsnorm(dim) + mlp + (2 * _rmsnorm(dim) if post_norm else 0)


# ── Config resolution ────────────────────────────────────────────────────────

def _defaults():
    return {
        "dim": CONFIG.LATENT_DIM,
        "vocab_size": VOCAB_SIZE,
        "num_heads": CONFIG.NUM_HEADS,
        "num_layers": CONFIG.PLAIN_LAYERS,
        "max_seq_len": CONFIG.MAX_SEQ_LEN,
        "post_norm": CONFIG.POST_NORM,
    }


def _resolve(overrides):
    """The model's config, with overrides applied. An unknown key is an error:
    silently ignoring `n_heads=8` would report the default's numbers under the
    caller's name, which is the whole class of bug this module exists to end."""
    config = _defaults()
    unknown = set(overrides) - set(config)
    if unknown:
        raise TypeError(f"unknown override(s): {sorted(unknown)}; accepted: {sorted(config)}")
    config.update(overrides)
    return config


# ── Parameters ───────────────────────────────────────────────────────────────

def param_breakdown(**overrides):
    """Parameter count per group, as a dict in a sensible reading order. The groups
    mirror the real param tree (the test asserts exactly that)."""
    config = _resolve(overrides)
    dim = config["dim"]
    return {
        "embeddings & tied head": _embed(config["vocab_size"], dim),
        "blocks": config["num_layers"] * _block(dim, config["num_heads"], config["post_norm"]),
        "heads & norms": _rmsnorm(dim),   # out_norm
    }


def total_params(**overrides):
    return sum(param_breakdown(**overrides).values())


# ── VRAM ─────────────────────────────────────────────────────────────────────

TOTAL_KEY = "TOTAL (floor — activations excluded)"


def _block_input_bytes(config, batch):
    """A LOWER BOUND on activation memory: every block's input, kept for the backward.

    There is no remat, so each block keeps far more than its input (the attention and
    MLP intermediates, XLA scratch); none of that is modelled here, which is exactly
    why this is a bound and not an estimate. The grad step scores two prediction
    windows and holds both graphs at once, hence x2."""
    per_window_state = batch * config["max_seq_len"] * config["dim"] * F16
    return 2 * config["num_layers"] * per_window_state


def vram_estimate(mode, batch=CONFIG.BATCH_SIZE, **overrides):
    """VRAM line items in MiB. `mode` is "train" or "infer".

    Every entry is a real, named tensor whose size follows from a count and a
    dtype — except the block-input line, which is a stated lower bound. The
    `TOTAL` entry is a FLOOR: the true peak is larger by the activation and
    allocator terms nothing analytic can pin down (MEASURED_PEAKS holds the
    measured ones). It is also the sum of the entries above it, so do not re-sum
    the dict.
    """
    if mode not in ("train", "infer"):
        raise ValueError(f"mode must be 'train' or 'infer', not {mode!r}")
    config = _resolve(overrides)
    params = total_params(**overrides)

    lines = {"parameters (f32)": params * F32}
    if mode == "train":
        lines.update({
            "gradients (f32)": params * F32,
            "AdamW mu (bf16)": params * BF16,
            "AdamW nu (f32)": params * F32,
            "MultiSteps accumulated grads (f32)": params * F32,
            "block input states (f16, lower bound)": _block_input_bytes(config, batch),
        })
    else:
        # Inference materializes full logits — the training path deliberately
        # does not (chunked CE, #19), which is why this line is here and not there.
        lines["logits [batch, seq, vocab] (f32)"] = (
            batch * config["max_seq_len"] * config["vocab_size"] * F32
        )

    megabytes = {name: value / MIB for name, value in lines.items()}
    megabytes[TOTAL_KEY] = sum(megabytes.values())
    return megabytes


def measured_peak(batch=CONFIG.BATCH_SIZE, **overrides):
    """The MeasuredPeak recorded for exactly this config, or None. Quoting a
    measured number against a different config would be the same category error
    this module is trying to stop."""
    config = {**_resolve(overrides), "batch": batch}
    for peak in MEASURED_PEAKS:
        if all(config.get(k) == v for k, v in peak.config.items()):
            return peak
    return None
