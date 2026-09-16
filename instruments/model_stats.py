"""What the model weighs — counted, not built.

This module **never imports or constructs a model class**. It is arithmetic over
the constants in `trm/config.py`, and that is the point: the tool it replaces
instantiated a whole `UniversalReasoner` on the way to printing a parameter
count — slow, and a real hazard to run against a busy 6GB card just to answer a
question that is a sum of products.

Purity has an obvious failure mode: a formula transcribed once and then left
behind when the architecture moves. That is handled where it belongs, in the
test suite — `tests/apparatus/test_model_stats.py` builds the real model on CPU
for both arches and asserts these numbers match the instantiated tree exactly,
per group as well as in total. The instrument stays pure; the test guarantees it
is not lying.

**On VRAM.** Parameters, AdamW moments and gradients are exactly computable:
count x dtype width, and `trm/train/optimizers.py` fixes the widths (bf16 first
moment, f32 variance, plus a full f32 gradient accumulator inside
`optax.MultiSteps`). Activations are *not* exactly computable, and pretending
otherwise is the specific failure this replaces: the old estimate claimed 15.47
MB of activations and a 2.08 GB total against a real ~5.0 GB peak. What that
missed is not one tensor — it is remat recomputation buffers, XLA scratch,
28 concurrently-alive compiled programs (one per sampled depth x the
accumulate/apply branches), and allocator overhead
(`docs/findings/2026-08-14-bfc-fragmentation-killed-every-base-run.md`).

So `vram_estimate` returns the terms it can defend and calls the sum a
**floor**. The one activation term included — the states `jax.checkpoint` keeps
at each remat boundary — is labelled a lower bound and is a lower bound; it is
not an activation total, and the floor is not a prediction of peak VRAM.
"""

from __future__ import annotations

from typing import NamedTuple

from trm.config import (
    LATENT_DIM,
    MAX_SEQ_LEN,
    MAX_STEPS_LIMIT,
    MODEL_ARCH,
    NUM_BLOCKS,
    NUM_HEADS,
    PLAIN_LAYERS,
    POST_NORM,
    REFINER_ENCODER_LAYERS,
    SHARED_SLOTS,
    TIME_SIGNAL,
    VOCAB_SIZE,
    BATCH_SIZE,
    INFERENCE_DEPTH,
)

MIB = 1024 ** 2

# Dtype widths in bytes. Parameters are stored f32 (NNX's default, see trm/config.py) whatever
# the compute dtype is; the optimizer's widths are set in trm/train/optimizers.py.
F32, BF16, F16 = 4, 2, 2


class MeasuredPeak(NamedTuple):
    arch: str
    config: dict   # resolved-config values that must all match, plus "batch"
    gb: float
    source: str    # where the number came from, and what kind of reading it is


# Training peaks measured on the card, each for one exact config — the only numbers
# the report may quote beside the floor. A config not listed has no measured peak,
# and the report says so rather than quoting a neighbour's (#317).
_PLAIN_ARENA_PEAKS_MIB = {8: 4112, 9: 4437, 10: 4762}   # trm/config.py, the PLAIN_LAYERS note
MEASURED_PEAKS = (
    MeasuredPeak("refiner", {"dim": 960, "encoder_layers": 7, "batch": 1}, 5.0,
                 "runs/run_20260813_214725 (#157), nvidia-smi on the 6GB RTX 2060"),
    *(MeasuredPeak("plain", {"dim": 960, "num_heads": 15, "num_layers": layers, "post_norm": False,
                             "batch": 1}, mib * MIB / 1e9,
                   f"instruments.vram_headroom_smoke 2026-09-13: {mib} MiB allocator arena peak "
                   f"under cuda_async (trm/config.py, PLAIN_LAYERS)")
      for layers, mib in _PLAIN_ARENA_PEAKS_MIB.items()),
)


def _linear(in_features, out_features, bias=True):
    """nnx.Linear: a [in, out] kernel plus an [out] bias (use_bias defaults True)."""
    return in_features * out_features + (out_features if bias else 0)


def _rmsnorm(features):
    """nnx.RMSNorm: a scale vector. No bias — use_bias defaults False."""
    return features


def _embed(rows, dim):
    return rows * dim


# ── The two architectures' blocks ────────────────────────────────────────────

def _refiner_mlp_hidden(dim):
    """SwiGLU width for refiner.Block — 8/3 x dim rounded UP to a multiple of 64."""
    return ((int(8 * dim / 3) + 63) // 64) * 64


def _reasoner_mlp_hidden(dim):
    """SwiGLU width for layers.StandardReasoningBlock — the same 8/3 x dim, but
    snapped to a multiple of 256 by a float floor-division. Deliberately NOT the
    same expression as the refiner's (docs/design/plan-a.md: "Why the two arches
    don't share block code"), so it is transcribed separately here too."""
    return int(256 * ((dim * 8 / 3 + 255) // 256))


def _refiner_block(dim, num_heads):
    """refiner.Block: MHA (RoPE + q/k norms) + SwiGLU MLP + two RMSNorms."""
    head_dim = dim // num_heads
    attention = 4 * _linear(dim, dim) + 2 * _rmsnorm(head_dim)   # q, k, v, o
    hidden = _refiner_mlp_hidden(dim)
    mlp = 2 * _linear(dim, hidden) + _linear(hidden, dim)        # gate, up, down
    return attention + 2 * _rmsnorm(dim) + mlp


def _reasoner_block(dim, num_heads, num_groups):
    """layers.StandardReasoningBlock: GQA (K/V projected to num_groups heads
    only) + SwiGLU MLP + two RMSNorms."""
    head_dim = dim // num_heads
    kv_features = num_groups * head_dim
    attention = (
        _linear(dim, dim)                 # q_proj
        + 2 * _linear(dim, kv_features)   # k_proj, v_proj — grouped, hence narrow
        + _linear(dim, dim)               # o_proj
        + 2 * _rmsnorm(head_dim)          # q_norm, k_norm
    )
    hidden = _reasoner_mlp_hidden(dim)
    mlp = 2 * _linear(dim, hidden) + _linear(hidden, dim)
    return attention + 2 * _rmsnorm(dim) + mlp


def _plain_block(dim, num_heads, post_norm):
    """plain.py reuses refiner.Block; post-norm adds an RMSNorm on each residual branch."""
    return _refiner_block(dim, num_heads) + (2 * _rmsnorm(dim) if post_norm else 0)


# ── Config resolution ────────────────────────────────────────────────────────

def _plain_defaults():
    return {
        "dim": LATENT_DIM,
        "vocab_size": VOCAB_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": PLAIN_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
        "post_norm": POST_NORM,
    }


def _refiner_defaults():
    return {
        "dim": LATENT_DIM,
        "vocab_size": VOCAB_SIZE,
        "num_heads": NUM_HEADS,
        "encoder_layers": REFINER_ENCODER_LAYERS,
        "max_depth": MAX_STEPS_LIMIT,
        "max_seq_len": MAX_SEQ_LEN,
        "time_signal": TIME_SIGNAL,
        "use_gate": True,
    }


def _reasoner_defaults():
    return {
        "dim": LATENT_DIM,
        "vocab_size": VOCAB_SIZE,
        "num_heads": NUM_HEADS,
        "num_groups": NUM_HEADS // 4,   # the reasoner's own constant (trm/model/layers.py); pinned by the test
        "num_blocks": NUM_BLOCKS,
        "shared_slots": SHARED_SLOTS,
        "max_depth": MAX_STEPS_LIMIT,
        "max_seq_len": MAX_SEQ_LEN,
        "use_forget": True,
    }


def _resolve(arch, overrides):
    """Config for this arch, with overrides applied. An unknown key is an error:
    silently ignoring `n_heads=8` would report the default's numbers under the
    caller's name, which is the whole class of bug this module exists to end."""
    defaults = {"plain": _plain_defaults, "refiner": _refiner_defaults, "reasoner": _reasoner_defaults}
    if arch not in defaults:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(defaults)}")
    config = defaults[arch]()
    unknown = set(overrides) - set(config)
    if unknown:
        raise TypeError(
            f"unknown override(s) for arch {arch!r}: {sorted(unknown)}; "
            f"accepted: {sorted(config)}"
        )
    config.update(overrides)
    return config


# ── Parameters ───────────────────────────────────────────────────────────────

def param_breakdown(arch=MODEL_ARCH, **overrides):
    """Parameter count per group, as a dict in a sensible reading order.

    The groups mirror the real param tree (the test asserts exactly that), so
    "shared refine block" is *one physical block* — the refiner loops that same
    copy up to `max_depth` times, and looping costs compute, never weights.
    """
    config = _resolve(arch, overrides)
    dim = config["dim"]
    if arch == "plain":
        return {
            "embeddings & tied head": _embed(config["vocab_size"], dim),
            "blocks": config["num_layers"] * _plain_block(dim, config["num_heads"], config["post_norm"]),
            "heads & norms": _rmsnorm(dim),   # out_norm
        }
    if arch == "refiner":
        block = _refiner_block(dim, config["num_heads"])
        embeddings = _embed(config["vocab_size"], dim)
        if config["time_signal"] == "table":
            # The learned per-step embedding; the sinusoidal signal (the default)
            # is a fixed function of the step index and has no parameters at all.
            embeddings += _embed(config["max_depth"] + 1, dim)
        # time_norm + out_norm, plus the norm on the time signal when there is one.
        norms = 2 * _rmsnorm(dim) + (0 if config["time_signal"] == "none" else _rmsnorm(dim))
        gate = _linear(2 * dim, dim) if config["use_gate"] else 0
        return {
            "embeddings & tied head": embeddings,
            "encoder layers": config["encoder_layers"] * block,
            "shared refine block": block,
            "heads & norms": norms + gate,
        }

    block = _reasoner_block(dim, config["num_heads"], config["num_groups"])
    stack_layers = config["num_blocks"] // 2
    embeddings = (
        _embed(config["vocab_size"], dim)
        + _embed(config["max_depth"] + 1, dim)     # time_embed
        + config["shared_slots"] * dim             # shared_token, the slot prior
    )
    heads_and_norms = (
        _linear(2, dim)              # meta_proj
        + 5 * _rmsnorm(dim)          # seq, time, forget, time_signal, hunch
        + _linear(2 * dim, dim)      # hunch_gate
        + 1                          # raw_tau, a scalar
        + (_linear(2 * dim, dim) if config["use_forget"] else 0)   # forget_head
    )
    return {
        "embeddings & tied head": embeddings,
        "encoder stack": stack_layers * block,
        "decoder stack": stack_layers * block,
        "shared reasoning block": block,           # one physical block, looped
        "heads & norms": heads_and_norms,
    }


def total_params(arch=MODEL_ARCH, **overrides):
    return sum(param_breakdown(arch, **overrides).values())


def shared_block_group(arch=MODEL_ARCH):
    """Which group holds the one physically-stored, repeatedly-applied block, or
    None for the plain stack, which applies each of its blocks once."""
    return {"plain": None, "refiner": "shared refine block"}.get(arch, "shared reasoning block")


# ── VRAM ─────────────────────────────────────────────────────────────────────

TOTAL_KEY = "TOTAL (floor — activations excluded)"


def _remat_boundary_bytes(arch, config, batch, depth):
    """A LOWER BOUND on activation memory: the states jax.checkpoint keeps alive.

    Both arches rematerialize (refiner.py gates it to the GPU backend; the
    reasoner checkpoints its scan step), so a block's internals are recomputed
    in the backward instead of stored — but the *boundary* state entering each
    checkpointed region is kept. Those are countable. Everything else the
    backward touches (the recompute peak inside a region, XLA scratch, the f16
    casts of f32 weights, buffers held by other alive programs) is not modelled
    here, which is exactly why this is a bound and not an estimate.

    The grad step scores two prediction windows and holds both graphs at once
    (refiner.py, "the grad step holds TWO windows' graphs at once"), hence x2.
    """
    dim, seq = config["dim"], config["max_seq_len"]
    per_window_state = batch * seq * dim * F16
    if arch == "plain":
        # No remat in plain.py: every block's input is kept for the backward, and
        # that residual stream is only the smallest of what each block keeps.
        return 2 * config["num_layers"] * per_window_state
    if arch == "refiner":
        boundaries = config["encoder_layers"] + depth
        return 2 * boundaries * per_window_state
    # Reasoner: remat'd encoder and decoder stacks over the sequence, plus one
    # checkpointed scan step per depth over the (much smaller) slot state.
    stack_boundaries = 2 * (config["num_blocks"] // 2)
    slot_state = batch * config["shared_slots"] * dim * F16
    return 2 * (stack_boundaries * per_window_state + depth * slot_state)


def vram_estimate(mode, batch=BATCH_SIZE, depth=None, arch=MODEL_ARCH, **overrides):
    """VRAM line items in MiB. `mode` is "train" or "infer".

    Every entry is a real, named tensor whose size follows from a count and a
    dtype — except the remat-boundary line, which is a stated lower bound. The
    `TOTAL` entry is a FLOOR: the true peak is larger by the activation and
    allocator terms nothing analytic can pin down (~2.6 GB of the measured
    ~5.0 GB at the 4B refiner config). It is also the sum of the entries above it,
    so do not re-sum the dict.
    """
    if mode not in ("train", "infer"):
        raise ValueError(f"mode must be 'train' or 'infer', not {mode!r}")
    config = _resolve(arch, overrides)
    if depth is None:
        depth = config.get("max_depth", MAX_STEPS_LIMIT) if mode == "train" else INFERENCE_DEPTH
    params = total_params(arch, **overrides)

    lines = {"parameters (f32)": params * F32}
    if mode == "train":
        lines.update({
            "gradients (f32)": params * F32,
            "AdamW mu (bf16)": params * BF16,
            "AdamW nu (f32)": params * F32,
            "MultiSteps accumulated grads (f32)": params * F32,
            ("block input states (f16, lower bound)" if arch == "plain"
             else "remat boundary states (f16, lower bound)"):
                _remat_boundary_bytes(arch, config, batch, depth),
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


def measured_peak(arch=MODEL_ARCH, batch=BATCH_SIZE, **overrides):
    """The MeasuredPeak recorded for exactly this config, or None. Quoting a
    measured number against a different config would be the same category error
    this module is trying to stop."""
    config = {**_resolve(arch, overrides), "batch": batch}
    for peak in MEASURED_PEAKS:
        if peak.arch == arch and all(config.get(k) == v for k, v in peak.config.items()):
            return peak
    return None
