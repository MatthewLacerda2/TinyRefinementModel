"""The terminal view of a run: what the model weighs, what it costs, how it is doing.

    PYTHONPATH=. python -m instruments.report                 # the latest run
    PYTHONPATH=. python -m instruments.report --log runs/run_20260813_214725/metrics.csv
    PYTHONPATH=. python -m instruments.report --model-only    # no run needed

Two rules this report follows, both of them reactions to how the previous one
misled us (#175, #177):

1. **Nothing is built to be measured.** Parameter counts and VRAM come from
   `instruments.model_stats`, which is arithmetic over `trm/config.py`. This
   process never constructs a model and never touches the GPU — the card is a
   serial queue and a report is not worth a slot in it.

2. **Every number says where it came from.**
     [measured]  read from what the run actually recorded (its CSV, its metadata)
     [sampled]   measured, but on a subsample — a few held-out rows, one
                 micro-step out of 128 — so it carries sampling noise
     [estimated] computed from config and recipe constants; nothing observed it

   The parameter count is marked [estimated] because this tool derived rather
   than observed it — it is nonetheless exact to the byte, and
   `tests/apparatus/test_model_stats.py` pins it against the instantiated model
   for both arches. The VRAM total is marked a FLOOR for the opposite reason:
   its terms are exact but incomplete, and saying "2.08 GB" against a real
   ~5.0 GB is precisely the mistake being retired here.
"""

from __future__ import annotations

import os

# A report must never wake the GPU: the card is committed to a live training run
# and a second JAX process on it is a real hazard. Set before anything imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse   # noqa: E402
import math       # noqa: E402

from trm.config import (           # noqa: E402
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    INFERENCE_DEPTH,
    LATENT_DIM,
    MAX_SEQ_LEN,
    MAX_STEPS_LIMIT,
    MODEL_ARCH,
    NUM_HEADS,
    REFINER_ENCODER_LAYERS,
    TIME_SIGNAL,
    TOKENS_PER_OPT_STEP,
    TRAIN_TOKEN_BUDGET,
    VOCAB_SIZE,
)
from instruments import model_stats, runlog   # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def _perplexity(ce):
    """exp(CE). Untrained CE can be ~11 nats; anything wilder overflows, and inf
    is the honest answer there rather than a crash."""
    try:
        return math.exp(ce)
    except OverflowError:
        return float("inf")


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _mean(values):
    return sum(values) / len(values) if values else None


# ── Model ────────────────────────────────────────────────────────────────────

def _arch_line(arch):
    if arch == "refiner":
        return (f"dim {LATENT_DIM}, {NUM_HEADS} heads (head_dim {LATENT_DIM // NUM_HEADS}), "
                f"{REFINER_ENCODER_LAYERS} encoder layers + 1 shared refine block looped "
                f"<= {MAX_STEPS_LIMIT}, vocab {VOCAB_SIZE:,}, seq {MAX_SEQ_LEN}, "
                f"time signal '{TIME_SIGNAL}'")
    return (f"dim {LATENT_DIM}, {NUM_HEADS} heads, encoder/decoder stacks + 1 shared "
            f"reasoning block looped <= {MAX_STEPS_LIMIT}, vocab {VOCAB_SIZE:,}, "
            f"seq {MAX_SEQ_LEN}")


def print_parameters(arch):
    breakdown = model_stats.param_breakdown(arch)
    total = sum(breakdown.values())
    print("\nPARAMETERS  [estimated — analytic over trm/config.py; pinned to the real")
    print("             model by tests/apparatus/test_model_stats.py]")
    shared = model_stats.shared_block_group(arch)
    for name, count in breakdown.items():
        note = f"  (1 physical copy, applied up to {MAX_STEPS_LIMIT}x)" if name == shared else ""
        print(f"  {name:<26} {count:>13,}   {100 * count / total:5.1f}%{note}")
    print(f"  {'':<26} {'-' * 13}")
    print(f"  {'total':<26} {total:>13,}   ({total / 1e6:.1f}M)")
    return total


def print_vram(arch, batch, train_depth, infer_depth):
    print("\nVRAM  [estimated — exact byte terms only; see the floor note]")
    for mode, depth, header in (
        ("train", train_depth, f"training  (batch {batch}, depth {train_depth})"),
        ("infer", infer_depth, f"inference (batch {batch}, depth {infer_depth})"),
    ):
        lines = model_stats.vram_estimate(mode, batch=batch, depth=depth, arch=arch)
        print(f"  {header}")
        for name, mib in lines.items():
            if name == model_stats.TOTAL_KEY:
                print(f"    {'':<42} {'-' * 13}")
                print(f"    {'FLOOR (activations excluded)':<42} {mib:9.1f} MiB "
                      f"({mib / 1024:.2f} GiB)")
            else:
                print(f"    {name:<42} {mib:9.1f} MiB")
    print("\n  The floor is not a peak. Excluded: activation recompute inside every")
    print("  remat'd region, XLA scratch, the f16 casts of f32 weights, and the")
    print("  allocator's own overhead across the ~28 compiled programs random-depth")
    print("  training keeps alive (see the 2026-08-14 BFC fragmentation finding).")
    if model_stats.measured_peak_applies(arch, batch=batch):
        floor_mib = model_stats.vram_estimate(
            "train", batch=batch, depth=train_depth, arch=arch)[model_stats.TOTAL_KEY]
        floor_gb = floor_mib * model_stats.MIB / 1e9
        print(f"  For this exact config the measured training peak is "
              f"~{model_stats.MEASURED_PEAK_GB:.1f} GB [measured]")
        print(f"    source: {model_stats.MEASURED_PEAK_SOURCE}")
        print(f"    so ~{model_stats.MEASURED_PEAK_GB - floor_gb:.1f} GB of the real cost "
              f"sits in terms this tool does not model.")


# ── Run ──────────────────────────────────────────────────────────────────────

def _tokens_per_opt_step(log):
    """The run's own tokens-per-opt-step, and whether it matches today's config.

    A run is a recipe, and BATCH_SIZE/ACCUMULATION_STEPS have been re-tuned since
    (#24). Scaling an old run's steps by today's constant would silently restate
    how much data it saw.
    """
    params = log.metadata.get("parameters", {})
    try:
        recorded = (int(params["ACCUMULATION_STEPS"]) * int(params["BATCH_SIZE"])
                    * 2 * int(params["MAX_SEQ_LEN"]))
    except (KeyError, TypeError, ValueError):
        return TOKENS_PER_OPT_STEP, True
    return recorded, recorded == TOKENS_PER_OPT_STEP


def _learning_rate(log, step):
    """The LR this run's schedule is at — rebuilt at the horizon the run recorded,
    not at whatever TRAIN_TOKEN_BUDGET happens to be set to in this shell."""
    from trm.train.schedules import DECAY_STEPS, build_learning_schedule

    decay_steps = log.metadata.get("parameters", {}).get("DECAY_STEPS") or DECAY_STEPS
    try:
        return float(build_learning_schedule(int(decay_steps))(step)), int(decay_steps)
    except (TypeError, ValueError):
        return None, None


def print_run(log):
    print(f"\nRUN  {log.run_id}")
    print(f"  source: {log.csv_path}")
    if not log.metrics:
        print("  no metric rows yet (header-only CSV) — nothing to summarize.")
        return
    if not log.metadata:
        print("  (no run_metadata.json — throughput, budget and LR fall back to config)")

    tokens_per_step, recipe_matches = _tokens_per_opt_step(log)
    step = log.last_step
    tokens = step * tokens_per_step

    print(f"  rows: {len(log.metrics):,}"
          + (f", {log.replayed_rows} replayed dropped" if log.replayed_rows else "")
          + (f", {log.torn_rows} torn dropped" if log.torn_rows else ""))
    if not recipe_matches:
        print(f"  ! this run used {tokens_per_step:,} tokens/opt-step; today's config says "
              f"{TOKENS_PER_OPT_STEP:,} — token counts below use the run's own recipe.")

    print(f"  step {step:,}  ->  {tokens:,} tokens ({tokens / 1e9:.3f}B)   [measured]")

    budget = log.metadata.get("parameters", {}).get("TRAIN_TOKEN_BUDGET") or TRAIN_TOKEN_BUDGET
    wall = log.wall_seconds
    throughput = tokens / wall if wall else None
    if budget:
        pct = 100 * tokens / budget
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  budget {budget / 1e9:.2f}B  [{bar}] {pct:5.1f}%   [measured]")

    print(f"  wall clock: {_duration(wall)}   [measured]")
    if throughput:
        print(f"  throughput: {throughput:,.0f} tok/s "
              f"({throughput * 86400 / 1e9:.2f}B/day)   [estimated — tokens / recorded session time]")
        if budget and tokens < budget:
            print(f"  ETA to budget: {_duration((budget - tokens) / throughput)}   [estimated]")

    lr, decay_steps = _learning_rate(log, step)
    if lr is not None:
        print(f"  LR now: {lr:.2e}  (cosine to step {decay_steps:,})   [estimated — the run's recorded schedule]")

    _print_losses(log)
    _print_diagnostics(log)


def _print_losses(log):
    _, ce = log.column("ce")
    if ce:
        last, best = ce[-1], min(ce)
        tail = _mean(ce[-100:])
        print(f"  train CE: {last:.4f} (ppl {_perplexity(last):,.1f})   [measured — one training batch]")
        print(f"    best {best:.4f} (ppl {_perplexity(best):,.1f}) · "
              f"last-{min(100, len(ce))} mean {tail:.4f} (ppl {_perplexity(tail):,.1f})   [measured]")
    if log.has("val_ce"):
        steps, val = log.column("val_ce")
        print(f"  val CE:   {val[-1]:.4f} (ppl {_perplexity(val[-1]):,.1f}) at step {steps[-1]:,}   "
              f"[sampled — the held-out probe's fixed rows]")
        print(f"    best {min(val):.4f} (ppl {_perplexity(min(val)):,.1f}) over {len(val)} evals   [sampled]")
    else:
        print("  val CE:   not logged in this run")


# Columns worth a line when present, with what they actually are. A column that
# is empty is omitted entirely — it is arch-optional (#105), not zero.
_DIAGNOSTICS = (
    ("grad_norm_avg", "grad norm", "sampled",
     "raw per-micro-step, BEFORE clip_by_global_norm(1.0) — not the clipped step size"),
    ("depth_avg", "sampled depth", "sampled", "uniform in [1, MAX_STEPS_LIMIT] per micro-step"),
    ("zero_frac_dense_max", "max dense zero-grad frac", "sampled", "f16 underflow watch (#82)"),
    ("out_entropy", "output entropy", "sampled", "nats, window 2"),
    ("logz_mean", "mean log Z", "sampled", "logit-scale thermometer (#80)"),
    ("max_abs_logit", "max |logit|", "sampled", ""),
    ("temporal_drift", "slot temporal drift", "sampled", "reasoner only"),
    ("avg_forget_cost", "forget cost", "sampled", "reasoner only"),
    ("diversity_loss", "diversity loss", "sampled", "reasoner only"),
    ("tau", "tau", "measured", "reasoner only"),
)


def _print_diagnostics(log):
    present = [(col, label, tag, note) for col, label, tag, note in _DIAGNOSTICS if log.has(col)]
    absent = [col for col, *_ in _DIAGNOSTICS if not log.has(col)]
    if present:
        print("  diagnostics (last value):")
        for col, label, tag, note in present:
            _, values = log.column(col)
            if log.is_constant(col, 0.0):
                # Written as a literal zero by a model that did not measure it —
                # pre-#105 placeholder, not a reading. Say so instead of showing
                # a number that means nothing.
                print(f"    {label:<26} {'0 throughout':>12}   [not measured — a pre-#105 "
                      f"placeholder column, not a reading]")
                continue
            suffix = f" — {note}" if note else ""
            print(f"    {label:<26} {values[-1]:>12.5g}   [{tag}]{suffix}")
    if absent:
        print(f"  not measured by this architecture: {', '.join(absent)}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default=None,
                        help="metrics.csv or a run dir (default: the latest run under runs/)")
    parser.add_argument("--arch", default=MODEL_ARCH, choices=("refiner", "reasoner"),
                        help=f"architecture to size (default: MODEL_ARCH={MODEL_ARCH})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size for the VRAM lines")
    parser.add_argument("--train-depth", type=int, default=MAX_STEPS_LIMIT)
    parser.add_argument("--infer-depth", type=int, default=INFERENCE_DEPTH)
    parser.add_argument("--model-only", action="store_true",
                        help="skip the run summary (no metrics.csv needed)")
    args = parser.parse_args()

    print(RULE)
    print(f"TinyRefinementModel — report   ·   architecture '{args.arch}'")
    print(f"  {_arch_line(args.arch)}")
    print(RULE)

    print_parameters(args.arch)
    print_vram(args.arch, args.batch, args.train_depth, args.infer_depth)

    if not args.model_only:
        try:
            log = runlog.load(args.log)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nRUN  unavailable: {exc}")
        else:
            print_run(log)

    print(f"\n{THIN}")
    print("[measured] recorded by the run · [sampled] measured on a subsample · "
          "[estimated] derived from constants")
    print("Accumulation is 1 opt step = "
          f"{ACCUMULATION_STEPS} micro-steps x {BATCH_SIZE} x 2 windows x {MAX_SEQ_LEN} tokens.")


if __name__ == "__main__":
    main()
