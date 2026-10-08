"""The terminal view of a run: what the model weighs, what it costs, how it is doing.

    PYTHONPATH=. python -m instruments.report                 # the latest run
    PYTHONPATH=. python -m instruments.report --log runs/<run>/metrics.csv
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
   `tests/apparatus/test_model_stats.py` pins it against the instantiated model.
   The VRAM total is marked a FLOOR for the opposite reason:
   its terms are exact but incomplete, and saying "2.08 GB" against a real
   ~5.0 GB is precisely the mistake being retired here.
"""

from __future__ import annotations

import os

# A report must never wake the GPU: the card is committed to a live training run
# and a second JAX process on it is a real hazard. Set before anything imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import math

from instruments import model_stats, runlog
from instruments.invariants import clean_column, describe, suspect_rows
from trm.config import VOCAB_SIZE
from trm.settings import CONFIG

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "parameters": ("estimated", "arithmetic over config, pinned to the instantiated model by tests/apparatus/test_model_stats.py"),
    "VRAM floor": ("estimated", "exact byte terms only; a FLOOR, never a peak"),
    "run metrics": ("measured", "each line carries its own [measured]/[sampled]/[estimated] tag"),
}

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

def _model_line():
    return (f"dim {CONFIG.LATENT_DIM}, {CONFIG.NUM_HEADS} heads (head_dim {CONFIG.LATENT_DIM // CONFIG.NUM_HEADS}), "
            f"{CONFIG.PLAIN_LAYERS} causal blocks, vocab {VOCAB_SIZE:,}, seq {CONFIG.MAX_SEQ_LEN}")


def print_parameters():
    breakdown = model_stats.param_breakdown()
    total = sum(breakdown.values())
    print("\nPARAMETERS  [estimated — analytic over trm/config.py; pinned to the real")
    print("             model by tests/apparatus/test_model_stats.py]")
    for name, count in breakdown.items():
        print(f"  {name:<26} {count:>13,}   {100 * count / total:5.1f}%")
    print(f"  {'':<26} {'-' * 13}")
    print(f"  {'total':<26} {total:>13,}   ({total / 1e6:.1f}M)")
    return total


def print_vram(batch):
    print("\nVRAM  [estimated — exact byte terms only; see the floor note]")
    for mode, header in (("train", f"training  (batch {batch})"), ("infer", f"inference (batch {batch})")):
        lines = model_stats.vram_estimate(mode, batch=batch)
        print(f"  {header}")
        for name, mib in lines.items():
            if name == model_stats.TOTAL_KEY:
                print(f"    {'':<42} {'-' * 13}")
                print(f"    {'FLOOR (activations excluded)':<42} {mib:9.1f} MiB "
                      f"({mib / 1024:.2f} GiB)")
            else:
                print(f"    {name:<42} {mib:9.1f} MiB")
    print("\n  The floor is not a peak. Excluded: the activations inside every block,")
    print("  XLA scratch, the f16 casts of f32 weights, and the allocator's own overhead")
    print("  (see the 2026-08-14 BFC fragmentation finding).")
    peak = model_stats.measured_peak(batch=batch)
    if peak is None:
        print("  No measured training peak on record for this config — the floor is all there is.")
    else:
        floor_mib = model_stats.vram_estimate("train", batch=batch)[model_stats.TOTAL_KEY]
        floor_gb = floor_mib * model_stats.MIB / 1e9
        print(f"  For this exact config the measured training peak is ~{peak.gb:.1f} GB [measured]")
        print(f"    source: {peak.source}")
        if peak.reading == "arena":
            # An allocator reading stops at the pool's edge: the CUDA context and the
            # driver's compiled-graph buffers sit outside it (#160), so the card holds more.
            print(f"    so ~{peak.gb - floor_gb:.1f} GB of the arena alone sits in terms this tool "
                  f"does not model — and the card holds more than the arena (CUDA context, "
                  f"driver graph buffers), so the real cost is higher still.")
        else:
            print(f"    so ~{peak.gb - floor_gb:.1f} GB of the whole card's cost "
                  f"sits in terms this tool does not model.")


# ── Run ──────────────────────────────────────────────────────────────────────

def _tokens_per_opt_step(log):
    """The run's own tokens-per-opt-step, and whether it matches today's config.

    A run is a recipe, and BATCH_SIZE/ACCUMULATION_STEPS have been re-tuned since
    (#24). Scaling an old run's steps by today's constant would silently restate
    how much data it saw.
    """
    recorded = runlog.recorded_tokens_per_opt_step(log.params)
    if recorded is None:
        return CONFIG.TOKENS_PER_OPT_STEP, True
    return recorded, recorded == CONFIG.TOKENS_PER_OPT_STEP


def _learning_rate(log, step):
    """The LR this run's schedule is at — rebuilt at the horizon the run recorded,
    not at whatever TRAIN_TOKEN_BUDGET happens to be set to in this shell."""
    from trm.settings import CONFIG
    from trm.train.schedules import Schedules, build_schedule

    params = log.params
    decay_steps = params.get("DECAY_STEPS") or Schedules.of(CONFIG).decay_steps
    # The run's own warmup, peak and schedule shape (#305, #386); the process's only
    # for runs that predate recording them, all of which were cosine.
    shape = {"warmup_steps": int(params.get("WARMUP_STEPS") or CONFIG.WARMUP_STEPS),
             "peak_lr": float(params.get("PEAK_LR") or CONFIG.PEAK_LR)}
    kind = params.get("LR_SCHEDULE") or "cosine"
    if kind == "wsd":
        shape["decay_fraction"] = float(params.get("WSD_DECAY_FRACTION") or 0.2)
        shape["decay_start"] = params.get("WSD_DECAY_START")
    try:
        return float(build_schedule(int(decay_steps), kind, **shape)(step)), int(decay_steps)
    except (TypeError, ValueError):
        return None, None


def _print_suspect(suspect):
    """Say plainly that some rows are excluded, and why.

    Silently dropping them would be worse than reporting the bad numbers: a
    reader who is not told cannot tell a filtered series from a clean run, and
    the whole point of the check is to make a corrupted accumulator visible.
    """
    if not suspect:
        return
    print(f"\n  ! {len(suspect)} row(s) excluded from every figure below — an "
          f"invariant that cannot be violated by any model was violated, so the "
          f"row's accumulator is wrong and nothing on it is trustworthy:")
    for line in describe(suspect):
        print(f"      {line}")
    print("    (the row is still in metrics.csv; see #194 for the resume artifact "
          "that causes this, and #195 for the check)")


def print_run(log):
    print(f"\nRUN  {log.run_id}")
    print(f"  source: {log.csv_path}")
    if not log.metrics:
        print("  no metric rows yet (header-only CSV) — nothing to summarize.")
        return
    if log.metadata is None:
        print("  (no run_metadata.json — throughput, budget and LR fall back to config)")

    tokens_per_step, recipe_matches = _tokens_per_opt_step(log)
    step = log.last_step
    tokens = step * tokens_per_step

    print(f"  rows: {len(log.metrics):,}"
          + (f", {log.replayed_rows} replayed dropped" if log.replayed_rows else "")
          + (f", {log.torn_rows} torn dropped" if log.torn_rows else ""))
    if not recipe_matches:
        print(f"  ! this run used {tokens_per_step:,} tokens/opt-step; today's config says "
              f"{CONFIG.TOKENS_PER_OPT_STEP:,} — token counts below use the run's own recipe.")

    print(f"  step {step:,}  ->  {tokens:,} tokens ({tokens / 1e9:.3f}B)   [measured]")

    budget = log.params.get("TRAIN_TOKEN_BUDGET") or CONFIG.TRAIN_TOKEN_BUDGET
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

    suspect = suspect_rows(log)
    _print_suspect(suspect)
    _print_losses(log, suspect)
    _print_diagnostics(log)


def _print_losses(log, suspect=None):
    _, ce = clean_column(log, "ce", suspect)
    if ce:
        last, best = ce[-1], min(ce)
        tail = _mean(ce[-100:])
        print(f"  train CE: {last:.4f} (ppl {_perplexity(last):,.1f})   [measured — one training batch]")
        print(f"    best {best:.4f} (ppl {_perplexity(best):,.1f}) · "
              f"last-{min(100, len(ce))} mean {tail:.4f} (ppl {_perplexity(tail):,.1f})   [measured]")
    if log.has("val_ce"):
        steps, val = clean_column(log, "val_ce", suspect)
        print(f"  val CE:   {val[-1]:.4f} (ppl {_perplexity(val[-1]):,.1f}) at step {steps[-1]:,}   "
              f"[sampled — the held-out probe's fixed rows]")
        print(f"    best {min(val):.4f} (ppl {_perplexity(min(val)):,.1f}) over {len(val)} evals   [sampled]")
    else:
        print("  val CE:   not logged in this run")


# Columns worth a line when present, with what they actually are. A column that
# is empty is not a zero (#105): it is named as not logged.
_DIAGNOSTICS = (
    ("applied_grad_norm", "grad norm (applied)", "measured",
     "norm of the window mean the clip sees; above 1.0 the clip, not the LR, sets the step (#180)"),
    ("grad_norm_avg", "  ...per micro-step", "sampled",
     "raw per-micro-step, BEFORE clip_by_global_norm(1.0) — not comparable to the clip"),
    ("applied_zero_frac_dense_max", "max dense zero-grad frac", "measured",
     "the gradient the optimizer applies: f16 underflow watch (#82, #191)"),
    ("zero_frac_dense_max", "  ...one micro-step", "sampled",
     "one micro-step's grads; per-draw artifacts that never reach the weights (#191)"),
    ("out_entropy", "output entropy", "sampled", "nats, window 2"),
    ("logz_mean", "mean log Z", "sampled", "logit-scale thermometer (#80)"),
    ("max_abs_logit", "max |logit|", "sampled", ""),
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
        print(f"  {runlog.NOT_LOGGED}: {', '.join(absent)}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default=None,
                        help="metrics.csv or a run dir (default: the latest run under runs/)")
    parser.add_argument("--batch", type=int, default=CONFIG.BATCH_SIZE, help="batch size for the VRAM lines")
    parser.add_argument("--model-only", action="store_true",
                        help="skip the run summary (no metrics.csv needed)")
    args = parser.parse_args()

    print(RULE)
    print("TinyRefinementModel — report")
    print(f"  {_model_line()}")
    print(RULE)

    print_parameters()
    print_vram(args.batch)

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
          f"{CONFIG.ACCUMULATION_STEPS} micro-steps x {CONFIG.BATCH_SIZE} x 2 windows x {CONFIG.MAX_SEQ_LEN} tokens.")


if __name__ == "__main__":
    main()
