# No-loss-scaling f16 shows no dense-kernel gradient underflow at this scale

Status: retracted (2026-09-13 addendum) — the base run this entry named as its confirming read refuted it at opt step 11,140 (#199)
Date: 2026-07-15
Commit: aa36bb5  Run: GPU smoke (no runs/ id)  Measured with: `PYTHONPATH=. python tools/smoke_refiner_gpu.py` (RTX 2060, f16 compute)

## Setup

The dtype policy trains in f16 compute *without* loss scaling — a deliberate
deviation from the standard mixed-precision recipe, adopted because the 2060 has
no bf16 tensor cores. #82 built the instrument to measure the deviation instead
of assuming it: fraction of exactly-zero gradient entries per top-level param
group, sampled by the trainer every logging block (`zero_frac_dense_max` in
`metrics.csv`). Pre-registered decision rule: dense max < 0.05, no upward trend.
Model: 52M-param CausalRefiner (Dim=512), optimizer resident, depths 1/4/8.

## Evidence

Dense-kernel zero fractions across all grad steps and depths, production f16 on
device:

| depth | dense max | embed | encoder | refine_block | time_embed (excluded) |
|---|---|---|---|---|---|
| 8 (steps 1–3) | 0.0001 | 0.000 | 0.000 | 0.000 | 0.111 |
| 4 | 0.0001 | 0.000 | 0.000 | 0.000 | 0.556 |
| 1 | 0.0003 | 0.000 | 0.000 | 0.000 | 0.889 |

**Dense max 0.0003 vs the 0.05 bar — two orders of magnitude of headroom, no
trend.** The `time_embed` readings are the pinned structural caveat (rows for
unsampled depths get no gradient: 7/9 at depth 2, etc.), not underflow, and are
excluded from the decision scalar.

Underflow is *measurable but negligible*: on the f16 lane the unit test's
encoder/refine_block structural-zero fractions diverge by ~4e-5 of the group —
a handful of entries the f32 CPU lane keeps as tiny non-zeros round to exact
zero in f16. The instrument resolves it; it is far below any level that moves
training.

## Limitations

- Init-adjacent measurement (few optimizer steps, near-init weights). Gradient
  magnitudes shrink as the loss falls, so underflow risk *grows* over training —
  the early stretch of the base run (#16) reads `zero_frac_dense_max` in
  `metrics.csv` as the confirming measurement; a rise reopens the loss-scaling
  question (optax loss-scaling adoption is the pre-named fallback).
- 52M-param smoke config, not the full base-run config; depth ≤ 8, batch 1.
- Novelty verdict: uncertain → treated as novel per rule 5. The *technique* is
  settled (zero-fraction monitoring is standard); the *result* — that f16
  without loss scaling holds at this scale/architecture with tied embeddings —
  is a measured recipe fact for this repo, cheap to delete later if literature
  surfaces it.

## Addendum 2026-09-13 — retracted by the run it deferred to

This entry's own Limitations named the confirming measurement: the base run's
`zero_frac_dense_max`, with loss scaling as the pre-named fallback if it rose. It
rose. On 2026-08-18, `run_20260813_214725` (#157) hit f16 underflow at opt step
11,140: the refiner's `gate` went from a 0.002 zero-gradient fraction to **1.000**
at step 11,075, `refine_block` starved at 26–71%, and 65 steps later the output
distribution was uniform (val CE 10.82 against ln(50304) = 10.83). Nothing was
non-finite at any point (#199).

The init-adjacent numbers above stand as measured; the conclusion they were read
to support — that no loss scaling is needed at this scale — does not. Gradient
magnitudes shrink as the loss falls, exactly as the Limitations warned. The
remedy, dynamic loss scaling (`trm/train/loss_scale.py`), has been live since
#199 and was part of the champion's recipe. The instrument that recorded the
failure was the one this entry introduced (#82); what was missing was anything
wired to act on it. Also since then: the zero-fraction reading is now taken on
the gradient the optimizer applies, not one micro-step's (#191).
