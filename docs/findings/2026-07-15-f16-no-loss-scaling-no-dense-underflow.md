# No-loss-scaling f16 shows no dense-kernel gradient underflow at this scale

Status: preliminary (init-adjacent smoke; the base run's early stretch is the confirming read — see #16)
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
