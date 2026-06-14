---
name: Optimization task
about: A performance / VRAM / throughput change with an expected gain and a validation plan.
title: "[opt] "
labels: ["optimization"]
---

## What
<!-- The change. Be specific: kernel, dtype, layout, caching, recompute, batch/seq shape, etc. -->

## Expected gain
<!-- Quantify the target. "Cuts peak VRAM by ~X GB" / "speeds up step by ~Y%". State how it was estimated. -->
- VRAM:
- Throughput / step time:

## Why it should work
<!-- Mechanism. What is the current bottleneck and how does this address it? -->

## Dependency / blocked-by
<!-- Other issues, library versions, or pinned deps this depends on. The deps are intentionally pinned — note any bump needed. -->

## Correctness risk
<!-- Could this change numerics or outputs? CPU-vs-GPU f16/f32 path differences are a known sensitivity here. -->

## Validation plan
<!-- How you confirm it is a win AND did not break anything. -->
- [ ] `JAX_PLATFORMS=cpu pytest tests/ -q` still green (incl. golden run)
- [ ] Measured before/after for the target metric
- [ ] Outputs unchanged (or change explained and accepted)
