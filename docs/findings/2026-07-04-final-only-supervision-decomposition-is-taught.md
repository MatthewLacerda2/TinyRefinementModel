# Final-answer-only supervision collapses the serial scratchpad to chance — the decomposition is taught by the grade, not induced by structure

Status: toy-proof (the #67 gate; bounds the #38 interpretation)
Date: 2026-07-04
Commit: 452b2a5  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python scratchpad_harness.py --arms serial,finalonly --seeds 0,1,2 --K 4 --m 7 --steps 2500`

## Setup

The missing cell in the #38 design matrix: serial (#38) had order + per-slot
grade and won; parallel had the grade but no order and failed; depthonly had
neither and sat at chance. #67 asks whether the grade or the structure carries
the win — the `finalonly` arm keeps the serial wiring and parameters exactly
(ordered write-once slots, slot k reads tokens + slots < k) but detaches the
slot grade behind a stop-gradient, so final-answer CE is the model's only
teacher. The slot readout still trains — on stop-gradiented slots — so it acts
as a pure linear probe of what each slot carries without being able to teach
the slots (wiring guard: tests/test_scratchpad_harness.py proves the slot CE
has exactly zero gradient into everything upstream of the readout head).

Same protocol as the #38 gate: affine chain K=4, m=7, dim 64, 2500 steps,
held-out 4096, seeds {0,1,2}, matched pairs (same seed ⇒ same pools; serial
control re-run on the same machine and reproducing the #38 numbers exactly).
Pre-registered on the issue: within 2σ_pooled of the graded serial (~0.99) →
**emergent**; collapse toward the depthonly/chance regime (~0.14) → **taught**;
clearly between → partial, report where the slot chain breaks.

## Evidence

Held-out final-answer accuracy:

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|
| serial (graded, #38 control) | 0.9988 | 0.9790 | 0.9988 | 0.9922 | 0.011 |
| finalonly | 0.1299 | 0.1372 | 0.1353 | **0.1341** | 0.004 |

Chance = 1/7 ≈ 0.143; the #38 depthonly control was 0.1397 ± 0.010. The verdict
is not close: finalonly sits 0.858 below the graded serial (~101× σ_pooled =
0.0085, bar was 2σ) and *inside* the depthonly/chance regime (0.7σ below its
mean). **Taught**, at the pre-registered extreme.

The probe readout shows where the chain never forms:

| arm | slot 1 (r₁) | slot 2 | slot 3 | slot 4 |
|---|---|---|---|---|
| serial (graded) | 1.000 | 0.999 | 0.997 | 0.991 |
| finalonly (probe) | 0.31–0.43 | 0.14 | 0.14 | 0.14 |

Slot 1 is weakly decodable above chance (r₁ = b₁, a shallow copy the write
block picks up incidentally); slots 2–4 carry nothing a linear probe can find.
Without the grade the model doesn't build a partial chain that a final-only
loss then fails to extend — it builds *no* chain, and the final answer never
leaves chance. Structure (order + write-once + capacity) supplied the road;
nothing drove on it.

## What this bounds

- The #38 win decomposes as: structure is *necessary* (parallel failed with the
  grade) but not *sufficient* (finalonly fails with the structure). The
  per-slot grade is load-bearing — it is what pulls the model onto the serial
  decomposition, and the slots then genuinely carry it (#62).
- For the LM-scale version this is the expensive branch: natural compression
  will not self-organize just from giving the model ordered write-once slots;
  it needs per-step targets (synthetic intermediate supervision or an
  equivalent curriculum). That cost is now a known input to the bet, before
  anything was built at scale.

## Limitations

- Toy scale, one task family, K=4, 2500 steps — a longer budget or an
  annealed grade (start supervised, decay λ) might still find the chain; this
  result kills "it emerges for free", not "it can never be induced".
- The probe is linear and trained online; a stronger nonlinear probe could in
  principle find structure a linear one misses, but the final answer sitting
  at chance says there was nothing usable to find.
- The optimization story (final-only gradient signal through 4 chained
  cross-attention writes is weak/noisy) is a plausible mechanism but untested
  here; it does not change the practical bound.
