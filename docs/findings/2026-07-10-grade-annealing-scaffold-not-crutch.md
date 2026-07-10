# Annealed to zero mid-training, the per-slot grade proves to be a scaffold — the chain survives on final-answer loss alone, but removal costs stability

Status: toy-proof (the #73 gate; completes the #67 story)
Date: 2026-07-10
Commit: 243b8b1  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python scratchpad_harness.py --arms serial,annealed --seeds 0,1,2 --K 4 --m 7 --steps 2500`

## Setup

#67 proved the per-slot grade is load-bearing: final-answer-only supervision
from scratch never forms the chain (0.1341 ± 0.004 vs the graded 0.9922 ± 0.011).
That kills "the decomposition emerges for free" but says nothing about whether
the grade must stay on forever. This ablation anneals it away mid-run: the
`annealed` arm is the graded serial arm with λ_slot fully on for steps 0–1000,
linearly decayed 1000–1500, and exactly zero for 1500–2500 — a full 1000
final-only steps to expose decay. One variable, matched pairs, seeds {0,1,2},
same protocol as #38/#67 (affine chain K=4, m=7, dim 64, held-out 4096). The
re-run graded control reproduced the #38/#67 numbers *exactly* (0.9988 /
0.9790 / 0.9988), so the harness refactor changed nothing. Held-out accuracy
is measured at the grade-off step (1500) and at the end, for both arms.

Pre-registered on the issue: annealed within 2σ_pooled of its matched graded
control → **scaffold**; ≥2σ below *and* trending down across the grade-free
stretch → **crutch**; ≥2σ below but flat → **frozen-but-stable**.

## Evidence

Held-out final-answer accuracy:

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|
| serial (graded throughout, control) | 0.9988 | 0.9790 | 0.9988 | 0.9922 | 0.011 |
| annealed @ step 1500 (grade just off) | 0.6902 | 0.9919 | 0.9915 | 0.8912 | — |
| annealed @ step 2500 (1000 grade-free steps) | 0.9524 | 0.9963 | 0.8494 | **0.9327** | 0.075 |

Δ vs control = 0.060 < 2σ_pooled = 0.108 → the pre-registered **scaffold**
verdict. The crutch criterion fails on both of its clauses: the gap is under
2σ, and the grade-free trend is not a decay — it is *mixed* (+0.26, +0.00,
−0.14 across seeds). Nothing returns to the chance regime: the worst seed ends
at 0.849, six-fold above the 0.134 final-only-from-scratch floor (#67).

Two reads underneath the headline number:

- **Final-only CE can refine a formed chain it could never have built.** Seed 0
  sat at 0.690 when the grade switched off and climbed to 0.952 across the
  grade-free stretch — the exact loss that stays at chance from scratch (#67)
  drives real improvement once the grade has nucleated the decomposition. The
  asymmetry is clean: per-step supervision creates the road; final-answer loss
  can then drive on it.
- **But the grade was also a stabilizer.** Seed variance grows ~7× when it goes
  (σ 0.011 → 0.075), and seed 2 genuinely decayed, 0.9915 → 0.8494, while its
  matched control held 0.999. One seed in three losing ~0.14 is not a collapse,
  but it is not the control's flat line either.

Slot probes (read through the grade head, frozen once λ = 0): at the cut the
annealed chains look like the control's (seed 0's slot 4 still forming at
0.572); at the end the frozen head reads 0.74–1.00. Seed 2's end probes dropped
roughly uniformly (0.80–0.85) rather than deepest-first as the issue guessed.

## What this bounds

- The LM-scale cost estimate collapses: per-step targets are a **warm-up
  budget, not a run-long budget**. The bet's expensive branch (#67) shrinks to
  a nucleation phase plus an anneal.
- Budget for the stability wrinkle: removing the grade entirely trades a small
  mean gap for a large variance. The natural follow-ups (filed as the
  pre-registration requires): how early can the anneal start, and does a small
  floor (λ ≈ 0.1) keep the variance of the graded arm at zero marginal cost.

## Limitations

- Toy scale, one task family, 3 seeds — σ_annealed estimated from 3 points is
  itself noisy; the scaffold verdict clears the pre-registered bar but the bar
  is wide because the annealed arm's own spread sets it.
- The end-of-run slot probe is read through a head frozen at step 1500; slot
  representations that drifted while staying informative under-read. Final
  accuracy is the load-bearing metric; the probes are directional only.
- 1000 grade-free steps can't distinguish "seed 2 found a stable plateau at
  0.85" from "seed 2 decays slowly"; a longer grade-free tail would resolve it.
