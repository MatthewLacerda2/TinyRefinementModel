# Identical per-step supervision without slots collapses at the composition point — the offload is load-bearing, not just the grade

Status: toy-proof (the #79 gate; completes the #38 attribution)
Date: 2026-07-07
Commit: 3accd14  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python scratchpad_harness.py --arms serial,densedepth,densedepth_tied --seeds 0,1,2 --K 4 --m 7 --steps 2500`

## Setup

The last open attribution question from #38: the depthonly control removed two
things at once (slots AND intermediate supervision), and #67 then proved the
grade is *necessary* — but the per-pass finding (2026-07-05) showed a densely
supervised slotless refiner solving non-commutative composition on statetrack,
raising the possibility that the grade is also *sufficient* and the slots are
just a convenient place to attach it. #79 fills the missing cell: the
**densedepth** arm is the CausalRefiner at depth K=4 given the serial arm's
exact supervision schedule — pass k's state at the answer position graded
against sub-result r_k through a dedicated linear head (the same grade path
shape as the serial arm's `slot_readout`), λ=1.0, plus final-answer CE — but
no slots: intermediate results must live in the recurrence's own hidden state.
A second variant (**densedepth_tied**) grades through the tied LM head
instead, as a robustness check on the grade-head choice. Wiring guard
(tests/test_scratchpad_harness.py) proves the intermediate grades are a live
gradient path into the shared refine block and encoder.

Same protocol as the #38/#62/#67 gates: affine chain K=4, m=7, dim 64, 2500
steps, held-out 4096, seeds {0,1,2}, matched pairs (same seed ⇒ same pools).
Param caveat stated in advance on the issue: removing the slots changes the
param tree (0.17M vs 0.22M) — inherent to the variable under test, as with the
#38 depthonly control. Pre-registered: densedepth within 2σ_pooled of serial →
the grade carried the win; serial ahead by ≥2σ_pooled → the offload is real.
The serial control was re-run on this machine and reproduced the #38 numbers
exactly (0.9988 / 0.9790 / 0.9988).

## Evidence

Held-out final-answer accuracy (chance = 1/7 ≈ 0.143):

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|
| serial (graded, #38 control) | 0.9988 | 0.9790 | 0.9988 | 0.9922 | 0.011 |
| densedepth (dedicated head) | 0.2703 | 0.1323 | 0.1655 | **0.1894** | 0.072 |
| densedepth_tied (tied head) | 0.1443 | 0.1411 | 0.1633 | 0.1496 | 0.012 |

Serial − densedepth = +0.803 ≈ **15.6× σ_pooled** (bar was 2σ; the tied
variant is ~72σ behind). The verdict lands at the pre-registered "offload is
real" extreme, and the grade-head choice does not change it.

Two readings sharpen the number:

1. **densedepth reproduces the parallel arm's regime, not depthonly's.** Its
   mean (0.1894) sits on top of the #38 parallel arm (0.1878 ± 0.030), and so
   does its per-step signature — r₁ and r₂ perfect (shallow token functions),
   collapse exactly where composing a prior *latent* result becomes
   unavoidable:

   | arm | step 1 (r₁) | step 2 | step 3 | step 4 |
   |---|---|---|---|---|
   | serial | 1.000 | 0.999 | 0.997 | 0.991 |
   | parallel (#38) | 1.000 | 0.996 | ~0.29 | ~0.15 |
   | densedepth | 1.000 | 0.998 | 0.15–0.78 | 0.13–0.31 |

   Identical supervision without ordered external slots lands exactly where
   slots-without-order landed. The common factor in both failures is the
   missing *committed, re-readable intermediate result*.

2. **Why this doesn't contradict the per-pass statetrack result.** There,
   every pass was graded against the *same* final target at every position —
   refinement toward a fixed point. Here each pass must produce a *different*
   value r_k and the next pass must build on a state the grade just pulled
   elsewhere: the recurrence has to overwrite its own working memory while
   still using it. The scratchpad's append-only frozen slots remove exactly
   that conflict — r_k stays parked where later computation can re-read it.
   That is the offload hypothesis, now with one-variable-adjacent support.

Combined with #67, the #38 win is now fully decomposed: the grade without the
structure fails (finalonly, 0.134), the structure without the grade fails
(#67's bound), and the grade with structure-but-no-slots fails (this). Only
grade + ordered external slots solves the chain. Neither ingredient is
decorative; the supervision *teaches* the decomposition and the slots *carry*
it (#62).

## Limitations

- Toy scale, one task family, K=4, one slotless architecture — this kills "the
  grade alone would have won" for *the refiner we'd ship*, not for every
  conceivable slotless design (e.g. Huginn-style input re-injection, or a
  recurrence with more state headroom, might absorb the moving target).
- The param trees differ (0.17M vs 0.22M) and the architectures differ beyond
  the slots (self-attention recurrence + gate vs cross-attention writes) —
  inherent to the variable, same status as the #38 depthonly control. The
  clean matched pair remains serial-vs-parallel; this result is the
  supervision-matched *complement*, not a substitute for it.
- densedepth's seed spread is wide (σ = 0.072; seed 0 reached 0.78 on step 3
  before dying at step 4) — the slotless arm is *trying* to form the chain and
  cannot hold it, which is consistent with the interference story but not
  proof of the mechanism.
- In densedepth_tied the final pass's grade and the final-answer CE score the
  *same* logits, so the final target is effectively weighted 2× there (serial
  and densedepth grade it through two different heads). A schedule asymmetry
  worth noting for exactness; the variant sits at chance either way.
