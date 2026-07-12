# Anneal the grade to a small floor, not to zero — and the anneal can't start at a fixed time, only once the chain has actually formed

Status: toy-proof (the #95 gate; the recipe #73's scaffold verdict asked for)
Date: 2026-07-12
Commit: 6f05ef0  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python scratchpad_harness.py --arms annealed,annealed@0.1,annealed@0.2,annealed@0.3,annealed@0.4f0.1 --seeds 0,1,2 --K 4 --m 7 --steps 2500`

## Setup

#73 left two questions: how early can the grade anneal start (the warm-up is
the LM-scale cost), and does annealing to a small residual λ instead of zero
buy back the stability that full removal cost (σ grew ~7×). The schedule is
now parameterized: `annealed@X` starts the decay at fraction X of training
(decay window fixed at 20% of the budget), `fY` decays to floor λ=Y and holds.
Same protocol as #38/#67/#73; the graded control's numbers are reused from
#73 (reproduced exactly there, same determinism). The plain `annealed` arm was
re-run as a replay check and reproduced #73's every number **bitwise**,
proving the parameterization refactor inert.

## Evidence

Held-out final-answer accuracy (control from #73: 0.9922 ± 0.011; chance 0.143):

| arm | grade off at | seed 0 | seed 1 | seed 2 | mean | σ | collapsed seeds |
|---|---|---|---|---|---|---|---|
| annealed@0.1 | step 750 | 0.1421 | 0.1401 | 0.1421 | 0.1414 | 0.001 | 3/3 |
| annealed@0.2 | step 1000 | 0.1548 | 0.2659 | 0.9937 | 0.4715 | 0.456 | 2/3 |
| annealed@0.3 | step 1250 | 0.1992 | 0.9895 | 0.9973 | 0.7287 | 0.459 | 1/3 |
| annealed (=@0.4, #73 replay) | step 1500 | 0.9524 | 0.9963 | 0.8494 | 0.9327 | 0.075 | 0/3 |
| **annealed@0.4f0.1 (floor)** | step 1500 | 0.9885 | 0.9954 | 0.9736 | **0.9858** | **0.011** | 0/3 |

**The floor buys the stability back, decisively.** Both pre-registered clauses
hold with room to spare: σ = 0.0111 vs the ≤ 2×0.0114 bar — statistically the
fully-graded control's spread, a 7× reduction vs annealing to zero — and the
mean sits 0.006 from the control (bar was 0.023). A residual λ=0.1 costs 10%
of the slot-grade signal and eliminates both failure modes of full removal
(the wobble, and #73-seed-2's slow decay). As a bonus the probe head keeps
learning, so the end-of-run slot readout stays calibrated (0.96–1.00 across
all seeds/slots) — the stale-probe caveat of #73 disappears.

**No earlier onset is reliable, and the failure is bimodal, not gradual.**
Onset 10% collapses to chance on all three seeds. At 20% and 30% each seed
either collapses (0.15–0.27) or fully recovers (0.99) — nothing in between.
The pre-registered "within 2σ_pooled" test is honest only for onset 10%
(cleanly not viable): for 20%/30% the collapses inflate the arm's own σ to
~0.46, widening 2σ_pooled to ~0.65 — a bar that wide rejects nothing, so the
letter of the criterion is vacuous there and the per-seed collapse count is
the real readout: **3/3 → 2/3 → 1/3 → 0/3 across onsets 10/20/30/40%.**

The slot probes at the grade-off step say *why*, cleanly: in every collapsing
run the chain was decodable only through slot 2 (slot 3 ≤ 0.29); in every
recovering run slot 3 had formed (≥ 0.57) — including #73's seed 0 (slot 4
still half-formed at 0.572, recovered to 0.95) and onset-20%'s seed 2 (slot 4
at just 0.331, recovered to 0.99). Final-answer loss can finish and polish a
chain that reaches the deep slots, but cannot extend one stuck at depth 2 —
and worse, the already-formed shallow slots then *erode* (slot 1 decodability
falls from ~0.99 to 0.35–0.70 in the collapsed runs). The viable-onset
boundary is therefore **a state, not a time**: the grade must stay until the
chain is decodable through the deep slots, and how long that takes varies by
seed (seed 0 needed ~40% of the budget; seed 2 was ready by 20%).

## What this bounds

- **The LM-scale recipe, updated:** anneal to a small floor (λ ≈ 0.1), not to
  zero — same warm-up saving, none of the instability. And don't schedule the
  anneal by step count; gate it on a measurement (per-slot decodability
  through the deep slots), because the nucleation time is run-dependent.
- The per-step-target budget from #73 stands, refined: targets must flow at
  full weight until the chain is *fully formed*, then a 10%-weight trickle
  suffices. At this scale nucleation took 20–40% of the budget depending on
  seed.
- A probe-gated anneal (switch on decodability, not step count) is the natural
  follow-up if the scratchpad ladder advances; at toy scale the slot-3 probe
  separated every outcome with a wide margin (0.29 vs 0.57).

## Limitations

- Toy scale, one task family, 3 seeds per arm. The onset arms' bimodality
  makes their means/σ descriptive only; the collapse fractions and the
  slot-probe separation carry the conclusion.
- Floor tested at a single value (0.1) and single onset (40%); the floor's
  lower bound and its interaction with earlier onsets are unmeasured (a
  floor might rescue onset 20–30% — untested).
- The slot-3-decodability threshold (somewhere in 0.29–0.57) is bracketed by
  9 runs, not mapped; treat it as a diagnostic direction, not a number.
