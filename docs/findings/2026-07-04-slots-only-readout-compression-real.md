# The serial scratchpad's compression is real: a readout blinded to the tokens stays within noise of the full readout

Status: toy-proof (the #62 gate; strengthens the #38 interpretation)
Date: 2026-07-04
Commit: c664951  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python scratchpad_harness.py --arms serial,slotsonly --seeds 0,1,2 --K 4 --m 7 --steps 2500`

## Setup

The #38 win left one door open: the answer readout attended to `tokens + slots`,
so the slots could have been decorative while the readout secretly re-derived
the answer from the problem tokens. #62 closes it with one data-flow change and
an identical parameter tree: the `slotsonly` arm removes the token states from
the readout's context — if the slots carry the computation, blinding the
readout should cost nothing.

Same protocol as the #38 gate: affine chain K=4, m=7, dim 64, 2500 steps,
held-out 4096, seeds {0,1,2}, matched pairs (same seed ⇒ same pools). A wiring
guard (tests/test_scratchpad_harness.py) proves the blind readout has exactly
zero gradient from token states, and that the control's token path is live.
Pre-registered: slots-only within 2σ_pooled of tokens+slots → compression real;
collapse toward the parallel/chance regime (≈0.19 / 0.14) → slots decorative
and #38 needs a caveat.

## Evidence

Held-out final-answer accuracy:

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|
| serial (tokens+slots) | 0.9988 | 0.9790 | 0.9988 | 0.9922 | 0.011 |
| slots-only | 0.9370 | 0.8767 | 0.9983 | **0.9373** | 0.061 |

The serial control reproduces the #38 numbers exactly (the readout refactor
changed nothing). Delta = 0.055 < 2σ_pooled = 0.088 → **within noise:
compression is real**. And categorically: 0.94 is nowhere near the 0.19
decorative regime — the answer is being read out of the slots.

Where the small gap lives, made visible by the per-slot grades: in both arms
final accuracy ≈ slot-4 accuracy (blind: 0.939/0.872/0.999 slot-4 vs
0.937/0.877/0.998 final). The blinded readout converts its last slot
essentially losslessly; what varies is how well the *write chain* forms
r₄ under different seeds. So the deficit, such as it is, sits in chain
formation during training — not in the readout re-reading the problem.

## Limitations

- 3 seeds; the blind arm's seed spread (σ=0.061) is ~5× the control's. The
  5.5pp mean gap is sub-threshold by the pre-registered bar but not obviously
  zero — more seeds would resolve whether blinding carries a real few-point
  training-dynamics cost (plausible: with tokens removed, all answer-loss
  gradient pressure lands on the slot chain).
- Toy scale, K=4, one task family — same license boundary as #38: this
  strengthens the mechanism's interpretation, it does not upgrade the claim to
  LM scale.
