# Corrected-eval rerun of #63 phase 2: the readout combines two retrieved values at ~0.99, killing the retracted readout-capacity diagnosis — but the recall task leaks through the token-visible writer, so retention under capacity pressure is STILL untested

Status: confirmed (the ceiling and the leak); the retention question stays open
Date: 2026-07-16
Commit: b6a8897 (main, post PR #76 — the corrected `final_target` eval)
Run: tiny-config ablation (synthetic, no runs/ id)
Measured with: `JAX_PLATFORMS=cpu python scratchpad_harness.py --arms unlimited --seeds 0,1,2 --K 5 --m 7 --steps 2500` then the same with `--arms budget1,budget2` (#114)

## Setup

Identical to the July 5 phase-2 run (`2026-07-05-budget-scratchpad-recall-inconclusive-readout-confound.md`,
retracted): recall target `(r_1 + r_K) mod m`, K=5, m=7, dim 64, 2500 steps,
seeds {0,1,2}, `read_tokens=False` for all three arms. The ONLY change is the
eval fix from PR #76 — `final_target()` is now shared by loss and eval, so the
arms are scored on the target they were trained on. Ceiling-first order per
rule 4 (#114 protocol): validate `unlimited` solves the task before reading
the budget comparison at all.

## Evidence

Held-out final-answer accuracy against the CORRECT target (chance = 1/7 = 0.1429):

| arm | seed 0 | seed 1 | seed 2 | mean | σ | July 5 (broken eval) |
|---|---|---|---|---|---|---|
| unlimited | 0.9888 | 0.9917 | 0.9927 | **0.9911** | 0.0020 | 0.1431 |
| budget1 (S=1) | 0.9097 | 0.9424 | 0.8391 | **0.8971** | 0.0528 | 0.1443 |
| budget2 (S=2) | 0.8044 | 0.1375 | 0.9746 | **0.6388** | 0.4424 | 0.1445 |

Pre-registered bars (carried over from July 5): budget2 wins by beating
budget1 by ≥2σ_pooled and approaching unlimited; budget2 at budget1's level
kills. Observed: budget2 − budget1 = −0.258, σ_pooled = 0.315 → −0.82σ.
The win bar fails decisively. The kill bar technically fires (within noise of
budget1) — but see the leak below: the comparison does not measure what the
bars were written to judge, so neither verdict is claimed.

## Reading — three separate conclusions

**1. The retracted diagnosis is now definitively dead.** `unlimited` solves
the recall task at 0.991 under the corrected eval — same config, same seeds
as July 5. The readout (one cross-attention pass + linear head) combines two
retrieved values just fine; the July 5 "all arms at chance" was 100% the
eval-target bug. The model had solved the task behind the broken grader.

**2. The task does not force retention — the writer is a side channel.** The
supposedly-retention-free control, budget1 (S=1, forced full overwrite),
scores 0.897. Mechanically (see `BudgetScratchpadNet.__call__`): with S=1 the
softmax address is identically 1, so final memory IS the step-5 write vector
`v_5`, and the readout sees nothing else. Yet 0.897 accuracy on
`(r_1 + r_5) mod 7` means `v_5` carries the combined answer. It can: each
write is `write_block(q_k, concat(h, memory))` where `h` is the FULL encoded
token sequence — the step-5 writer sees everything, so it can compute
`(r_1 + r_5) mod m` directly and pack it into the 64-dim slot vector alongside
the grade-mandated r_5. Slot COUNT pressure is not INFORMATION pressure when
every writer is omniscient over tokens. The #62 `read_tokens=False` wiring
closed the readout-side shortcut but left the writer-side one open.

**3. Two slots destabilize optimization (now visible in a valid metric).**
budget2 spans 0.14–0.97 across seeds (σ = 0.44, vs 0.05 for budget1 and 0.002
for unlimited); seed 1's per-slot grades collapse at writes 4–5
(`end[1.000 1.000 0.990 0.139 0.133]`). The July 5 run saw the same pattern in
the per-slot grades and could not claim it; with the final metric now valid,
the addressing competition is a real optimization hazard on a task both
neighbors solve — though n=3 keeps this an observation, not a measured effect.

## Limitations

Toy scale, 3 seeds, one step budget (2500), dim 64. Conclusion 2 is a
mechanical deduction (S=1 memory ≡ v_5) plus the 0.897 measurement — solid —
but the follow-up (a writer that only sees local context, so r_1 is genuinely
gone unless memory keeps it) is the experiment that would actually test
phase 2's hypothesis. Filed as the successor to #114; new pre-registration
required.

## Relation to prior work

- Retracts-and-replaces the July 5 phase-2 diagnosis (that file's 2026-07-16
  addendum points here). Phase 1 (`2026-07-05-budget-scratchpad-overwrite-carries-chain.md`)
  is untouched: its target is r_K, which S=1 overwrite legitimately carries.
- The budget2 instability echoes #79's "slots are load-bearing" line: memory
  layout choices show up first as optimization variance, not as capability.
