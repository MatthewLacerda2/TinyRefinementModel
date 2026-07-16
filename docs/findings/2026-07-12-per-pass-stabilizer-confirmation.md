# The per-pass stabilizer claim dies as registered — parity's rescue was islands-specific — but chain-intact per-pass supervision eliminates the statetrack collapse mode across 12 matched seeds

Status: killed per pre-registration (conjunction failed on the parity clause); the surviving statetrack clause is now well-supported, recorded as scoped
Date: 2026-07-12
Commit: 436c723  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python ablation_harness.py --task {parity,statetrack} --depths 8 --steps 2500 --seed {0..5|6..11} [--per-pass-loss] [--readouts]` (CPU, f32)

## Setup

#77's hypothesis, spun out of #75's readouts: grading every refinement pass
(`--per-pass-loss`, gradient chain intact) removes the depth-8 collapse mode of
deep weight-shared recurrence. Pre-registered keep required BOTH clauses:
parity (c) beats final-only (a) by ≥2σ_pooled over 6 seeds, AND statetrack
collapse count (acc < 0.5) over 6 fresh seeds lower/equal in (c) with zero (c)
collapses. Kill: parity gap under 2σ → "the #75 parity readout was an
islands-specific or seed artifact; record and stop."

Matched pairs throughout (same seed ⇒ same init/pools/minibatch order). The
three #75 parity controls were reused: `ablation_harness.py`/`plan_a_model.py`
are byte-identical since commit 3bfe1fd (diff checked), and pipeline
determinism has reproduced exact numbers twice before.

## Evidence

**Parity, depth 8, seeds {0..5} (chance 0.5) — the primary clause: FAILED.**

| arm | seeds 0..5 | mean ± σ |
|---|---|---|
| (a) final-only | .582 / .628 / .532 / .627 / .577 / .560 | 0.584 ± 0.037 |
| (c) per-pass, chain intact | .531 / .624 / .788 / .574 / .915 / .560 | 0.665 ± 0.153 |

Δ = +0.081 at +0.73σ against the 2σ bar. Per the pre-registration the #75
parity rescue (0.929 ± 0.059) was **islands-specific, not a seed artifact**:
that rescue used per-pass grading PLUS a severed chain, and grading alone does
not reproduce it (two seeds lift, four stay at coin-flip). On parity, the
unstable cross-pass feedback loop apparently IS the pathology — #75's cut
removed it; supervision alone only sometimes fights it.

**Statetrack, depth 8, fresh seeds {6..11} — the secondary clause: PASSED.**

| arm | seeds 6..11 | mean ± σ | collapses |
|---|---|---|---|
| (a) final-only | .981 / .750 / .894 / **.414** / .983 / .980 | 0.834 ± 0.225 | 1 (+2 degraded) |
| (c) per-pass | .987 / .985 / .988 / .987 / .986 / .987 | 0.9867 ± 0.0013 | 0 |

Pooled with #75's seeds {0..5}, the 12-matched-seed picture: control
0.852 ± 0.230 with 2 collapses (seeds 3, 9) and 2 degraded seeds (7: 0.750,
8: 0.894); per-pass 0.9864 ± 0.0022 with **12/12 seeds in [0.982, 0.989]**.
The control's failure mode is real and recurrent (~2/12 collapse, ~4/12
off-nominal); per-pass supervision has never produced an off-nominal seed.

## Verdict and what survives

The claim **as registered is killed** — the keep was a conjunction and parity
failed it. What survives, scoped honestly: chain-intact per-pass supervision
eliminates the training-instability mode **on statetrack**, the task family
where the refiner's depth actually earns its keep
(2026-06-16-plan-a-depth-ablation.md), with a 100× variance reduction
(σ 0.0022 vs 0.2295) at zero inference cost. It is not a universal deep-
recurrence stabilizer: parity's pathology needs the chain cut, which costs
accuracy where cross-pass credit matters (#75). Related evidence from the
other instrument: the scratchpad's per-slot grade also stabilizes
(2026-07-10-grade-annealing-scaffold-not-crutch.md — σ up 7× when annealed
away), so "local grading stabilizes what it touches" now has two independent
demonstrations, each scoped to its own mechanism.

## Limitations

Toy scale, depth 8, uniform loss weighting. The statetrack stabilizer clause
passed its pre-registered form, but the accuracy-mean comparison remains
untested at ≥2σ rigor in the working mode (both #75 attempts stayed inside the
bar). Before any real-scale adoption: the per-pass loss multiplies LM-head CE
cost by depth, which collides with chunked CE (#19) — needs a design pass, and
the grade-annealing result suggests a cheaper shape (per-pass grade as warm-up
scaffold with a floor, cf. #95/#101) rather than run-long full weighting.
