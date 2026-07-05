# Per-pass supervision cannot replace the trajectory gradient (islands killed), but it stabilizes deep recurrence — parity's depth-8 collapse rescued at +6.5σ

Status: confirmed for the pre-registered kills/nulls; preliminary for the stabilizer observation (emerged from readouts, needs its own confirmatory test)
Date: 2026-07-05
Commit: 3bfe1fd  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python ablation_harness.py --task {statetrack,cumsum5,parity} --depths 8 --steps 2500 --seed {0..5} [--per-pass-loss] [--islands] [--readouts]` (CPU, f32)

## Setup

The #75 hypothesis, built on #64's kill (`2026-07-05-truncated-backprop-depth-kill.md`):
there, the ungraded prefix passes had no training signal and truncation collapsed to
near-chance. Fix under test: grade EVERY pass against the target (deep supervision,
uniform mean over passes); then cutting the gradient chain at every pass boundary
("islands") might hold accuracy at ~O(1) activation memory in depth.

Arms, pre-registered on the issue before any run — matched pairs (same seed ⇒ same
init/pools/minibatch order), statetrack depth 8, dim 96, 2500 steps, seeds {0,1,2}:
**(a)** control, final-pass loss + full chain (#64's control; seed-0 spot-check
reproduced its 0.9580 exactly, validating reuse); **(b)** per-pass loss + islands;
**(c)** per-pass loss, chain intact. Keep: (b) within 2σ_pooled of (a). Partial
keep: (c) beats (a) by ≥2σ. Kill: neither. Secondary readouts (no keep/kill
weight): cumsum5 + parity (a) vs (b); per-pass accuracy; gate openness; depth
transfer 1..12. After (c) posted +1.54σ at 3 seeds, a seed extension {3,4,5} for
(a)/(c) was pre-registered before running: keep (c) iff ≥2σ_pooled over 6 seeds.

## Evidence

**statetrack held-out accuracy (chance 0.20):**

| arm | seeds 0..5 | mean ± σ |
|---|---|---|
| (a) control | .958 / .981 / .977 / **.352** / .972 / .984 | 0.871 ± 0.254 (bimodal) |
| (b) islands | .712 / .724 / .485 | 0.640 ± 0.135 |
| (c) supervised, chain intact | .986 / .989 / .982 / .988 / .983 / .988 | 0.986 ± 0.003 |

- **(b) keep: NOT MET** — Δ = −0.332 vs the 3-seed control, −3.46σ. Killed. But
  supervision moved truncated training from #64's 0.31 (near-chance) to 0.64–0.72
  with monotonically-improving passes: local grading teaches passes to contribute;
  it cannot replace cross-pass credit where the task needs composed sequential
  work. Consistently: cumsum5 (no composition needed) is free (−0.18σ), and see
  parity below.
- **(c) partial keep: NOT MET, twice** — +1.54σ at 3 seeds; extension verdict
  +0.64σ at 6 seeds (control seed 3 collapsed, inflating σ); even the
  working-mode-only view (collapsed seed excluded) is +1.58σ. Per the
  pre-registrations, no mean-accuracy claim is kept. Formally this lands the
  kill branch of #75.

**What the readouts recorded (not pre-registered criteria — observations):**

1. **Parity depth-8 rescue, +6.5σ.** Control reproduces the documented parity
   collapse (0.581 ± 0.048, chance 0.5); islands+per-pass scores 0.929 ± 0.059.
   The one task where deep weight-shared recurrence was known unstable is fixed
   by grading every pass.
2. **Matched-pair collapse rescue.** Control seed 3 collapsed to 0.352; arm (c),
   same seed, same everything but the per-pass loss: 0.988. One-variable
   evidence (n=1 collapse event) that supervision removes the collapse mode;
   (c) posted 0 collapses in 6 seeds, σ 0.003 vs the control's working-mode 0.010.
3. **Passes converge early and start high.** (c)'s pass-1 draft scores 0.61–0.68
   (control pass 1: 0.29) and the curve saturates by pass ~5 while the control is
   still climbing at pass 8. Depth-9 transfer holds better (0.96–0.99 vs 0.88).
4. **Gate behavior.** (c) settles near 0.48–0.52 openness (drafts reach a fixed
   point and the gate stops rewriting them); control keeps gates wider (0.58–0.72)
   and is still churning at pass 8.
5. **Depth ≥10 is chance for every arm** — running past max_depth+1 clips the
   time-embedding row (jnp.take) and the state degenerates. Harness property,
   not an arm difference; relevant to any future depth-extrapolation probe.

## Limitations

Toy scale, depth 8 only, uniform per-pass loss weighting only, 3 seeds on the
secondary tasks, and the stabilizer claim rests on one pre-specified but
non-primary comparison (parity) plus one collapse event. The #75 claims die as
registered; the stabilizer effect is filed as its own pre-registered hypothesis
(follow-up issue) — if it replicates, per-pass supervision is a training-recipe
knob (cost: per-pass head FLOPs at train time), not a memory knob. For the
original O(1)-memory goal, `jax.checkpoint` remains the exact-gradient fallback.
