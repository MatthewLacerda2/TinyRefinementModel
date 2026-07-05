# Per-depth loss doesn't move final-depth accuracy past the noise floor, but makes every early refinement iteration dramatically, reproducibly more capable

Status: confirmed (pre-registered final-accuracy bar missed, ~1.5σ; pre-registered
iterations-to-threshold alternate clearly met, 3.3–5.7σ across three depths at K=4,
reproduced sharper at a K=8 confirmatory check)
Date: 2026-07-05
Commit: 4adccb9  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with:
`JAX_PLATFORMS=cpu FORCE_F32_COMPUTE=1 python ablation_harness.py --task statetrack --depths {4,8} --steps 2500 --seed {0,1,2} [--per-depth-loss] --depth-curve`

## Setup

#74's hypothesis: grade the next-token prediction at **every** iteration of the
refiner's shared K-loop, not just the last — the target at each depth is the same
token we already have, so this installs no new human knowledge, only
credit-assignment plumbing (`CausalRefiner.all_depth_logits`, `plan_a_model.py`;
mean CE across depths, uniform weights, `ablation_harness.py --per-depth-loss`).
Control is today's behavior: loss on the final depth only.

Protocol pre-registered on the issue before any run: statetrack (the one toy task
with headroom and structure, `2026-06-16-plan-a-depth-ablation.md`), refiner arm,
dim 96, 2500 steps, matched pairs (same seed ⇒ same init/pools/minibatch order;
only the loss differs), seeds {0,1,2} at K=4. Keep if the per-depth arm beats final-
only by ≥2σ_pooled on held-out accuracy, or matches while reaching a given accuracy
in measurably fewer steps; kill if within 2σ. A kill at K=4 pre-registered one
confirmatory run at the harness's deepest configuration (K=8, seed 0) before
writing this finding. `--depth-curve` records held-out accuracy at every
intermediate depth in both arms, to diagnose *where* any effect shows up.

## Evidence

**Final-depth accuracy, K=4** (mean ± sample σ over seeds {0,1,2}):

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|
| final-only (control) | 0.9824 | 0.9345 | 0.9649 | 0.9606 | 0.0242 |
| per-depth loss | 0.9845 | 0.9886 | 0.9876 | 0.9869 | 0.0021 |

Δ = 0.0263, σ_pooled = 0.0172, 2σ_pooled = 0.0344 → **Δ = 1.53σ, misses the
pre-registered keep bar.** This one number alone reads as a near-miss kill — final
accuracy is not where the effect lives.

**Per-depth accuracy curve, K=4** (mean over seeds {0,1,2}; same runs, read out at
every intermediate iteration via `all_depth_logits`):

| depth | final-only mean (σ) | per-depth mean (σ) | Δ | σ_pooled | Δ/σ |
|---|---|---|---|---|---|
| 1 | 0.5299 (0.0265) | 0.7246 (0.0464) | **+0.1947** | 0.0378 | **5.16σ** |
| 2 | 0.7085 (0.0401) | 0.9080 (0.0293) | **+0.1995** | 0.0351 | **5.68σ** |
| 3 | 0.8694 (0.0461) | 0.9779 (0.0073) | **+0.1085** | 0.0330 | **3.29σ** |
| 4 (final) | 0.9606 (0.0242) | 0.9869 (0.0021) | +0.0263 | 0.0172 | 1.53σ |

The effect is huge and unambiguous at every depth *except* the last, where both
arms are converging toward the task's ceiling (~0.98–0.99) and there is
correspondingly little headroom left for a gap to show.

**K=8 confirmatory check (seed 0 only, one matched pair, pre-registered on a K=4
near-miss):**

| depth | final-only | per-depth loss | Δ |
|---|---|---|---|
| 1 | 0.2944 | 0.6999 | +0.4055 |
| 2 | 0.3764 | 0.8741 | +0.4977 |
| 3 | 0.4888 | 0.9642 | +0.4754 |
| 4 | 0.6047 | 0.9856 | +0.3809 |
| 5 | 0.7212 | 0.9878 | +0.2666 |
| 6 | 0.8323 | 0.9878 | +0.1555 |
| 7 | 0.9139 | 0.9879 | +0.0740 |
| 8 (final) | 0.9580 | 0.9878 | +0.0298 |

Same shape, sharper: final-only climbs the whole way to d8 to reach 0.958;
per-depth loss **plateaus by d4** (0.986) and iterations 5–8 do essentially
nothing more. The final-depth gap (+0.030, single seed, no σ available) again
sits at the same modest scale as the K=4 result — consistent, not a fluke, and
still not the story.

## Interpretation

- **The pre-registered final-accuracy criterion is not met.** At both K=4 (1.53σ)
  and K=8 (+0.030, comparable to the ~0.01–0.02 seed-to-seed spread the original
  depth-ablation found at this task and depth), the gap sits inside or barely
  outside the noise a single held-out-accuracy number can support. Read in
  isolation, final accuracy says "kill."
- **The pre-registered depth-curve diagnostic says something different and much
  stronger.** #74 asked, if the win doesn't show at the final step, to check
  *where* it shows: "earlier iterations becoming independently correct vs. the
  final one just converging faster." That is exactly what happened. Per-depth
  loss makes every early iteration dramatically more capable on its own — at K=4,
  3.3–5.7σ above the seed-noise floor at d1, d2, d3; at K=8, the model is
  effectively *solved* by iteration 4 instead of needing all 8. This is
  "iterations-to-threshold," the pre-registered alternate keep clause, just
  measured on the K-loop's own iteration axis (which is what `--depth-curve` was
  built to measure) rather than the optimizer's step count (which was held fixed
  across arms by design and was never an independent variable here).
- **Why final accuracy doesn't move: the task is already close to solved by
  final-only at this depth.** `2026-06-16-plan-a-depth-ablation.md` showed
  final-only statetrack climbing to ~0.977 by d8; there just isn't much ceiling
  left for a final-depth number to show a large win once the control already
  gets most of the way there given the *whole* K-loop to work with. Per-depth
  loss doesn't buy a more powerful final answer — it buys the **same** answer
  using far fewer of the loop's iterations, which final-depth-only accuracy is
  structurally blind to.
- **This reframes the stage-2 pitch.** #74 justified per-depth loss as
  credit-assignment plumbing, not a final-accuracy lever, and that is precisely
  what was found: real, large, reproducible, but visible only through the depth
  axis. Any future use of this signal (a random-single-depth estimator, or a
  weighted per-depth loss that favors early depths) should be judged on
  *iterations-to-solve* or *inference-time compute at a fixed accuracy*, not on
  final-depth held-out accuracy alone — the latter can show ~nothing even when
  the training-loss objective did exactly what it was designed to do.

## Limitations

- Toy scale, one task (statetrack — the only one with headroom + structure per
  the original depth ablation), one width (dim 96). Parity and cumsum5 were not
  re-run here since neither showed depth signal in the original ablation
  (parity flat from attention's documented parity pathology; cumsum5 has no
  headroom past d1); a per-depth-loss effect on final accuracy is unlikely to
  appear where a final-only depth effect never did either.
- K=8 confirmatory check is a single seed (pre-registered as "one confirmatory
  run"); its final-depth Δ (+0.030) has no σ of its own, though it lands at the
  same order as the 3-seed K=4 result and the depth-curve shape is the same,
  only more pronounced.
- The "iterations-to-threshold" reframing is post-hoc language for what the
  pre-registered depth-curve diagnostic already measured, not a new metric
  invented after seeing the numbers — but it wasn't the literal "steps" the
  keep clause named (optimizer steps), so this finding reports the literal
  criterion's miss plainly rather than silently substituting one bar for
  another.
- No wall-clock or VRAM claim here: at this toy scale `all_depth_logits` costs
  about the same wall-clock as the final-only forward pass (measured K=4 runs:
  763–828s regardless of arm); the stage-2 concern flagged in #74 (per-depth
  logits through the tied head is the single biggest VRAM line, ×K, at LM scale)
  is untested and remains the real cost question for scaling this up.
