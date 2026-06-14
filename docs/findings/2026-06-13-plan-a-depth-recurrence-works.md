# Causal within-window depth recurrence helps where the cross-window hunch did not: deeper recurrence monotonically improves a sequential-composition task

Status: preliminary
Date: 2026-06-13
Commit: feat/ablation-harness @ HEAD  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `venv/bin/python -u ablation_harness.py --task statetrack --depths 1,2,4,8 --steps 6000 --lr 1e-3`

## Setup

Plan A (docs/design/plan-a.md) replaces the inert cross-window slot-hunch loop
[[cross-window-hunch-inert]] with Universal-Transformer-style refinement: a shared
transformer block looped K times causally over the token representations, decode
from the refined state (CausalRefiner in plan_a_model.py; causality verified —
perturbing position j leaves all logits at < j bit-identical at every depth). To
test "does depth help" on the right instrument (fineweb perplexity is the wrong
yardstick), a tiny-config harness trains the model from scratch at fixed depths on
a non-commutative state-tracking task (permutation composition, seq 24, 5 states,
4 generators) — no sum shortcut, so getting position t right requires sequentially
tracking the state. dim 96, 2 encoder layers, 6000 steps, held-out eval.

## Evidence

Held-out accuracy by inference depth, single LR (1e-3):

| depth | val_acc | val_ce |
|---|---|---|
| 1 | 0.8480 | 0.3651 |
| 2 | 0.9194 | 0.2010 |
| 4 | 0.9837 | 0.0510 |
| 8 | 0.9755 | 0.0755 |

Depth helps monotonically 1 -> 2 -> 4 (+0.136 accuracy, 7x lower CE); depth 8
holds (no further gain past 4 — diminishing returns, depth 4 is the sweet spot
here). This is the depth-helps signal the cross-window hunch never produced
[[cross-window-hunch-inert]]: there, more reasoning steps left next-window CE flat
or slightly worse; here, more causal refinement iterations measurably improve the
prediction they refine.

Robust across 3 seeds (0, 1, 2), mean accuracy [range]:

| depth | mean acc | range |
|---|---|---|
| 1 | 0.856 | 0.848-0.871 |
| 2 | 0.934 | 0.919-0.943 |
| 4 | 0.983 | 0.982-0.984 |
| 8 | 0.981 | 0.976-0.985 |

The depth-1 -> depth-4 gain (+0.127 mean) is unambiguous: depth-1's *best* seed
(0.871) is below depth-4's *worst* (0.982) — no overlap. Depth 4 and 8 are
statistically indistinguishable (saturation past 4), and depth 8 is stable across
every seed at lr 1e-3 (the LR fix holds, not a one-seed fluke).

Stability note worth keeping: at the harness's default lr 2e-3, depth 8 *collapsed*
to chance (0.374), reproducibly (it also degraded parity in an earlier run). This
was LR-driven instability in the deep unrolled loop, not structural — at lr 1e-3
depth 8 recovers to 0.976, at 5e-4 to 0.981. Biasing the update gate toward
retention did NOT fix it (falsified hypothesis); lowering the LR did. Fix: a
depth-aware LR (deeper recurrence wants a lower LR). The production schedule's peak
LR is already 1e-4, well below the rescuing 1e-3, so it likely won't manifest at
real scale — but the depth-aware rule is the safe default.

## Length generalization (run 7, 3 seeds)

Train on length 24, evaluate on length 48 (RoPE positions 24-47 unseen, state chain
2x longer). Mean accuracy [range]:

| depth | acc @ 48 | range | acc @ 24 (run 6) |
|---|---|---|---|
| 1 | 0.542 | 0.537–0.545 | 0.856 |
| 2 | 0.600 | 0.597–0.605 | 0.934 |
| 4 | 0.639 | 0.633–0.643 | 0.983 |
| 8 | **0.683** | 0.674–0.691 | 0.981 |

Two-sided, and the interesting part is the sign flip vs in-distribution:

- **Absolute length generalization is weak** — best 0.68 at 2x length vs 0.98
  in-distribution (above chance 0.20, so partial transfer, not solved). Report
  honestly: the model leans partly on length-bound structure and does not freely
  extrapolate RoPE to unseen positions.
- **Depth is the lever that helps under length shift, and stops saturating.**
  In-distribution accuracy plateaued at depth 4 (4 ≈ 8); out-of-distribution it
  climbs monotonically 1->2->4->8 (+0.141 mean, no adjacent-depth overlap on any
  seed). A longer sequence is a longer sequential composition, so it needs more
  refinement iterations — depth-as-compute scaling with problem size. This is
  direct evidence for letting the inference depth dial scale with context length
  rather than pinning it at a fixed 4 (and motivates the untried adaptive-depth
  direction, distinct from the ACT halting removed in f6cb905).

## Limitations

Tiny scale (dim 96), single synthetic task family, 3 seeds (depth gain robust
across them). Length generalization tested (run 7 above) and found weak in absolute
terms, though the depth benefit survives and grows under length shift. cumsum/parity
tasks from run 1 were less informative (cumsum too easy — depth-1 saturates; parity
too noisy). This shows
depth recurrence *can* help on a task that needs it at toy scale; it does NOT yet
show the gain transfers to fineweb language modeling at 79.6M — that requires
wiring CausalRefiner into the production trainer and a real run, neither done.
What would strengthen it: multi-seed (noise floor), a length-generalization split,
and confirmation at real scale. What would weaken it: no measurable depth benefit
once integrated at scale (plausible if web-text next-token prediction rarely needs
sequential composition — the open question the real run answers).
