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

Stability note worth keeping: at the harness's default lr 2e-3, depth 8 *collapsed*
to chance (0.374), reproducibly (it also degraded parity in an earlier run). This
was LR-driven instability in the deep unrolled loop, not structural — at lr 1e-3
depth 8 recovers to 0.976, at 5e-4 to 0.981. Biasing the update gate toward
retention did NOT fix it (falsified hypothesis); lowering the LR did. Fix: a
depth-aware LR (deeper recurrence wants a lower LR). The production schedule's peak
LR is already 1e-4, well below the rescuing 1e-3, so it likely won't manifest at
real scale — but the depth-aware rule is the safe default.

## Limitations

Tiny scale (dim 96), single synthetic task family, single seed, in the same-length
regime (no length generalization tested). cumsum/parity tasks from run 1 were less
informative (cumsum too easy — depth-1 saturates; parity too noisy). This shows
depth recurrence *can* help on a task that needs it at toy scale; it does NOT yet
show the gain transfers to fineweb language modeling at 79.6M — that requires
wiring CausalRefiner into the production trainer and a real run, neither done.
What would strengthen it: multi-seed (noise floor), a length-generalization split,
and confirmation at real scale. What would weaken it: no measurable depth benefit
once integrated at scale (plausible if web-text next-token prediction rarely needs
sequential composition — the open question the real run answers).
