# A single physically-overwritten slot carries the affine chain as well as K append-only slots — forgetting by capacity, phase 1

Status: confirmed (phase 1 of #63; phase 2 in a companion entry)
Date: 2026-07-05
Commit: c3e9fb25f6ca65e864b722c76277c97b47dba17d  Run: tiny-config ablation (synthetic,
no runs/ id)  Measured with: `python scratchpad_harness.py --arms overwrite,serial
--seeds 0,1,2 --K 4 --m 7 --steps 2500` (CPU, JAX_PLATFORMS=cpu)

## Setup

Design: `docs/design/budget-scratchpad.md`. `overwrite` is `BudgetScratchpadNet`
with a single physical memory slot (S=1): K=4 sequential writes, each write
reading tokens + the current slot and then **fully replacing it** — a softmax
address over a size-1 axis is identically 1, so this is architecturally forced
overwrite, no addressing parameters do anything. `serial` is the existing #38
scratchpad (K=4 append-only slots, nothing ever evicted) — the unlimited-memory
control. Both arms share the affine-chain task (`r_k = (r_{k-1}·a_k + b_k) mod
7`, non-commutative, chance=1/7), graded per-step on the sub-result at each
write, dim 64, heads 4, enc 2, 2500 steps, held-out 4096, seeds {0,1,2}. Win bar
(pre-registered): `overwrite` matches `serial` within 2σ_pooled on held-out final
accuracy.

## Evidence

Held-out final-answer accuracy, 3 seeds:

| arm | params | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|---|
| overwrite | 0.22M | 0.9954 | 0.9998 | 0.9934 | **0.9962** | 0.0033 |
| serial | 0.22M | 0.9988 | 0.9790 | 0.9988 | 0.9922 | 0.0114 |

overwrite − serial = +0.0040, σ_pooled = 0.0084 → 0.48σ apart — well inside the
2σ win bar, and the sign points the "wrong" way (the 1-slot arm is nominally
*ahead* of the K-slot arm, itself inside seed noise). No kill criterion fired:
per-slot accuracies for `overwrite` are ≥0.994 at every one of the 4 sequential
writes in every seed —

| arm | slot 1 (r₁) | slot 2 | slot 3 | slot 4 |
|---|---|---|---|---|
| overwrite (seed 0) | 1.000 | 1.000 | 0.998 | 0.995 |
| overwrite (seed 1) | 1.000 | 1.000 | 1.000 | 1.000 |
| overwrite (seed 2) | 1.000 | 1.000 | 0.999 | 0.994 |

— so the grade stayed alive through every overwrite; there is no sign of the
RMT forget-gate collapse the design doc named as the failure mode to watch for.

## Reading

Phase 1's hypothesis was narrow: does forgetting-by-overwrite (physically
replacing the only slot at every step) cost anything relative to never
forgetting at all (K slots, append)? It does not, at this toy scale — a single
slot, rewritten 4 times, carries the same non-commutative composition chain
that needed 4 dedicated append-only slots in the #38 result. This is consistent
with the sub-result at step k only ever needing r_{k-1} as input (the task's own
recurrence, `r_k = f(r_{k-1}, a_k, b_k)`) — nothing downstream of step k-1 needs
step k-2's value, so a chain with no genuine reason to retain old state pays
nothing for being forced not to.

This licenses phase 2, which is the sharper question: does the *same*
architecture learn a genuine keep-vs-evict policy when something actually needs
outlasting its "natural" overwrite point? Phase 1 alone cannot show that — a
task where nothing needs remembering can't distinguish "learned to forget" from
"never had anything worth keeping." See the companion phase-2 finding.

## Limitations

Toy scale only (0.22M params, K=4, m=7), matches the #38 precedent exactly so
it does not introduce a new scale caveat. Only one budget point (S=1) was
tested against the K=4 ceiling; whether S=1 continues to match at larger K
(where more distinct sub-results share the one slot for a longer sequence) is
untested. The chain task by construction never requires holding anything past
one step — so a positive phase-1 result was the more likely outcome regardless
of whether the addressing mechanism (irrelevant here, since S=1 has none) works
at all; the real test of the mechanism is phase 2.

## Relation to prior work

The graveyard precedent this design set out to avoid: the `forget_head`/
cross-window hunch died because the forget/blend gate could collapse to
near-zero and the gradient never revived it
(`docs/findings/2026-06-13-cross-window-hunch-inert.md`) — forgetting there was
optional, and the model routed around the memory entirely. Here forgetting is
not optional (S=1 has no "keep" branch to collapse to), which is exactly why
this result is uninteresting on its own and only becomes informative once a
real retention/eviction *choice* exists — phase 2.
