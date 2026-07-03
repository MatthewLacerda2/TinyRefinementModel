# The supervised serial latent scratchpad beats both controls — order plus per-slot grades turn an unlearnable composition task into a solved one

Status: toy-proof (win at the #38 gate; no LM-scale claim)
Date: 2026-07-03
Commit: claude/project-code-quality-5r6pbn-38 @ HEAD  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python scratchpad_harness.py --arms serial,parallel,depthonly --seeds 0,1,2 --K 4 --m 7 --steps 2500` (CPU, FORCE_F32_COMPUTE=1)

## Setup

The #38 bet: reasoning as a serial latent scratchpad — K ordered sub-slots, each
written exactly once, each *graded* (per-slot CE against sub-result r_k), slot k
reading only tokens and slots < k. Design pinned in
docs/design/serial-scratchpad.md before any run; criteria pre-registered on the
issue. Task: chained affine maps, r_k = (r_{k-1}·a_k + b_k) mod 7, K=4 — chosen
over a running sum because affine composition is non-commutative, so no
order-free shortcut exists and the answer decomposes exactly into the graded
sub-results. Three arms, matched pairs (same seed ⇒ same pools):

- **serial** — the bet (shared cross-attn write block, slots see earlier slots);
- **parallel** — byte-identical parameter tree, all slots written in one step
  from tokens only (order removed; the single variable, guarded by a
  gradient-flow test in tests/test_scratchpad_harness.py);
- **depthonly** — CausalRefiner at depth K, final-answer supervision only.

dim 64, heads 4, enc 2, 2500 steps, held-out 4096 sequences, seeds {0,1,2},
chance = 1/7 ≈ 0.143. Win bar (pre-registered): serial beats BOTH controls by
≥ 2σ_pooled on held-out final accuracy.

## Evidence

Held-out final-answer accuracy, 3 seeds:

| arm | params | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|---|
| serial | 0.22M | 0.9988 | 0.9790 | 0.9988 | **0.9922** | 0.011 |
| parallel | 0.22M | 0.2185 | 0.1853 | 0.1597 | 0.1878 | 0.030 |
| depthonly | 0.17M | 0.1482 | 0.1287 | 0.1423 | 0.1397 | 0.010 |

Serial − parallel = +0.804 — roughly 36× the pooled seed σ (bar was 2σ). Serial −
depthonly is larger still. Neither kill criterion fired: no write-path collapse
(serial's slot accuracies all ≈ 1.0 — the grades kept every link alive, unlike
the starved forget-gate of findings 2026-06-11/13), and the order gap is
enormous.

Per-slot accuracies are the mechanism made visible:

| arm | slot 1 (r₁) | slot 2 | slot 3 | slot 4 |
|---|---|---|---|---|
| serial | 1.000 | 0.999 | 0.997 | 0.991 |
| parallel | 1.000 | 0.996 | ~0.29 | ~0.15 |

Parallel slots nail r₁ and r₂ (shallow token functions: r₁ = b₁,
r₂ = b₁a₂ + b₂) then collapse to near-chance exactly where composition of a
prior *latent* result becomes unavoidable. With the identical parameters, giving
slot k read-access to slots < k carries the chain to the end. Depth-only at
chance says undecomposed depth-4 recurrence cannot learn K=4 affine chains here
at all — the win is not "the task was easy".

## Relation to prior work

Plain serial latent feedback is Coconut (published); the contribution here is the
*structured, per-slot-supervised* decomposition, and this is the first
non-bypassable slot design in this repo that survived — the two unsupervised
predecessors died by gradient starvation and by leaking the future
[[slot-future-leak]] [[cross-window-hunch-inert]]. The bypass principle held:
what made the difference is precisely that the loss could not be minimized
without the chain existing.

## What this does and does not license

Toy-scale only. It earns the next rung — harness-scale probes of the same wiring
on language-shaped data — not a production run (user gate per AUTONOMY.md). It
also unblocks #39 (convergence halting) per that issue's blocker. The λ=1.0 grade
weight was fixed in advance and untuned; sensitivity to λ, K > 4, and larger m
are open follow-ups.
