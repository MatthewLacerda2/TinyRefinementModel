# Truncated backprop through refinement depth collapses state-tracking to near-chance — the trajectory gradient is load-bearing, idea killed

Status: confirmed (negative result; pre-registered kill criterion met, ~26–32σ past the bar)
Date: 2026-07-05
Commit: cc9ee7a  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python ablation_harness.py --task statetrack --depths 8 --steps 2500 --seed {0,1,2} [--grad-last {1,2}]` (CPU, f32)

## Setup

The #64 hypothesis: gradient through only the last j refinement iterations is
enough — earlier iterations would learn indirectly, because the state they hand
forward must make the graded final step(s) succeed ("learn the destination,
forget the path"). If true, deep refinement trains at ~O(1) activation memory in
depth instead of O(depth). Mechanism: `grad_last=j` on the CausalRefiner detaches
the loop state at iteration depth−j (deviation from the encoder output only, so
the encoder keeps an identity gradient path).

Protocol pre-registered on the issue before any run: statetrack, refiner arm,
depth 8, dim 96, 2500 steps, matched pairs (same seed ⇒ same init, pools, and
minibatch order; only `grad_last` varies), seeds {0,1,2}. Keep if j=1 or j=2 is
within 2σ_pooled of full backprop; kill if the loss exceeds 2σ even at j=2.

## Evidence

Held-out accuracy, mean ± σ over seeds {0,1,2} (chance = 0.20):

| arm | acc (seeds 0/1/2) | mean ± σ | val CE | Δ vs full | 2σ_pooled |
|---|---|---|---|---|---|
| full backprop | .958 / .981 / .977 | 0.972 ± 0.012 | 0.081 ± 0.034 | — | — |
| j=2 | .306 / .290 / .253 | 0.283 ± 0.028 | 1.508 ± 0.051 | **−0.689 (32σ)** | 0.043 |
| j=1 | .318 / .339 / .272 | 0.310 ± 0.034 | 1.450 ± 0.051 | **−0.662 (26σ)** | 0.051 |

The kill criterion is met without ambiguity: both truncated arms sit ~0.7
accuracy points below full backprop against a noise floor of ~0.02, and j=2 buys
nothing over j=1 — the failure is not marginal, so no intermediate j was probed.

**Truncation is worse than shallowness.** The instructive comparison: a depth-1
refiner with full backprop scores ~0.82 on statetrack
(2026-06-16-plan-a-depth-ablation.md), yet the depth-8 truncated arms score
~0.28–0.31 — far below "just use the graded suffix as a shallow model." The
mechanism is weight sharing: the refine block's weights are trained only for
their role as the final graded step(s), but the same weights run the 7 ungraded
prefix iterations. Every update blindly changes what the prefix does to the
state, the graded suffix chases a moving, never-corrected input distribution,
and the composition scrambles rather than refines. The prefix isn't a frozen
no-op; it's an untrained transformation applied 7 times.

**Why this diverges from the literature.** Geiping et al. (Huginn) train a
recurrent-depth model successfully with truncated backprop — but their
recurrence re-injects the input embedding at every iteration (anchoring the
prefix trajectory) and backprops through k=8 steps of a much longer recurrence.
Our refiner feeds the encoder state in once at step 0 and was cut to j∈{1,2}.
The negative result is about *this* gate/no-reinjection architecture at tiny
scale, not about truncated backprop in general.

**The compute win was real — it just costs the task.** Truncated runs took ~2×
less wall-clock per run under equal core allocation (j=2: ~1050s vs full:
~2080s, uncontended CPU), confirming the backward pass through the prefix is
genuinely skipped. The memory/compute mechanism works; the model it trains
doesn't.

## Limitations

Toy scale, one task, one depth (8), j ∈ {1, 2} only, 3 seeds. Two follow-ups
could revive the *goal* (O(1)-memory deep refinement) without contradicting this
kill: (a) Huginn-style input re-injection at every iteration, which changes the
architecture (a separate `idea`, not a re-run of this one); (b) gradient
checkpointing (`jax.checkpoint` per iteration), which buys O(√depth) or O(1)
activation memory for a ~33% FLOPs surcharge with *exact* gradients — the
boring, guaranteed-correct alternative for the dim960/#16 and batching/#24
memory pressure this idea was chasing. The claim as stated in #64 is dead:
last-step-only gradient does not hold accuracy on the gated CausalRefiner.
