# Letting the halting head reread the model's whole thought trajectory does not prevent ACT-style collapse — on a validated-solvable task, halting slams to minimum depth with or without visibility, pinning the graveyard failure on the incentive, not the observability

Status: confirmed (negative — bar 3 of the pre-registration)
Date: 2026-07-19
Commit: 9071ae9 (claude/123-trajectory-halting — HaltingScratchpadNet + variable_chain_task)
Run: tiny-config ablation (synthetic, no runs/ id)
Measured with: `JAX_PLATFORMS=cpu python scratchpad_harness.py --arms halt_off,halt_state,halt_traj --seeds 0,1,2 --K 4 --steps 10000` (log `aux_123_retry.log`; ceiling-validation diagnostics in `aux_123_protocol.log` / `aux_123_diag*.log`) (#123)

## Setup

Hypothesis (#123): thinking-token models can judge "have I thought enough"
because their past thinking is visible to attention; the graveyard's ACT
collapse (learned halting → minimum depth, surviving reward shaping) was
measured with a halting decision blind to everything but the current state.
Two live autopsies — observability (the head couldn't see the evidence) vs
incentive (the gradient prefers quitting regardless) — separated here for the
first time.

Design: halting as a READOUT choice on the serial scratchpad — all K writes
always run (grades on, #67), a per-step answer is read from slots 1..k, and
p = softmax(halt logits) weights the per-step answer CE plus a ponder cost
λ_p·E[halt step] (λ_p = 0.2, pre-registered). Min-depth collapse is
expressible but never architecturally forced. One variable, identical param
tree: `halt_traj`'s halt query cross-attends slots 1..k (can reread the
trajectory); `halt_state`'s sees slot k alone (the graveyard configuration).
`halt_off` (read out at K, no halting) is the rule-4 ceiling.

Task: variable-length affine chain, k_eff ~ U{1..K} real links then
recognizable pad links, sub-targets stationary after the chain ends — "done"
is detectable as the state no longer changing, and non-degenerate halting has
k_eff to track. First attempt at K=6/2500 steps FAILED its validity gate
(ceiling 0.46) — recorded on the issue; ceiling-first diagnostics found
K=4/10000 steps solvable (0.999) and the comparison was re-pre-registered
there before any arm ran.

## Evidence

K=4, m=7, dim 64, 10000 steps, 3 seeds (chance = 1/7 ≈ 0.143; random halting
would put p(halt@1) ≈ 0.25 and corr ≈ 0):

| arm | full-depth acc | halted acc | corr(halt step, k_eff) | p(halt@1) | mean halt step |
|---|---|---|---|---|---|
| halt_off (ceiling) | 0.9990 / 0.9915 / 0.9985 (**0.996**) | — | — | — | — |
| halt_state | 0.670 / 0.675 / 0.630 (**0.658**) | 0.665 | +0.03 / +0.10 / +0.03 (**0.05**) | 0.999 / 0.980 / 0.982 | 1.00–1.04 |
| halt_traj | 0.481 / 0.629 / 0.620 (**0.577**) | 0.663 | +0.02 / +0.02 / +0.01 (**0.02**) | 0.995 / 0.999 / 0.998 | 1.00–1.02 |

Pre-registered bar 3 fires: `halt_traj` collapses — corr ≈ 0.02 (bar: ≥ 0.8
to win, < 0.5 = collapse), halt mass ≥ 99.5% on step 1 in every seed,
statistically indistinguishable from the blind control. Full trajectory
visibility rescued nothing; there is not even a marginal visibility edge
(traj 0.02 vs state 0.05 — both zero).

Secondary observation: the halting objective ROTS the surrounding model.
Both halting arms' full-depth accuracy falls to ~0.58–0.66 on a task their
halting-free twin solves at 0.996 — with p collapsed onto step 1, the answer
CE trains almost only the step-1 readout, and later-step competence starves.
Collapse is not just a wrong halting policy; it degrades the capability it
was supposed to meter.

## Reading

The graveyard's ACT autopsy is now specific: **the collapse is an incentive
pathology, not an information one.** "Halt now" pays its ponder-cost savings
deterministically on every example; "halt later when the chain is longer"
pays rarely and noisily through the answer CE — the optimizer takes the sure
thing, even when the evidence for continuing sits fully readable in the halt
head's attention window, on a task the same parameters demonstrably solve.
Any future attack on learned halting here must therefore change the payment
structure (and the graveyard already shows naive reward shaping is not
enough), not the halting head's inputs. Explicit effort remains the policy;
the depth dial stays with the caller (#86 made it an open dial).

## Limitations

Toy scale (0.27M params, K=4, one λ_p = 0.2 point, 3 seeds); halting-as-
readout differs from classic ACT's halting-as-compute-cut, so the mapping to
the original graveyard entry is by failure signature (min-depth mass, shaping-
resistant), not identical mechanics. A λ_p sweep was not run (pre-registered
single point); λ_p → 0 trivially removes collapse but also removes the point
of halting. corr uses argmax-halt vs k_eff on held-out data.

## Relation to prior work

- Sharpens the graveyard candidate finding (ACT collapses at small scale,
  survives reward shaping) from "it fails" to "it fails FOR the incentive
  reason": the observability alternative is now measured and dead.
- Components are standard (ACT/PonderNet halting; memory slots) — the
  contribution is the matched-pair separation of observability vs incentive
  as the collapse cause, with the halting context as the single variable on
  a validated-solvable task. Not found in the literature: recorded as novel
  per rule 5 (uncertain → novel). A novel negative is exactly the moat.
- Sibling: #39 (cosine convergence halting) proposes NON-learned halting —
  a rule on state movement, no gradient economics — and is untouched by this
  result; if anything, "done = state stops changing" being detectable here
  (the task's stationarity is trivially visible in the slots) makes the
  rule-based sibling more attractive than the learned version.
