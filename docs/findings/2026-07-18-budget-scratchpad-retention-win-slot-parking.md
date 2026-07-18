# Learned retention under capacity pressure is real: with the writer leak closed, 2 slots for 5 writes solves recall perfectly by parking r_1 in one slot and churning the other — while the 1-slot control's in-vector carry trains unreliably

Status: confirmed
Date: 2026-07-18
Commit: e5f9c80 (claude/116-local-writer — local_writer arms + leak-guard tests)
Run: tiny-config ablation (synthetic, no runs/ id)
Measured with: `JAX_PLATFORMS=cpu python scratchpad_harness.py --arms unlimited_local --seeds 0,1,2 --K 5 --m 7 --steps 2500`, then `--arms budget1_local,budget2_local` (#116); addressing probe: deterministic seed-0 replay printing mean softmax address per write

## Setup

Take 3 of the #63 phase-2 question, with BOTH prior invalidators fixed: the
eval-target bug (PR #76, `final_target`) and the writer side channel (#114 —
a full-sequence causal encoder let the last write compute the recall answer
directly, so no slot budget forced anything). The `_local` arms encode each
(a_k, b_k) link as its own independent 2-token sequence (`encode_links`);
write k sees link k + current memory ONLY. Two leak-guard tests pin this
architecturally: link k's encoding is bit-identical regardless of other
links' contents, and the gradient from the final answer to link 1 is exactly
zero once the memory chain is stop-gradiented after write 1. Any correct
recall of `(r_1 + r_K) mod m` therefore proves information survived in
memory across K-1 subsequent writes — there is no other path.

Task and bars pre-registered on #116 (including the pre-run amendment):
recall target `(r_1 + r_K) mod m`, K=5, m=7, dim 64, 2500 steps, seeds
{0,1,2}, slots-only readout. Ceiling first (rule 4): `unlimited_local` must
reach ≥~0.9 or stop. Win bar: budget2_local beats budget1_local by
≥2σ_pooled and approaches the ceiling. Kill bar: budget2_local at
budget1_local's level. Amendment (before any number was seen): budget1 at
chance is NOT required for validity — with local writers, budget1 can
legitimately retain by re-copying r_1 through its single write vector; either
outcome is informative.

## Evidence

Held-out final-answer accuracy (chance = 1/7 = 0.1429), 3 seeds:

| arm | memory | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|---|
| unlimited_local (ceiling) | 5 slots, append-only | 0.9995 | 1.0000 | 1.0000 | 0.9998 | 0.0003 |
| budget2_local (bet) | 2 slots, 5 writes | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| budget1_local (control) | 1 slot, 5 writes | 0.8740 | 0.1548 | 0.1460 | 0.3916 | 0.4178 |

Win bar: budget2 − budget1 = +0.608, σ_pooled = 0.295 → **2.06σ** — clears
the pre-registered 2σ, and budget2 doesn't approach the ceiling, it EQUALS it
(three exact 1.0000s, σ = 0). The kill bar does not fire. Per-slot grades are
≥0.994 at every write in EVERY seed of every arm — including budget1's two
chance seeds, so what fails there is only the carry, never the chain
computation.

Learned addressing policy, budget2_local seed 0 (bit-exact replay, confirmed
by reproducing final_acc 1.0000; mean softmax address = fraction of each slot
overwritten, held-out batch):

| write | slot 0 | slot 1 |
|---|---|---|
| 1 (r_1) | 0.396 | 0.604 |
| 2 | 0.994 | 0.006 |
| 3 | 0.996 | 0.004 |
| 4 | 0.996 | 0.004 |
| 5 | 0.995 | 0.005 |

Write 1 lands (mostly) in slot 1; writes 2–5 route ~99.5% into slot 0,
touching slot 1 at ~0.5% per step (r_1's imprint survives ≈0.995⁴ ≈ 98%
intact). The write-1 spill into slot 0 is harmless — write 2 overwrites slot
0 anyway. This is slot-parking: protect one address after its deposit, churn
the other.

## Reading

**Retention under capacity pressure works, and it is an addressing behavior,
not a vector-capacity behavior.** Given somewhere to park (S=2), the model
learns keep-vs-evict reliably — perfectly, on every seed — with no forget
gate, no retention bonus, nothing but capacity pressure and the final-answer
loss. Forced to instead thread r_1 through its working vector (S=1, forced
full overwrite), it usually loses it: 2/3 seeds at chance with flawless
per-step computation. A 64-dim vector has plenty of ROOM for two mod-7
values; what's missing at S=1 is a stable optimization path to using that
room. The bits that matter are where the gradient can find them — a physical
slot the addressing can choose not to touch — not merely somewhere they'd
fit.

Secondary: the #114 budget2 seed instability (0.14–0.97 under the leaky
full-context writer) is GONE — σ = 0.0000 here. Consistent with the leak
having offered two competing gradient paths (compute-at-the-end vs. retain);
with only retention available, training is calm. Observation at n=3, not a
claim.

The 2.06σ margin is thin because the control's bimodality inflates σ_pooled;
the statistics-free statement is stronger: budget2_local = ceiling, exactly,
three for three.

## Limitations

Toy scale: n=3 seeds, single K=5/m=7/dim-64/2500-step point. The addressing
probe covers one seed (deterministic replay). Whether S=1's in-vector carry
becomes reliable with more steps/dim, and whether parking survives larger K
(longer protection horizon) or S=2 with TWO values to protect, are untested —
larger K was pre-named on #116 as the next knob if both arms saturated; only
budget1 didn't, so the current point already separates the arms.

## Relation to prior work

- Completes the #63 phase-2 arc: phase 1 (2026-07-05, overwrite-is-free)
  showed capacity costs nothing when nothing needs keeping; this shows the
  complement — when something must be kept, the model chooses correctly.
  The two prior invalid attempts: eval-target bug
  (2026-07-05-…-recall-inconclusive-…, retracted), writer leak
  (2026-07-16-…-writer-leaks).
- Settles the locus question left open by the #114 finding (amended, PR
  #117): with the writer architecturally unable to precompute the answer, the
  0.9998 ceiling proves the READOUT retrieves two slots and combines them.
- The graveyard contrast (2026-06-13-cross-window-hunch-inert): the hunch's
  optional forget/blend gate collapsed because gradients could route around
  memory entirely. Here the memory pathway is mandatory and capacity does the
  forcing — same desideratum (learned retention), opposite wiring, opposite
  outcome. That inversion — make memory the only path and let capacity
  pressure teach the policy — is the design lesson to carry into the LM
  scratchpad (#102).
