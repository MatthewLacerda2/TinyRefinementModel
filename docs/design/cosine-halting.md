# Design — convergence (cosine) halting for the serial scratchpad (#39)

Status: toy-proof stage. Pins the exact wiring and the pre-registered criteria
before any run, per the AUTONOMY design-doc-first rule. Builds on the #38 win
(docs/findings/2026-07-03-serial-scratchpad-beats-controls.md), which explicitly
unblocked this issue.

## The idea, in one paragraph

Let the model decide its own reasoning depth: stop writing scratchpad slots when
the latent stops moving. After writing slot k, compare it to slot k−1 by cosine
similarity; when the similarity rises above a threshold τ, the state has
committed — halt, and read the answer from the slots written so far. This is a
**deterministic** stopping rule in the DEQ/fixed-point family — explicitly NOT a
learned halt gate (ACT collapses at small scale and is killed in the roadmap).
Inference-time only: training is untouched, so the #38 models and their proof
are unaffected.

## Why halting needs a variable-difficulty split

On the #38 task every instance needs exactly K live steps — a correct halter
should never fire early, so uniform instances measure only false positives.
The payoff case ("fewer refine steps on easy tokens") needs instances that
commit early. The in-distribution way to get them:

**Identity-tail split (`affine_chain_varlen`).** Effective length L ~ U{1..K};
steps 1..L are sampled as in the base task, steps after L are the identity map
(a=1, b=0 — both legal token values, so the tokens stay inside the training
alphabet). Then r_k = r_L for all k > L: the true sub-results stop moving at
step L, the graded slots should converge, and a working cosine detector fires
at write L+1 (it needs to see one repeated state). Oracle average writes at
K=4: E[min(L+1, K)] = 3.25 vs the fixed 4.

## Wiring

- **Models:** the exact #38 serial arm, trained unchanged on the uniform task
  (seeds {0,1,2}; the fixed-K uniform accuracy must reproduce the recorded
  0.9922 ± 0.011 — that reproduction is the anchor, as in #62/#67/#79).
- **Signal:** c_k = cosine(s_k, s_{k−1}) on the raw slot latents, f32, k ≥ 2.
- **Rule:** halt at the first k with c_k > τ; else run all K. The readout
  context is tokens + slots 1..k.
- **Exactness:** slots are write-once and causal (slot k reads only tokens and
  slots < k), so truncating a full run's first n slots is *bit-identical* to
  having stopped after n writes — one forward serves every threshold. Guarded
  by a truncation-equivalence test.
- **Threshold sweep (fixed grid, pre-registered):**
  τ ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999}.

## Measurements

1. Uniform split (the #38 held-out pool): fixed-K accuracy (reproduction
   anchor), and halting accuracy + mean writes per τ (false-positive check).
2. Identity-tail split: fixed-K accuracy (the control — also the confound
   guard below), and halting accuracy + mean writes per τ.
3. **Mechanism readout:** on the identity-tail split, the distribution of c_k
   at *converged* steps (r_k = r_{k−1}) vs *computing* steps (r_k ≠ r_{k−1}).
   The separation of these two distributions is the mechanism made visible;
   if they overlap, no threshold can work and the verdict table will show why.
4. Diagnostic only (recorded, no claim): cosine of the slot grade logits
   (semantic commitment) at the same steps — a candidate fallback signal if
   the raw-latent cosine is polluted by the slot-index embedding.

## Pre-registered criteria

One **global** τ, shared across seeds, judged on 3-seed means (no per-seed
threshold cherry-picking). σ_pooled from the seed spreads of the two compared
readings, as in prior findings.

- **KEEP:** some τ from the grid satisfies all three —
  (a) identity-tail halting accuracy within 2σ_pooled of the identity-tail
      fixed-K accuracy (halting costs nothing where it should fire),
  (b) identity-tail mean writes ≤ 3.5 (a real saving; oracle is 3.25), and
  (c) uniform halting accuracy within 2σ_pooled of the uniform fixed-K
      accuracy (no false-positive damage where it should not fire).
- **KILL:** no τ on the grid satisfies (a)+(b)+(c) → the cosine of the slot
  latents does not separate commitment from computation on this wiring;
  negative-result finding.
- **Confound guard (contingency, pre-registered):** if the fixed-K accuracy on
  the identity-tail split is itself below the uniform accuracy by ≥ 2σ_pooled,
  the split is probing generalization, not halting. Then — and only then —
  train a matched pair on a 50/50 uniform/identity-tail pool (same seeds,
  everything else identical) and re-run the whole ladder on it, reporting both.

## Budget

3 CPU training runs (the serial arm, ~minutes each) + eval-only halting ladder.
Well under an hour.

## Outcome (2026-07-15)

The KILL branch fired — no τ satisfied (a)+(b)+(c). Non-novel per rule 5
(the mechanism mismatch is derivable: a fixed-point detector on a write-once
memory), so the record is PR #96 + the Graveyard tombstone in
`docs/ROADMAP.md`, not a findings entry.

## Files

| Piece | Where |
|---|---|
| Varlen task + halting eval + `--halting` runner | `scratchpad_harness.py` |
| Truncation-equivalence, varlen-task, halt-rule tests | `tests/test_scratchpad_harness.py` |
| Verdict | PR #96 (kill) + tombstone in `docs/ROADMAP.md` Graveyard — non-novel per rule 5, so no findings entry |
