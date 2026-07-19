# Fixed-budget recall test is uninterpretable: the unlimited-memory ceiling also sits at chance, so budget2-vs-budget1 cannot be read as a retention result — phase 2 (#63)

Status: retracted — the run is void, not merely confounded. 2026-07-16
addendum (bottom): the eval scored the recall arms against r_K while they
were TRAINED on (r_1+r_K) mod m, so "at chance" is what the table shows
whether the readout failed or solved the task. The readout-capacity
diagnosis below is unsupported; only "this run decided nothing" survives.
Date: 2026-07-05
Commit: c3e9fb25f6ca65e864b722c76277c97b47dba17d  Run: tiny-config ablation
(synthetic, no runs/ id)  Measured with: `python scratchpad_harness.py --arms
budget1,budget2,unlimited --seeds 0,1,2 --K 5 --m 7 --steps 2500` (CPU,
JAX_PLATFORMS=cpu)

## Setup

Design: `docs/design/budget-scratchpad.md`. Recall variant of the affine-chain
task: final target `(r_1 + r_K) mod m`, K=5 so budget2 (S=2) must genuinely
choose what to spare, m=7, dim 64, heads 4, enc 2, 2500 steps, held-out 4096,
seeds {0,1,2}, `read_tokens=False` for all three arms (the #62 wiring — the
answer can only come from memory, closing the tokens-side inversion shortcut).
Three arms: `budget1` (S=1, no retention possible, the no-op control),
`budget2` (S=2, the bet — must park r_1 and churn r_2..r_5 through the other
slot), `unlimited` (K=5 append-only slots, the #38 serial arm — the ceiling,
nothing needs to be evicted). Win bar (pre-registered): budget2 beats budget1
by ≥2σ_pooled AND sits within [budget1, unlimited] approaching unlimited.
Kill bar: budget2 sits at the budget1 level.

## Evidence

Held-out final-answer accuracy ((r_1+r_K) mod 7, chance = 1/7 = 0.1429), 3 seeds:

| arm | params | seed 0 | seed 1 | seed 2 | mean | σ |
|---|---|---|---|---|---|---|
| budget1 | 0.22M | 0.1404 | 0.1460 | 0.1465 | 0.1443 | 0.0034 |
| budget2 | 0.22M | 0.1575 | 0.1360 | 0.1401 | 0.1445 | 0.0114 |
| unlimited | 0.22M | 0.1423 | 0.1470 | 0.1401 | 0.1431 | 0.0035 |

budget2 − budget1 = +0.0002, σ_pooled = 0.0084 → **0.03σ apart** — nowhere near
the 2σ win bar, so the pre-registered kill criterion fires: budget2 sits at the
budget1 level.

**But the load-bearing number is `unlimited`, not the budget1/budget2 gap.**
`unlimited` — every one of K=5 sub-results kept in its own append-only slot,
zero retention pressure, the intended ceiling — is *also* at chance (0.1431,
indistinguishable from all three arms and from 1/7). Per-slot grades show the
model DOES learn each individual sub-result correctly in the well-behaved
seeds (e.g. unlimited seed 0: `[1.000 1.000 0.999 0.999 0.998]`; budget1 seed 1:
`[1.000 0.997 0.997 0.997 0.997]`) — so r_1 and r_K are both being computed and
graded near-perfectly, in memory that unlimited never has to evict from. The
final answer is still chance. The bottleneck is not retention; it is the
readout's ability to compute `(r_1 + r_K) mod 7` from two known-good retrieved
values within this step budget — a capability none of the three arms
demonstrate, ceiling included.

One secondary, non-decisive observation: `budget2`'s per-step grades are far
noisier across seeds than budget1's or unlimited's (σ visible directly in the
per-slot rows) — seed 0 degrades smoothly across all 5 writes
(`[0.674 0.614 0.508 0.420 0.355]`) and seed 1 collapses sharply only at the
last write (`[1.000 0.999 0.986 0.946 0.140]`), while seed 2 is clean
(`[1.000 0.999 0.997 0.996 0.994]`) — versus budget1/unlimited, where every
seed is uniformly high. This is consistent with the addressing competition
making optimization measurably harder even when it does not (yet) show up in
final accuracy, but with the final task itself unsolved for every arm this
cannot be separated from ordinary seed variance with only 3 seeds — noted, not
claimed.

## Reading

This is **not** a clean negative on "genuine learned retention" — it is a
confounded null. The design's own kill criterion (budget2 == budget1) is
technically met, but the diagnostic that makes the whole comparison
interpretable — the unlimited ceiling actually solving the task — never
happened. Per the working agreement (CLAUDE.md, rule 4: "earn the comparison
with a strong baseline"), a baseline that itself sits at chance cannot license
a verdict on the thing being compared against it. The honest read: **this
harness has not yet tested phase 2's hypothesis** — it has shown that
combining two retrieved values via a single cross-attention readout + linear
head does not train, at this dim/step budget, regardless of how much memory is
available to retrieve them from.

## What would fix this (not chased here — new pre-registration, not this run)

Candidates for a follow-up, none attempted post-hoc to avoid goalpost-moving:
- More optimizer steps — 2500 was carried over unchanged from the phase-1/#38
  configs, which never asked the readout to compute a function of two values.
- A wider or deeper `read_block`/`answer_head` — the current readout is one
  cross-attention pass + one linear layer, sized for "retrieve one thing," not
  "retrieve two things and combine them."
- A simpler recall target that needs no arithmetic combination (e.g. predict
  r_1 and r_K as two separate heads, or concatenate-then-predict) to isolate
  "can the readout even see both slots" from "can it compute their sum mod m."
- Confirming the ceiling (`unlimited`) trains before re-running budget1/budget2
  at all — the correct order per rule 4, inverted here because the recall task
  was designed and gated all at once instead of validating the ceiling first.

## Limitations

Toy scale, single step-budget (2500), single dim (64), 3 seeds. The elevated
budget2 per-step variance is suggestive, not evidential, with n=3. No claim is
made about whether learned retention under capacity pressure works or fails —
that question stays open pending a readout that first demonstrably solves the
recall task at unlimited memory.

## Relation to prior work

The graveyard precedent (`docs/findings/2026-06-13-cross-window-hunch-inert.md`)
failed by gradient starvation of an *optional* memory pathway — a different
failure mode than this one, where the pathway is mandatory (per-step grades all
trained near-perfectly) but the final combinator never learned regardless of
memory shape. Worth distinguishing for the record: this is a readout-capacity
gap, not a bypass or a forget-gate collapse.

## Addendum (2026-07-16): eval-target bug voids the table — diagnosis retracted

Found while rebasing PR #76 onto main: `eval_all` computed
`final_acc = mean(argmax == te_sub[:, -1])` for EVERY arm — the recall arms
were trained on `(r_1 + r_K) mod m` but scored against `r_K`. Since r_1 is
~uniform, a model that perfectly solved the recall task would still score
≈ 1/m against the wrong target. **The table above therefore cannot
distinguish "the readout never learned to combine" from "the readout solved
the task and the eval measured the wrong thing."** Everything downstream of
the table is unsupported:

- The readout-capacity diagnosis (## Reading, and the contrast drawn in
  ## Relation to prior work) is retracted — it may still be true, but this
  run is not evidence for it.
- The per-slot grade rows are unaffected (scored against `sub` correctly),
  so "each sub-result was computed and stored near-perfectly" stands.
- Phase 1 (`2026-07-05-budget-scratchpad-overwrite-carries-chain.md`) is
  unaffected: its arms train on the plain chain task whose final target
  really is r_K, so its eval was correct.

The fix (in the PR #76 rebase): loss and eval now share one
`final_target(arm, sub, m)` so they can never disagree, with a regression
test (`test_recall_arms_scored_against_their_trained_target`) pinning an
oracle model to ~zero CE. Phase 2 must be rerun under the corrected eval
before any of the follow-ups listed above are worth spending on — the rerun
might simply show a win (or a clean kill) at zero extra design cost.

Rerun done same day (#114): see
`2026-07-16-budget-scratchpad-recall-rerun-readout-fine-writer-leaks.md` —
`unlimited` solves at 0.991 (the readout-capacity diagnosis above was wrong),
and the S=1 control solves at 0.897, exposing a second, design-level leak:
the token-visible writer can smuggle the combined answer past any slot
budget. Retention stays untested pending a local-context writer.
