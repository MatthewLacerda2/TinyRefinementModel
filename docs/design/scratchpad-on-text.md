# Design — what a scratchpad slot is graded against on text (#89)

Status: design, for the owner's review before anything is built. Builds on the toy
ladder (#38, #62, #67, #73, #79, #95, #116; `docs/design/serial-scratchpad.md`) and
feeds #102, the rung that wires the scratchpad into the language model.

## The question

On the toy task every slot had a true target: slot k was graded against r_k, the k-th
intermediate of an affine chain, and that grade is what made the mechanism work.
Final-answer-only supervision never formed the chain (#67), and the grade without
slots collapsed at the composition point (#79). Text has no r_k. Before any probe is
built, we have to choose what a slot is graded against when the input is language.

## What the answer must satisfy

1. **Non-bypassable.** The grade must be satisfiable only by storing something in the
   slot. A target the model can meet from the token stream alone brings back the
   failure that killed both earlier slot designs (the gradient routes around the
   memory; findings 2026-06-11 and 2026-06-13).
2. **Scalable, per the bitter lesson.** The target should come from the data at any
   scale, not from a hand-built generator whose coverage we choose.
3. **A warm-up budget, not a run-long one.** #73 and #95 already settled that the grade
   is a scaffold: full weight during warm-up, then an anneal to a floor of λ ≈ 0.1,
   started once the probes show the chain has formed. Whatever the targets cost, they
   are paid mostly early.
4. **Plain architecture.** The live model is `plain` since depth recurrence was retired
   (2026-09-12). The slots attach to the plain stack, not to the refiner loop.

## The options

### A. Synthetic problems with known intermediates, mixed into pretraining

Generate multi-step problems rendered as text (entity/state tracking, arithmetic
chains) whose intermediates are known, and grade the slots only on that slice.

- **For:** exactly the condition the toy results proved. The targets are true
  sub-results, so the ladder's findings carry over as they are.
- **Against:** the supervision is hand-built. The slots learn the decompositions we
  chose to generate, and nothing says that transfers to prose, code or math. It is
  the option the bitter lesson warns against: its reach is set by our generator, not
  by the data. It also needs a data pipeline of its own, and at warm-up its cost is a
  share of the mixture.

### B. Grade annealing

This isn't a source of targets. It is the schedule already settled by #73 and #95 (on
the issue it was listed as an option because it could change A's economics). It
applies to whichever option is chosen, so it is no longer a choice.

### C. Targets from the sequence itself: slot k predicts the token k steps ahead

At each position t, slot k is graded on predicting token t + k + 1, with slot k reading
the stream and the slots before it (ordered, write-once, as on the toy).

- **For:** free at any scale, and it comes from the data. Nothing is hand-built. The
  target can't be met by the next-token head, which is already graded on t + 1, so it
  forces the slot to carry something about further-ahead content. The ordering is
  natural, too: predicting t + 3 is easier given a representation of t + 2.
- **This is a known technique under another name: sequential multi-token prediction.**
  Gloeckle et al. (2024) grade K *parallel* heads on t+1..t+K. DeepSeek-V3 (2024)
  chains them *sequentially*: module k takes module k−1's state. So our serial-vs-
  parallel pair, the one that won the toy (#38), maps exactly onto an open comparison
  in that literature: sequential vs parallel future-token slots. That comparison is
  where the novelty is. MTP itself is not novel.
- **Against, stated plainly:** the slots stop meaning "sub-results of a computation".
  They become "what is coming". The toy findings licensed ordered slots with a
  sub-result grade, and they don't transfer automatically to a different grade, so the
  probe has to re-establish the serial-over-parallel gap. There is also a documented
  scale caveat: Gloeckle et al. report that MTP *hurts* small models (their benefit
  appears from roughly 1B+ params). At 148M this may cost next-token quality. Only a
  matched pair on our model can say.

## Recommendation

**C, with A only as an instrument.** C is the option that scales and is the one the
bitter lesson points to. A's value is as a diagnostic: a small synthetic slice with
known intermediates that we *read* with probes (does slot k decode r_k?) but never
train on. That tells us whether the slots formed a chain at all, without making that
chain the objective.

## The probe this would ship

It runs on the real model and the real data (CLAUDE.md: a toy result licenses no verdict
about the language model), at the recipe tier: plain 9 × 960, Muon ×16.667 on 6e-4,
512 opt steps, 3 seeds, matched.

| arm | what it is |
|---|---|
| **vanilla** | today's model, next-token loss only (the control) |
| **parallel** | K future-token slots, each reading only the stream (Gloeckle's shape) |
| **serial** | K future-token slots, slot k also reading slots < k (the #38 shape; DeepSeek's) |

- K = 2 to start: each slot adds one chunked-CE pass through the 50,304-wide head,
  which is the largest VRAM line. The fit is checked against #385's headroom before
  anything else is decided.
- The slot grade runs at full weight for the warm-up, then anneals to 0.1, gated on the
  slot probes (#95). The anneal gate needs a probe that decides "the chain has formed"
  for C's targets, and defining it is part of building the harness.
- **Readout:** tokens to target on next-token held-out CE, the metric every pair uses.
  It asks the only question that matters for the base model: does the scratchpad make
  next-token prediction better? The slots' own accuracy is a readout, not the
  criterion.
- **Pre-registered shape** (the numbers go in the spec): serial must beat vanilla by
  ≥ 2σ to keep the line alive on text. Serial vs parallel at ≥ 2σ is the novel claim.
  Serial ≤ vanilla − 2σ kills the text version of the line and closes #102 as
  superseded by that finding.
- **Prediction registered in advance:** vanilla ≥ serial at this scale and length (the
  MTP scale caveat above), serial > parallel on slot accuracy. My predictions on this
  project have mostly run in one direction per line: optimistic on depth recurrence,
  timid on the recipe pairs. This one is deliberately pessimistic about the model and
  optimistic about the mechanism.

## Questions for the owner

1. **C over A?** C changes what the slots mean. If the scratchpad line is about
   *reasoning sub-results* specifically, A is the faithful option, and it comes with its
   hand-built cost.
2. **Before or after the base run?** The probe is a recipe-tier pair (~21 h per arm set
   on the card), so it fits under the standing permission. But #102's rule says no
   architecture bet is interpretable before a base run reaches the SmolLM2 bar. Running
   the probe now answers "does it work at 512 steps"; the verdict that licenses #102
   would still wait for the base model.
3. **K.** 2 is what fits comfortably. 4 matches the toy.

## Not in this doc

The wiring into the trainer (#102's design doc), the synthetic generator for A's
diagnostic slice, and the anneal controller. Each follows the owner's answers above.
