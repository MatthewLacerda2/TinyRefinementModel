# Architecture & Experiment Plan — Phases 2 and 3

Phase 1 (attention scale fix, explicit decoder slot positions, `--steps` inference
flag) is implemented on this branch. **Everything below is proposed, not applied.**
Review against the latest training logs before approving.

## Read the logs first

These metrics from `metrics.csv` decide which items below matter:

| Metric | What it tells you |
|---|---|
| `grad_norm` trend | A slow fade toward zero means f16 gradient underflow → loss scaling moves up in priority |
| `mean_halt_step` vs opt step | Pinned at max depth = the halt probe is dead weight; falling over time = ACT is alive |
| `forget_density` | Should be moving since the wiring fix; frozen near 0.5 means the gate gets no useful signal |
| `ce1` vs `token_loss` gap | ce2 consistently above ce1 means the carried hunch hurts rather than refines |
| `temporal_drift` | Near zero = the reasoning loop barely changes the slots = the loop is decorative |

**Caveat:** all logs so far were produced under the double-scaled attention bug.
Treat absolute values with suspicion; trends are still informative.

---

## Phase 2 — Make the experiment answer the thesis

### 2.1 The steps A/B (no code beyond Phase 1)
Same checkpoint, same prompts: generate with `python infer_local.py --steps 1`
and again with `--steps 8`, put transcripts side by side in `auxmd.md`, judge by
reading. This is the cheapest existing test of whether the latent loop does
anything. If depth 8 reads no better than depth 1, every mechanism in the loop
is on notice.

*Effort: none (tooling shipped in Phase 1). Risk: none.*

### 2.2 Make the refinement loss measure refinement
Today it penalizes the second text segment scoring worse than the first — but
those are different tokens with different difficulty, so the signal is mostly
noise about which half of the document was harder.

**Proposal:** every Nth micro-step (N=8), run segment 2 twice — once with the
carried hunch, once with fresh slots — and penalize only when carried is worse
than fresh **on the same tokens**. Delete the cross-segment term. This directly
asks "did the scratchpad help," which is the question the loss pretends to answer.

Cost: one extra forward pass on one segment every 8 micro-steps (~6% compute).
Gate the new term behind the loss-wiring test like every other component.

*Effort: about a day, carefully. Risk: moderate — touches the loss.*

### 2.3 ACT sanity pass (only if logs show halting is dead)
The ponder penalty ramps over opt-steps 2000–5000, but the depth curriculum
only unlocks depth above 2 at step 4000 — the halt probe spends most of its
ramp with almost nothing to halt over.

**Proposal:** tie the ponder ramp to the depth curriculum (start ramping when
depth reaches 4) instead of fixed step boundaries. Skip entirely if
`mean_halt_step` shows healthy movement.

*Effort: small. Risk: low.*

### 2.4 Flagged, not proposed: the loop is still optional
The decoder re-reads the full token sequence alongside the scratchpad, so
training can ignore the scratchpad entirely and still predict well. If 2.1
shows identical transcripts at depth 1 and 8 AND `temporal_drift` is near zero,
the next conversation is restructuring the decoder so part of its input only
arrives through the slots. Big change; decide only with A/B evidence in hand.

### 2.5 Open question from Phase 1 (decide at home)
Decoder slot keys sit at the RoPE positions of the **final reasoning step at
full depth** (now explicit in `model.py`), even while the depth curriculum runs
fewer steps — so early in training, slots are written at one set of positions
and read at another. Phase 1 preserved this behavior, only making it visible.
Worth deciding deliberately: key the slots at the positions of the step that
actually produced them, or keep the fixed home. Either is defensible; the
current state was an accident that happened to work.

---

## Phase 3 — Throughput (independent of architecture)

### 3.1 Chunked cross-entropy
Compute the loss over vocabulary chunks so the full 512-by-100k logits tensor
never materializes. This is the keystone: it removes the main OOM ceiling and
unlocks real batching on 6GB.
Verify by comparing chunked vs unchunked loss on a fixed batch to float
tolerance before trusting it.

*Effort: about a day. Risk: moderate — loss path.*

### 3.2 Real batching
After 3.1: raise `BATCH_SIZE` to 8–16 and cut `ACCUMULATION_STEPS` to keep the
effective batch at 128 sequences. Identical training math, several times the
throughput — the GPU is mostly idle at batch 1.

*Effort: trivial after 3.1. Risk: low (hunch cache is already batch-sized and asserted).*

### 3.3 Deferred: 32k vocabulary
Biggest remaining single win (the 100k-wide LM head dominates per-step compute
at this model size), but it requires re-tokenizing the dataset through prefill.
Do it once, right before the first long committed run — not before.

---

## Reasoning-depth recommendations (discussed 2026-06-10)

**Is a max of 8 steps right for this model size?** As a ceiling, yes — keep it.
The reasoning loop runs over 32 slots, not 512 tokens, so each extra step is
cheap relative to the encoder/decoder (full depth adds roughly half the cost of
the rest of the model, not a multiple of it). The honest answer to "how many
steps does a 50M-param model benefit from" is nobody knows — that is exactly
what the `--steps` A/B and `mean_halt_step` are for. Expect the sweet spot at
2–4; let ACT argue for more.

**Start the depth curriculum at 2 instead of 1?** Yes. At depth 1 there is no
refinement: the forget gate sees one transition, the halt probe has nothing to
decide, and the model spends its first thousand opt-steps learning that the
loop does nothing — then gets distribution-shocked when depth becomes 2. Depth
2 is the smallest configuration where "refine the scratchpad" exists as a
behavior, and the extra cost early is small.

Proposed change in `schedules.py` when approved:
`DEPTH_CURRICULUM = [(4000, 2), (8000, 4)]` with `DEPTH_FINAL = 8`.

---

## Suggested order

1. Verify Phase 1 on the training machine: `pytest` (new attention-scale test
   must pass), then a short smoke run.
2. **Fresh run** — checkpoints trained under the scale bug are not worth
   continuing.
3. 2.1 transcripts while it trains; 2.2/2.3 informed by logs.
4. Phase 3 before the first long committed run.

---

## Inference snapshot — 2026-06-13 (stopped run, ~3k opt steps)

Checkpoint: `run_20260611_234058` step 384639 (~opt step 3005, train CE ~4.77).
Tokenizer cl100k_base. `infer_local.py` defaults: temperature 0.5, top_k 50,
top_p 0.9, max_new_tokens 256, hunch refresh every 4 tokens, fixed depth 8.
This is a pre-instruction-tuned base model at ~3k steps — it continues text, it
does not answer. Captured to confirm "how it would've responded," not as a
quality bar.

**Prompt: "What is the capital of France?"**
> What is the number of the following? What is the number of the two numbers?
> ### What is the number of the number?
> ### How do you answer? ### How do you write?
> ### What are the number of numbers? [...loops "What is the number of numbers?" to the 256-token cap]

**Prompt: "What are the colors of the American flag?"**
> The first of the following is the 8th century, with the 8th century, is this
> is the 8th century, and is the 8th century, which is the 8th century, the 3th
> century, and the 8th century, which is the 2th century... [...loops "the Nth century" to the cap]

**Prompt: "Peter Piper picked a peck of"**
> the 1st century, and the 1st century, the 1st century, the 1st century, the 1st
> century... [...degenerates immediately into "the 1st century" repeated to the cap]

**Read:** no knowledge, no instruction-following, severe repetition/looping, and
zero recall of the tongue twister. Expected for an undertrained base model — the
backbone learned local fluency (grammatical fragments, "### " markdown headers
from web data) but nothing factual or long-range. Nothing here reflects on Plan A
either way; it's a snapshot of the abandoned architecture's base at 3k steps.
