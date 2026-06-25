# Optimization & Scaling Roadmap — Plan A rebuild

Status: the original run (run_20260611_234058) was stopped at ~3k opt steps after
the depth curve showed the cross-window hunch inert
(docs/findings/2026-06-13-cross-window-hunch-inert.md). The architecture is being
rebuilt around Plan A — causal within-window depth recurrence. Sort rule:
dependency is the hard constraint (do X before Y when Y needs X, for code or
validation reasons); then within what's unblocked, lead with what can be done
CONFIDENTLY — small, well-understood, low-risk changes with nothing to "find
out" — before the big uncertain piece. Bank the certain wins, save the
worry-budget for the part that actually needs validating (Plan A).

Goal: prove depth-recurrence earns its compute, THEN spend freed VRAM on scale —
in that order. Scaling an unproven design is the mistake the stopped run avoided.
The freed VRAM doesn't expire; it's the reward for proof, not bait before it.

## How this is tracked

This doc is the **narrative + sequencing rationale** (the why, the order, the proof
gate). Per-item *state* lives in **GitHub issues** (backed up, history, closeable by
commits) labelled by lane so parallelism is explicit:
- **`gpu`** — needs the single RTX 2060; a serial queue, one run at a time. Ablations
  and smokes are minutes and slot around the big run.
- **`cpu`** — code, tests, tooling, docs; runs in parallel to any GPU run.
- **`blocked`** — has an unmet dependency, stated as "Blocked by #N" in the issue body.

The binding constraint is the one GPU, so "parallel" means **one GPU lane + one CPU
lane**, not N-wide fan-out. Issue refs are noted per phase below; `gh issue list
--label roadmap` is the live board.

## Phase 0 — the rebuild itself (code/design, can proceed now; no proof gate)
Done at CURRENT or SMALLER size for fast iteration. None of these are scaling —
they're the architecture + efficiency baseline we then prove Plan A on. Ordered
by confidence/ease within the phase: certain small wins first, the experiment last.

- **known-flaw fixes (do first — small, certain, independent)**: rolling-latest
  checkpoint (ends the save-on-best-only flaw); validation-probe cadence (fires
  every 320 not the configured 64 because it sits inside the every-5-steps
  logging block). Tiny, well-understood, nothing to find out — land them now.
- **bf16 optimizer-state storage (contained, low-risk)** — #18 (cpu): moments stored
  bf16, upcast to f32 for the Adam math (~300MB freed). Turing-safe — storage only,
  tensor cores never see bf16, f16 compute policy untouched. One unknown: optax
  state-dtype handling.
- **chunked cross-entropy (contained, known technique)** — #19 (cpu): prerequisite for
  Phase 2 batching, and frees the seq×vocab logit/softmax activation peak. Do it
  in the rebuild regardless.
- **Plan A architecture (the experiment — biggest, uncertain, must-validate)** — DONE,
  integrated behind `MODEL_ARCH=refiner` (docs/findings/2026-06-14-plan-a-integrated-into-trainer.md):
  loop a shared block causally over the current positions — refine each
  position's representation N times under a causal mask, decode from the refined
  state. No learned halting (ACT collapses at small scale). Keep random-depth
  sampling. **Design doc before any code.** This is the part the worry-budget is
  reserved for; everything above is landed and stable before we touch it.

## Phase 0b — r50k tokenizer — BLOCKED on prefill (user is holding prefill) — #21
- **r50k_base (~50k, off-the-shelf)**: halves the embedding (51.4M → 25.7M
  params) — the single biggest VRAM lever. But it requires re-tokenizing ALL
  prefill data, and prefill is frozen pending the user's plans/questions.
  Cannot proceed until the user unblocks prefill — and the from-scratch training
  run is gated on this, since the run needs the re-tokenized data. (Phase 0 code
  work does not need it.) Custom 32k deferred as a later squeeze.

## Phase 1 — PROOF GATE (nothing in Phase 2 happens until this passes) — #16
- Does looping the causal block N times beat N=1? **Choose the metric with
  care.** Recurrent-depth / latent-reasoning wins are documented on reasoning
  and compositional tasks, NOT on raw web-text perplexity — so a flat fineweb CE
  curve would not by itself mean Plan A failed. Pick a yardstick where multi-step
  inference actually matters before declaring proof or failure.
- The verdict is only readable against a known noise floor — #17 (seed-variance)
  establishes what "no effect" looks like before we trust small CE deltas.

## Architecture bet — the latent scratchpad (the next "what does depth carry?")
Plan A asks whether looping depth helps *at all*. This asks the harder follow-on:
*what should the loop carry?* It is the redesign of the dead slots, built on the
one lesson the inert hunch taught.

**The bypass principle (the lesson from the inert hunch).** Anything the model
can route around, it will. The v1 slots died of exactly this — a side memory the
within-window attention never *needed*, so the gradient starved it (forget gate
collapsed by ~step 750; the documented Recurrent-Memory-Transformer failure
mode). The fix is NOT to *incentivize* using the memory: a soft bonus on the main
loss is optional, and the optimizer takes the easier within-window basin every
time (the gate was rejected twice). A latent memory survives only when it is
**non-bypassable**, and there are exactly two ways to make it so:
1. **Architecture** — it is the *only* route to the answer; the prediction cannot
   be computed without reading it.
2. **A dedicated supervised target** — the slot is graded on its own loss term,
   one that cannot be minimized unless the slot carries the right content. A
   grade, not a bonus — the distinction the hunch blurred and died on.

**Core hypothesis — supervised serial latent scratchpad** — #38. Reasoning
unrolls as a *serial* latent chain: step k's committed state is the *input* to
step k+1 (feed-the-state-forward, à la Coconut — which proves latent reasoning
trains at all). The novel turn is **structured decomposition**: not one
continuous thought but a small ordered set of sub-slots, each written once and
supervised to carry sub-result k — the latent analogue of "split the question
into three answers and solve each in turn." Seriality gives the chain an order;
the per-slot supervised target makes each link non-bypassable (principle #2);
writing only from earlier steps makes it causal by construction (no future leak —
the v1 leak that had to be amputated).

- **Proof gate (its own, independent of Plan A's).** A decompose-able toy task
  whose answer genuinely needs N sequential sub-results (e.g. chained modular
  arithmetic), against two controls: a **parallel-slot** arm (same slots, read
  all at once, no induced order) and a **depth-only** arm (Plan A recurrence, no
  scratchpad). Win = the serial supervised scratchpad beats *both*. **Kill-
  criterion up front:** the write path collapses like the forget gate did, OR no
  gap versus the parallel-slot control. Runs on the tiny ablation harness
  (minutes), so it is a cpu-lane bet, not a real-model run.
- **Parked refinements — gated behind the proof; adding them now confounds it:**
  - *Convergence halting* — #39: stop refining when the latent stops moving
    (cosine of step k vs k-1 below a threshold). This is the Deep-Equilibrium /
    fixed-point family and the 2025 recurrent-depth line (Geiping et al.), NOT
    learned per-token halting (ACT, which is killed below). Not novel as a
    mechanism; worth it for adaptive compute (fewer steps on easy tokens).
  - *Slot dimensionality / "vagueness"* — a wide continuous slot can hold a soft,
    under-specified idea (superposition) where a token must commit; the state
    starts vague and sharpens toward commitment — which is what convergence
    halting detects. But dimensionality buys *capacity*, not *commitment*: it does
    not fix bypassability and is no substitute for the supervised target. A knob
    to sweep once the core works, not a fix on its own.

## Phase 2 — scaling, AFTER proof, one knob at a time (spend the freed VRAM here)
Bang-for-buck order; do singly so each gain is attributable.

- **model core size** (biggest bang) — #22: bigger models are more sample-efficient
  per token. Expect LR/warmup/batch re-tuning; the golden run resets.
- **context window** (expensive) — #23: attention is O(n²) in BOTH compute and
  activation — 512→1024 ~quadruples attention cost. Do alone, after model size,
  only if headroom and the task want it.
- **real batching** (BATCH_SIZE=2, ACCUMULATION=64) — #24: pipeline project, not a
  config flip — each lane is a persistent document stream needing parallel
  loaders + per-lane checkpoint/resume. Needs chunked CE (#19) + freed
  headroom first. Regenerate the golden-run test (data order changes). The
  batch-size-schedule curriculum folds in here; the seq-len-schedule curriculum
  folds into the context-window step (#23).

## Phase 2 — efficiency & recipe levers (post-proof, from the #10 triage)
Optimizer/training-recipe improvements. They are NOT architecture, so they would
**confound the proof if introduced now** — gate all of them behind #16 and apply
to both arms (or only after the verdict). Ordered certain-small-first:
- **cautious weight decay** — #25: near-free update mask (sign-agreement); on top of
  the masked AdamW we already run. Cheapest, try first.
- **Muon optimizer** — #26: orthogonalized-momentum updates for 2D matrices; strong
  recent sample-efficiency/wall-clock wins — the lever a single-GPU run wants most.
  Validate at tiny scale and in f16 (Newton-Schulz likely needs f32).

## Beyond proof + scale — far future
- **RL post-training** — #29: preference / reasoning elicitation. On the AGI path but
  out of sequence — you RL on a base that already has capability to elicit; revisit
  only after pretraining + scaling produce a base worth aligning.

## Explicitly NOT planned
- bf16 *compute*: no tensor-core bf16 on Turing — f16 compute policy stands.
- Reviving the cross-window hunch with **soft auxiliary bonuses**: a bonus on the
  main loss for using an *optional* memory is what the gradient rejected twice
  (see finding); don't throw compute at a mechanism the data refused. The line:
  a *dedicated supervised target* that cannot be minimized without the slot is
  NOT this — that's the legitimate non-bypassable path the latent-scratchpad bet
  (#38) uses. Bonus = dead; grade = allowed.
- Chasing Chinchilla token counts as a target: it's a compute-allocation result,
  not a quality threshold; for a fixed model size it prescribes nothing.

### Killed in the #10 triage (with reasons, so they stay dead)
- **per-token halting / ACT**: collapses at small scale — documented dead-end.
- **MoE / mixture-of-slots**: MoE keeps all experts resident → spends VRAM (scarce on
  6GB) to save FLOPs (abundant) — backwards for this card; slots are tied to the dead
  cross-window arch.
- **logit softcapping**: redundant with the q/k RMSNorm already in place; Gemma-2
  itself dropped it. Marginal stability knob, not a capability lever.
- **diffusion LM**: a different paradigm, not an add-on — adopting it abandons the
  depth-recurrence bet for a second project. (Its useful piece, multi-token
  prediction, was salvaged as #27.)
- **learned / meta optimizers + reactive LR**: fragile, poor-ROI; reactive global LR
  also fights golden-run determinism. Frontier pretraining uses fixed schedules
  (cosine/WSD) + Adam's per-parameter adaptivity — a proper decay schedule covers the
  real intent. Killed.
- **multi-token prediction**: NOT a dead-end, but **scale-gated** — the literature
  (Gloeckle et al.; DeepSeek-V3 at 671B) shows the benefit emerges at ~3B+ and is
  weak/absent for small models, and its inference-speedup payoff isn't our bottleneck.
  Out of the roadmap at 79.6M; reconsider only if we ever pass ~1-3B.
