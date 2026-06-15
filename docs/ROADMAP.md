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
  headroom first. Regenerate the golden-run test (data order changes).

## Explicitly NOT planned
- bf16 *compute*: no tensor-core bf16 on Turing — f16 compute policy stands.
- Reviving the cross-window hunch with auxiliary losses: the gradient rejected
  it twice (see finding); don't throw compute at a mechanism the data refused.
- Chasing Chinchilla token counts as a target: it's a compute-allocation result,
  not a quality threshold; for a fixed model size it prescribes nothing.
