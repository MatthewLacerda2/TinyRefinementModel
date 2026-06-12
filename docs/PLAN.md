# Refactor and Gate Fixes Plan

## Clean Code Audit
- **Goal:** Improve repository organization, readability, and state-of-the-art patterns.
- **Actions:**
  - Create a cohesive directory structure (`src/model/`, `src/utils/`, `src/training/`).
  - Move logic from `train_local.py` into dedicated modules.
  - Implement cleaner configuration management.

## Gate Fixes (Hunch Gate & Forget Gate)
- **Goal:** Analyze and optimize the 'hunch gate' and 'forget gate'.
- **Hunch Gate Analysis:**
  - The hunch gate uses a linear projection on the concatenated previous hunch and context.
  - Potential fix: Introduce non-linearity and stabilize gating.
- **Forget Gate Analysis:**
  - The forget gate is currently aggressive.
  - Potential fix: Evaluate state vanishing issues and consider mechanism adjustment.

## Future Work: Incremental Decoding (KV Cache)
- **Context (2026-06-09):** The unused KV-cache machinery in `RotaryAttention` was removed
  during the Phase 2 dead-code cleanup (it was never exercised — `use_cache=True` appeared
  nowhere). Inference currently re-runs the full forward pass over all `MAX_SEQ_LEN`
  positions for every generated token.
- **Goal:** A real incremental-decode path designed for this architecture, not a flag flip.
- **Design questions to answer first:**
  - How the hunch cache interacts with per-token decoding (currently refreshed every
    `HUNCH_REFRESH_EVERY` tokens in `infer_local.py`).
  - Whether the reasoning loop over slots must re-run per token (it reads the full
    encoder sequence as cross-attention context) or can be amortized.
  - Slot KV positions advance per reasoning iteration — a cache must account for the
    extended RoPE position scheme.

## Roadmap additions (2026-06-11) — validated findings from the cross-machine review

### Slot read/write position mismatch (decide deliberately)
Slots are written at the RoPE positions of whichever reasoning step produced them,
but the decoder always reads them at fixed negative positions that wrap around the
RoPE cache onto the final-step, full-depth positions. Under random-depth training
the write position now varies every micro-step while the read position stays pinned
at depth 8. This was an accident that happened to work, not a decision. Options:
key the slots at the positions of the step that actually produced them, or keep the
fixed home. Decide with depth-curve evidence in hand, not before.

### The seg1-vs-seg2 CE gap is confounded — build the clean diagnostic
`seg1_ce` vs `token_loss` compares different tokens with different intrinsic
difficulty, so the gap mostly reflects which half of the document was harder, not
whether the carried hunch helped. Do not read it as a refinement signal. The clean
measurement: run segment 2 twice — once with the carried hunch, once with fresh
slots — and compare on the same tokens. Build it as an offline diagnostic (sibling
of `tools/eval_depth_curve.py`: it isolates the hunch cache the way the depth curve
isolates depth), not as a loss term.
