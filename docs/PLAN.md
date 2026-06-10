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
