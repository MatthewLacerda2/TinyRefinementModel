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
