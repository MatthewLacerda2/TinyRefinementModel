# Plan A ablation results log

Branch `feat/ablation-harness`. Each run: train `CausalRefiner` from scratch at a
fixed depth, same budget, report held-out accuracy/CE. Question: does looping the
shared block more times help a depth-needing task?

## Run 1 — 2026-06-13 — cumsum tasks (dim 96, enc 2, seq 24)

| task | depth 1 | depth 2 | depth 4 | depth 8 |
|---|---|---|---|---|
| cumsum mod 5 (3000 steps) | 0.9851 | 0.9572 | 0.9868 | 0.9874 |
| parity / mod 2 (4000 steps) | 0.7381 | **0.7909** | 0.6535 | 0.5784 |

**Read:** inconclusive, leaning negative, but the tasks are miscalibrated:
- cumsum-mod-5 is too *easy* — depth 1 already solves it (98.5%), so there's no
  headroom for depth to show value. A cumulative sum is computable by one
  attention layer (uniform attend + sum), so it doesn't need sequential depth.
- parity is too *hard/noisy* — nothing solves it, and crucially depth **degrades**
  past 2 (depth 8 ≈ chance). That is the documented instability of deep
  weight-shared recurrence (looping a shared block 8× without depth-aware
  init/LR), not evidence about whether depth helps the function.

**Decisions forward:**
- Do NOT launch the cl100k run — proof gate is not green.
- Need a task depth-1 genuinely *cannot* shortcut: non-commutative state tracking
  (permutation composition), where the answer requires sequential composition.
- Watch whether the depth-8 degradation persists on the harder task. If it does,
  Plan A needs a stability fix (e.g. depth-scaled residual / lower LR for deep)
  before high depth is usable — a real finding worth knowing before any big run.
