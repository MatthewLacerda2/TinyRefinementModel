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

## Run 2 — 2026-06-13 — state tracking (dim 96, enc 2, seq 24, 6000 steps, held-out eval)

| depth | val_acc | val_ce |
|---|---|---|
| 1 | 0.8772 | 0.2941 |
| 2 | 0.8965 | 0.2553 |
| 4 | **0.9843** | 0.0473 |
| 8 | 0.3743 | 1.3416 |

**The headline positive result of the night:** on a task that genuinely needs
sequential composition (non-commutative, no sum shortcut), **depth helps
monotonically 1 -> 2 -> 4**: +0.107 accuracy and 6x lower CE from depth 1 to 4.
This is the depth-helps signal the cross-window hunch never produced — causal
within-window depth recurrence does real work. Plan A's core thesis is supported
on the right instrument.

**But depth 8 collapses to chance (0.37, chance=0.20)** — the same collapse seen
on parity in run 1, so it is reproducible: deep weight-shared recurrence is
training-unstable past ~4 with the current (balanced-gate) setup. Production
MAX_STEPS_LIMIT=8 sits squarely in the broken regime, so the cl100k run stays
held until this is fixed.

## Run 3 (in progress) — stability probe: gate biased to retention

Hypothesis: a balanced gate (sigmoid(0)=0.5) compounds large updates over many
iterations and diverges. Biasing the gate to retention at init (gate_bias=-2 ->
sigmoid=0.12, small early steps) should stabilize deep recurrence. Re-running the
full depth sweep with --gate-bias -2; success = depth 8 recovers toward depth-4
accuracy without hurting depths 1-4.
