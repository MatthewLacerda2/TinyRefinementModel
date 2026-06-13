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

Result (gate_bias=-2): depth 1 0.871, depth 2 **0.959**, depth 4 0.982, depth 8
**0.274**. Hypothesis **falsified** — retention bias did not rescue depth 8 (still
collapsed, slightly worse). Depth 2 did improve. So the collapse is not a gate-init
issue. Collapse to *near-chance* (not partial) points to training instability, not
slow convergence — next probe: lower LR at depth 8 (LR-driven divergence is the
likeliest culprit for deep unrolled recurrence).

## Run 4 (in progress) — depth-8 LR probe

depth 8 at lr 5e-4 and 1e-3 (vs the 2e-3 default). If a lower LR recovers depth 8,
the fix is a depth-aware LR schedule (trivial). If depth 8 still collapses at low
LR, the instability is structural (signal degradation through the shared loop) and
needs a design decision — escalate to Matheus rather than brute-force overnight.

Result: depth 8 @ lr 2e-3 = 0.374 (collapsed); @ **lr 1e-3 = 0.985**; @ lr 5e-4 =
0.981. **The collapse was LR-driven instability, not structural.** At a sane LR
depth 8 solves the task, matching depth 4 and far above depth 1. Fix: deeper
recurrence needs a lower LR (depth-aware LR schedule). The production schedule's
peak LR is already 1e-4 (10x below the rescuing 1e-3), so this likely won't even
manifest at real scale — but a depth-aware LR is the safe rule.

## Run 5 (in progress) — clean confirmatory depth curve, single LR

Depths 1,2,4,8 all at lr 1e-3 (the stable LR), to give one rigorous monotonic
curve instead of the mixed-LR numbers above. This is the capstone figure.

| depth | val_acc | val_ce |
|---|---|---|
| 1 | 0.8480 | 0.3651 |
| 2 | 0.9194 | 0.2010 |
| 4 | **0.9837** | 0.0510 |
| 8 | 0.9755 | 0.0755 |

**Definitive: depth helps monotonically 1 -> 2 -> 4** (+0.136 acc, 7x lower CE),
and depth 8 holds strong (0.976, no collapse) at a stable LR. Diminishing returns
past depth 4 on this task (4 ≈ 8), so depth 4 is the sweet spot here.

## Conclusion (2026-06-13 overnight)

Causal within-window depth recurrence (Plan A) does real work where the
cross-window hunch did not: on a task that requires sequential composition, more
recurrence iterations monotonically improve held-out accuracy. The depth-8
instability was LR-driven, not structural — fixed by a lower LR (use a
depth-aware LR schedule). Open items for Matheus: (1) wire CausalRefiner into the
production fineweb trainer (different interface from UniversalReasoner), (2) run
the full test battery on that integration, (3) only then the gated cl100k run.
Not started — they need review/go per docs/AUTONOMY.md.
