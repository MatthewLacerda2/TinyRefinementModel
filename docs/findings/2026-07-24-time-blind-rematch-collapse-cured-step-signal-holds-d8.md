# Per-pass grading cures the time-blind collapse — but the step signal still earns full trained depth, while the time-blind arm length-extends better (+2.3σ at d16)

Status: confirmed
Date: 2026-07-24
Commit: 61f197d (feat/138-time-blind-rematch — `time_signal="none"` arm in CausalRefiner + harness)
Run: tiny-config ablation (synthetic, no runs/ id)
Measured with: `PYTHONPATH=. python ablation_harness.py --task statetrack --dim 96 --steps 2500 --seed {0,1,2} --time-signal {sinusoidal,none} --per-pass-loss` with `--depths 1,2,4,8` (leg A) and `--depths 8 --test-seq 48 --eval-depths 12,16` (leg B); GPU, f32, log `aux_138_protocol.log` (#138; launcher `aux_138_launch.sh`)

## Setup

#97's third arm found that a fully time-blind refiner (no step signal at all — the
refine block conditions only on the state) matched sinusoidal on 2 of 3 seeds but
**deterministically collapsed during depth-8 training on seed 2** (0.277 / 0.238,
reproduced twice), reading as "the step signal is a training stabilizer, not a
computational necessity" (ROADMAP graveyard, PR #139). Per-pass grading (#75) was
separately confirmed as the stabilizer for deep-recurrence training
(`2026-07-12-per-pass-stabilizer-confirmation.md`). This rematch asks the question
those two results assemble: **does time-blind + per-pass grading train stably, and
if so, does it reach parity?**

One variable, matched pairs: control `sinusoidal` + `--per-pass-loss`, treatment
`none` + `--per-pass-loss`, statetrack, dim 96, 2500 steps, seeds {0,1,2}, all runs
on one backend (GPU). Pre-registered on #138 before any run:

- **KEEP:** no collapsed seed (trained-d8 accuracy ≥ 0.5 per seed) AND within
  2σ_pooled of the control at every trained depth {1,2,4,8} and at {12,16} on the
  length-48 split.
- **KILL:** any seed collapses, or loses ≥2σ_pooled anywhere.

## Evidence

**Stability — the collapse mode is gone (3/3 seeds).** Time-blind trained-d8
per-seed: 0.9600 / 0.9539 / 0.9665 — including seed 2, the exact seed that
cratered twice in #97. The #97 crater was 0.24–0.28; the collapse bar (0.5) is
nowhere in sight.

**Leg A — parity at trained depths** (3-seed mean ± σ; σ_pooled = √((σ²_n+σ²_s)/2)):

| depth | none | sinusoidal | Δ | verdict |
|---|---|---|---|---|
| 1 | 0.8089±0.0037 | 0.8050±0.0125 | +0.0038 (+0.4σ) | parity |
| 2 | 0.9220±0.0354 | 0.9257±0.0078 | −0.0037 (−0.1σ) | parity |
| 4 | 0.9807±0.0062 | 0.9806±0.0054 | +0.0002 (+0.0σ) | parity |
| 8 | 0.9601±0.0063 | 0.9868±0.0013 | **−0.0267 (−5.9σ)** | **KILL bar fires** |

**Leg B — trained d8, length-48 split:**

| eval depth | none | sinusoidal | Δ |
|---|---|---|---|
| 8 | 0.7863±0.0281 | 0.7557±0.0287 | +0.0306 (+1.1σ) |
| 12 | 0.8172±0.0425 | 0.7597±0.0217 | +0.0576 (+1.7σ) |
| 16 | 0.8207±0.0482 | 0.7275±0.0315 | **+0.0932 (+2.3σ)** |

Every time-blind seed *gains* accuracy from never-trained loops 9–16 (e.g. seed 2:
0.801 → 0.855 → 0.866); every sinusoidal seed fades or stays flat (seed 0:
0.736 → 0.694 at d16). The verdict is **KILL** as pre-registered — the −5.9σ loss
at trained depth 8 fires the kill bar, and one global arm can't be adopted on a
split verdict. Sinusoidal stays the production signal.

## Reading

Three separate results in one grid:

1. **The stabilizer hypothesis is confirmed, on the motivating seed.** Per-pass
   grading removes the time-blind training collapse entirely. #97's reading
   ("stabilizer, not computational necessity") was right about the failure mode
   and its cure.
2. **But stability ≠ parity.** Even trained stably, the step-blind arm gives up
   0.027 at trained depth 8 — cleanly, 5.9σ, all seeds — while matching at 1/2/4.
   The step signal is doing real work exactly where recurrence is deepest and
   in-distribution. "Which pass is this" is not fully recoverable from the state
   alone at d8, or at least not learned from this budget.
3. **The extensibility inversion (unexpected).** In #122 (final-loss-only),
   *sinusoidal* was the arm that converted extra loops into accuracy. Under
   per-pass grading the roles flip: the time-blind arm extends (+2.3σ at d16) and
   sinusoidal's extension advantage disappears. A step-conditioned model appears
   to co-adapt to "step 8 is the last step" — per-pass supervision teaches every
   pass to be a good stopping point, and only the arm with no step signal can
   exploit that anonymously at depths it never saw. At n=3 toy scale this is an
   observation with one significant point (d16), not a mechanism claim.

## Limitations

Toy scale (statetrack, dim 96, 3 seeds, one training depth for leg B). The d16 win
is +2.3σ on n=3 — real by the repo's bar but barely; d12 (+1.7σ) is not
individually significant, only directionally consistent (every seed, both eval
depths). No LM-scale read. The d8 deficit could conceivably close with longer
training (the arm matches at d4 where the task saturates); untested. The
extensibility inversion makes the *combination* (which time signal × which
supervision) matter for the base run — measured here only at 2500 steps.

## Relation to prior work

- Confirms #97's stabilizer reading and its licensed rematch (#138), and extends
  `2026-07-12-per-pass-stabilizer-confirmation.md` to the time-blind failure mode.
- Reverses #122's extensibility assignment *under per-pass supervision* — the two
  findings are consistent because they measured different loss regimes; together
  they say the extension behavior belongs to the (signal × supervision) pair, not
  to the signal alone.
- Step/timestep conditioning is standard in recurrent-depth models (Universal
  Transformer line) and diffusion; deep supervision is standard separately. A
  matched step-signal ablation *under* deep supervision in a depth-recurrent
  model, with the collapse-cure and the extensibility inversion, is not something
  we can find reported — recorded as novel per rule 5 (uncertain → novel).
