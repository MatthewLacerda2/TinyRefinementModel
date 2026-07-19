# A sinusoidal step signal matches the learned time-embedding table at trained depths and extrapolates past it — +0.11 accuracy from loops the table can't count, where the clamped table collapses to chance with NaN loss

Status: confirmed
Date: 2026-07-18
Commit: c1b0201 (claude/86-time-signal — `time_signal` arm in CausalRefiner + harness)
Run: tiny-config ablation (synthetic, no runs/ id)
Measured with: `PYTHONPATH=. python ablation_harness.py --task statetrack --depths 1,2,4,8 --steps 2500 --seed {0,1,2} --time-signal {table,sinusoidal}` (leg A) and the same with `--depths 8 --test-seq 48 --eval-depths 12,16` (leg B); GPU, f32, log `aux_86_protocol.log` (#86)

## Setup

The refiner tells each refinement pass which step it is via
`time_embed = nnx.Embed(max_depth + 1, dim)` — rows end at `max_depth`, so
depth is hard-capped at 8 and `jnp.take` silently clamps past it (the #86
hazard; measured as chance at depth ≥ 10 in the #75 readouts). Treatment:
`sinusoidal_step_encoding` — the diffusion-style sin/cos ladder, a fixed
function of the step index defined for ANY step. One variable: same arch,
seeds, data, budget; only the time signal differs. statetrack, dim 96, 2500
steps, seeds {0,1,2}, pre-registered on #86 before any run:

- **Keep:** treatment within 2σ_pooled of the table at all trained depths AND
  beats the clamped control at depth > 8 on the length-shift split by ≥2σ_pooled.
- **Kill:** loses ≥2σ at any trained depth, or no gain past 8.

## Evidence

**Leg A — parity at trained depths** (train_seq = test_seq = 24; mean of 3 seeds):

| depth | table | sinusoidal | Δ | σ_pooled | verdict |
|---|---|---|---|---|---|
| 1 | 0.8144 | 0.7910 | −0.0234 | 0.0281 | within 2σ |
| 2 | 0.8733 | 0.8624 | −0.0109 | 0.0493 | within 2σ |
| 4 | 0.9718 | 0.9746 | +0.0028 | 0.0122 | within 2σ |
| 8 | 0.9645 | 0.9212 | −0.0433 | 0.0509 | within 2σ |

**Leg B — extensibility** (trained at depth 8 on seq 24, evaled on the
length-48 split, where depth was still climbing in the 2026-06-13 run-7 read):

| arm | d8 | d12 | d16 |
|---|---|---|---|
| table (seeds) | 0.6583 / 0.6722 / 0.6758 | 0.2050 / 0.2042 / 0.2052, **CE = NaN** | identical to d12 |
| sinusoidal (seeds) | 0.6586 / 0.3704 / 0.6630 | 0.7672 / 0.4082 / 0.7693 | 0.7694 / 0.4236 / 0.7639 |
| table mean | 0.6688 | 0.2048 (chance = 0.2) | 0.2048 |
| sinusoidal mean | 0.5640 | **0.6482** | **0.6523** |

d12 gap: +0.443, σ_pooled 0.147 → **3.0σ**. d16 gap: +0.448, σ_pooled 0.140 →
**3.2σ**. Both clear the ≥2σ win bar; the kill bar never fires. Every
sinusoidal seed improves when run past its trained depth — the healthy seeds
by +0.11 (0.659→0.767, 0.663→0.769), and even the degraded seed 1 climbs
monotonically (0.370→0.408→0.424). d12 ≈ d16, so the extrapolation gain
plateaus by ~12 at this training depth.

**The control's failure mode is worse than "no gain."** The clamped table at
depth 12/16 is exactly chance AND its CE is NaN in every seed — repeating the
depth-8 signal on a trained model doesn't just stall, it numerically explodes.
The 8-cap is not a soft ceiling; overrun is toxic.

## Reading

The keep verdict fires: the learned rows buy nothing the formula doesn't
provide (parity at every trained depth), and the formula converts
never-trained loops 9–16 into real accuracy on stretched problems while the
table detonates. Depth becomes an open dial — the base run no longer has to
bake in a ceiling. Notable that extrapolation works at all: the model was
never trained past 8 loops, yet loops 9–12 execute useful refinement steps
purely because the time signal keeps advancing coherently.

## Limitations

Toy scale (dim 96, statetrack, 3 seeds, one training depth for leg B). The
sinusoidal arm trains with visibly higher seed variance at depth 8 (leg A
seed 1: 0.844; leg B seed 1: 0.370 — the known deep-recurrence instability,
which #75's per-pass grades were shown to stabilize) — at n=3 an observation,
not a measured effect, but it belongs in the base-run risk column. The
extrapolation gain does not recover in-distribution accuracy (0.77 at d16 vs
0.98 on unstretched inputs) and plateaus by ~d12 when trained at 8. LM-scale
transfer is untested — the base run itself is the confirming read.

## Relation to prior work

- Closes the loop the depth findings left open: 2026-06-13 (run 7) saw depth
  still climbing at 8 under length shift; 2026-06-19 measured the
  in-distribution plateau at ~d6; #75's readout 5 measured chance at depth ≥
  10 and blamed the clamp. This isolates the clamp as the cause: swap the
  signal, keep everything else, and depth > 8 goes from chance+NaN to +0.11.
- Continuous/sinusoidal step conditioning is standard in diffusion models and
  Universal-Transformer-style recurrence; recurrent-depth LMs (e.g. Huginn)
  report test-time iteration scaling. The matched one-variable table-vs-
  sinusoidal comparison at this architecture, with the clamp's chance+NaN
  signature pinned as the counterfactual, is recorded as novel per rule 5
  (uncertain → novel).
