# A sinusoidal time signal matches the learned table at trained depths and converts never-trained extra depth into accuracy — the 8-cap was an artifact of the table, not the architecture

Status: confirmed (pre-registered keep criterion met: parity within 0.5σ everywhere, +5σ past the cap)
Date: 2026-07-10
Commit: a9ed29d  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with:
`python ablation_harness.py --task statetrack --depths 1,2,4,8 --steps 2500 --seed {0,1,2} --time-signal {learned,sinusoidal,none}` (parity)
`python ablation_harness.py --task statetrack --depths 8 --steps 2500 --seed {0,1,2} --time-signal {learned,sinusoidal,none} --test-seq 48 --eval-depths 12,16` (extensibility; CPU, f32)

## Setup

The refine loop's "which pass am I on" input (#86). Three arms, one variable:
**learned** — the production `nnx.Embed(max_depth+1, dim)` table (control);
**sinusoidal** — `sinusoidal_step_encoding(step, dim)`, the transformer-PE /
diffusion-timestep construction, no parameters, defined for any step;
**none** — time-blind, the block conditions only on the state it refines (added at
the user's direction, pre-registered on the issue before any run). All arms build
an identical param tree and init stream (the table exists even where unused), so
same seed ⇒ same init, pools, minibatch order. statetrack, dim 96, 2500 steps,
seeds {0,1,2}.

Criteria pre-registered on #86: keep a treatment iff it is within 2σ_pooled of the
table at every trained depth {1,2,4,8} (in-distribution) AND beats the clamped
table at depths {12,16} on the length-48 split by ≥2σ_pooled. Preference if
several pass: none > sinusoidal > learned.

**Premise correction, found while wiring (before any run):** the issue assumed
depth > 8 "silently clamps onto the depth-8 signal". On our pinned JAX,
`nnx.Embed`'s out-of-range gather **NaN-fills** instead — every pre-#86
depth-overrun reading (e.g. the 2026-07-05 transfer probe's "d10+ reads as
chance") was argmax over NaN logits, not a clamped signal. The model now clamps
explicitly (`min(step, max_depth)`), so the control arm here genuinely measures
"the table's best effort past its rows is its last row".

## Evidence

Parity — trained-at-depth, in-distribution, mean ± σ over seeds (chance = 0.20):

| arm | d1 | d2 | d4 | d8 |
|---|---|---|---|---|
| learned | 0.809 ± 0.019 | 0.874 ± 0.036 | 0.961 ± 0.024 | 0.972 ± 0.012 |
| sinusoidal | 0.803 ± 0.009 | 0.888 ± 0.046 | 0.949 ± 0.046 | 0.962 ± 0.029 |
| none | 0.799 ± 0.014 | 0.847 ± 0.033 | 0.976 ± 0.003 | 0.728 ± 0.391 |

Sinusoidal deltas vs the table: −0.4σ, +0.3σ, −0.3σ, −0.5σ — indistinguishable.
The learned rows earn nothing the formula doesn't provide.

Extensibility — trained at depth 8 on seq 24, evaluated on the length-48 split:

| arm | d8 | d12 | d16 |
|---|---|---|---|
| learned | 0.653 ± 0.016 | 0.476 ± 0.037 | 0.400 ± 0.033 |
| sinusoidal | 0.644 ± 0.035 | **0.737 ± 0.065 (+5.0σ)** | **0.725 ± 0.087 (+4.9σ)** |
| none | 0.502 ± 0.230 | 0.575 ± 0.293 | 0.587 ± 0.303 |

Two separate facts in that table:

1. **The clamped table doesn't just stop paying past its rows — extra loops
   actively hurt** (0.653 → 0.476 → 0.400). Repeating the last time signal walks
   the state off a cliff. Serving a table-signal model above its trained depth is
   worse than not turning the dial.
2. **The sinusoidal arm converts depth it never trained on into accuracy**
   (0.644 → 0.737 → 0.725): +26 accuracy points over the control at d12, +33 at
   d16, on the split where the 2026-06-13 run-7 curve was still climbing.
   Inference-time depth is an open dial, exactly as diffusion-style continuous
   step conditioning suggested it would be.

**The time-blind arm fails the bar, but instructively.** Mechanical verdict: kill
(+0.5σ / +0.9σ past the cap — no ≥2σ gain). The per-seed detail matters: seeds 0
and 1 *extend as well as sinusoidal* (extend d16: 0.752 / 0.772), but seed 2
collapsed during depth-8 training (0.277 in the parity grid, 0.238 ≈ chance in the
extend grid — deterministically, twice: same init, same data, both grids). The
collapse inflates σ so much that a naive z-test would call the arm "within 2σ" at
trained depths; we decline to launder a 0.69-accuracy crater as noise. Honest
summary: **the state alone is sufficient for the computation (2 of 3 seeds match
everything, and extend), but the step signal acts as a training stabilizer for
deep recurrence** — without it, the depth-8 collapse mode (#77's subject) fires
more readily. This is evidence FOR #77's per-pass-supervision-as-stabilizer test:
if per-pass grading removes the collapse mode, the time-blind arm (zero params,
maximally extensible, no counter for halting to latch onto) deserves a rematch.

## Decision

The refiner's production time signal defaults to **sinusoidal** from this commit
(`REFINER_TIME_SIGNAL` in config.py, env-overridable; the model class default
stays `learned` so the class remains config-free and old harness comparisons stay
reproducible). The param tree is unchanged in all modes, so nothing about
checkpoint shape moves. The base run (#16) therefore launches with an open depth
dial instead of a baked 8-cap — the decision #86 existed to force.

## Limitations

Toy scale, one task (statetrack), 3 seeds, fixed-depth training (the production
trainer samples depth 1..8 per step; parity under random-depth sampling is
unverified, though nothing in the mechanism is depth-sampling-specific). d12 vs
d16 are within each other's noise — where the extended curve plateaus is unknown;
depth >16 unprobed. The extensibility read is one length-shift split (48). The
stability asymmetry (time-blind collapses 1/3 seeds; table and sinusoidal 0/3
here) is too few seeds to rank table vs sinusoidal on stability — and the table
arm has its own recorded collapse history (#77: a learned-table statetrack
control collapsed at seed 3), so no arm is immune.
