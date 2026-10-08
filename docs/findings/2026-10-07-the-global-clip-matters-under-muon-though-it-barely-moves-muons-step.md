# Removing the global gradient clip costs 0.046 val CE under Muon, though the clip barely changes Muon's step size

Status: confirmed at 512 opt steps (3 seeds); INCONCLUSIVE by the registered tokens bar; CLIP_NORM stays 1.0, now chosen rather than inherited
Date: 2026-10-07
Spec: experiments/recipe/specs/447-clip-off-pair.toml
Commit: see PR #560  Measured with: `python -m instruments.experiment experiments/recipe/specs/447-clip-off-pair.toml`

## Setup

Today's plain recipe: 8 × 960, Muon with Polar Express, LR 6e-4 on a schedule completing
in-run, f32 stream, z-loss 1e-4, Adam b2 0.95, batch 2. The pair ran 512 opt steps ≈ 67M
tokens, seeds 0–2. The only change: `CLIP_NORM=1e9` (never binds) against the default 1.0,
i.e. `optax.clip_by_global_norm` before the Muon/AdamW partition is effectively removed.
The per-micro-step guard clip (#201) is unchanged.

#447's CPU probe (2026-09-20) had shown that under Muon this clip leaves the matrix
partition's step size within 0.6% even at a 10,000× spike. Newton-Schulz drives singular
values to ~1. The clip caps the AdamW partition at ~0.79 and still rescales the gradient
that enters Muon's momentum.

## Evidence

Held-out val CE, mean of 3 seeds:

| opt step | clip 1.0 | no clip | Δ |
|---|---|---|---|
| 40 | 7.558 | 7.578 | +0.020 |
| 104 | 6.231 | 6.392 | +0.161 |
| 200 | 5.357 | 5.434 | +0.078 |
| 304 | 4.929 | 4.995 | +0.066 |
| 448 | 4.471 | 4.521 | +0.050 |
| 512 | **4.347** | **4.393** | **+0.046** |

- **Final per seed:** clip 4.3481 / 4.3527 / 4.3401, no clip 4.3920 / 4.4000 / 4.3877. Every clipped seed beats every unclipped one.
- **The registered bar:** tokens to 4.50 were ~57.7–58.7M clipped against ~59.8M unclipped, Δ 2.1M (3.5σ). That is under the 2.9M (5%) bar, so the verdict is **INCONCLUSIVE**. The final CE reading has the same sign.
- **clip_active:** 46–49% of logged steps with the clip on (it was 70–85% on the older 9 × 960 AdamW-era recipe), and 0 without it.
- **Activations without the clip:** branch_max (#536) peaks 41–53 against 37–41 with the clip, and act_max 103–110 against 93–103.

## What it means

The clip is not inert under Muon. It doesn't set Muon's step size, but removing it costs
~0.05 nats by step 512, ahead at every probe after step 40. That fits the probe's narrow
statement: the clip shapes Muon's *direction* through its momentum, and it caps the
embedding's AdamW updates. Which of the two carries the effect is not separated here.

The registered prediction was "parity: curves on top of each other". It was wrong: the
clip earns its place. CLIP_NORM 1.0 stays, now a measured choice. Whether a different
value does better (0.5, 2.0) is a sweep this pair did not run.

## Limitations

512 steps (the recipe tier). One value against none. The two mechanisms, Muon momentum
direction and the AdamW cap, are not separated.
