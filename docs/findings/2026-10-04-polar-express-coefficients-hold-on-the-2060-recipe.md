# Polar Express's per-step Newton-Schulz coefficients beat the 2024 quintic by 0.023 val CE at 512 steps, at the same speed

Status: confirmed at 512 opt steps (3 seeds); adopted (`MUON_NS_COEFFS=polar_express`)
Date: 2026-10-04
Spec: experiments/recipe/specs/375-polar-express-pair.toml
Commit: 2827a76 (PR #538)  Measured with: `python -m instruments.experiment experiments/recipe/specs/375-polar-express-pair.toml`

## Setup

The shipped plain recipe (8 × 960, Muon, LR 6e-4, f32 residual stream, f16 compute), 512 opt
steps ≈ 67M tokens, seeds 0–2, one shared control. The only change is the coefficients of
Muon's 5 Newton-Schulz steps. The control uses Keller Jordan's 2024 quintic
(3.4445, −4.775, 2.0315) at every step. The arm uses Polar Express (Amsel et al., 2025), a
minimax-optimal quintic per step (`trm/train/polar_express.py`).

Not novel: the paper reports the same direction for GPT-2-scale Muon training. This entry
records that the result holds in our setting, which differs in f16 compute, the 2060's
batch 2, and an f32 stream. The audit asks every KEEP spec for a cited record; the Muon
pairs set that precedent.

## Evidence

Held-out val CE, mean of 3 seeds:

| opt step | control | Polar Express | Δ |
|---|---|---|---|
| 72 | 7.045 | 6.961 | −0.084 |
| 200 | 5.389 | 5.357 | −0.032 |
| 392 | 4.682 | 4.655 | −0.027 |
| 512 | **4.370** | **4.348** | **−0.023** |

Final per seed: control 4.368 / 4.378 / 4.365, Polar Express 4.347 / 4.356 / 4.340. The worst
arm seed beats the best control seed. From step ~230 the lead holds steady at −0.021 to
−0.027. Wall-clock for steps 64→512 was 109.4–109.9 min, against 110.1–110.4.

The registered bar, tokens to val CE 5.85, met KEEP on a single-probe gap (17.8M vs 18.9M
tokens, σ = 0 from probe quantization, #547). The full curve is what carries the verdict.

## Limitations

Peak per-block act_max rose (85–107 against 84–88). That is harmless with the f32 stream,
but it is worth watching on a base run's telemetry. The arm was not run together with
POST_NORM (#494), so whether the two gains add is unknown.
