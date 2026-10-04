# Zero-initialising the attention output projection is 1.5 nats ahead at step 8 and 0.086 behind at step 512

Status: confirmed at 512 opt steps (3 seeds); not adopted
Date: 2026-10-04
Spec: experiments/recipe/specs/361-zero-init-o-pair.toml
Commit: 2827a76 (PR #538)  Measured with: `python -m instruments.experiment experiments/recipe/specs/361-zero-init-o-pair.toml`

## Setup

The shipped plain recipe (8 × 960, Muon with the 2024 quintic, LR 6e-4, f32 residual stream), 512 opt steps
≈ 67M tokens, seeds 0–2, one shared control. The only change: `ZERO_INIT_ATTN_OUT=1` zeroes
the attention output projection `o` at init, as the MLP's `down_proj` already is, so every
block starts as a no-op and the model at init is a tied-embedding bigram.

## Evidence

Held-out val CE, mean of 3 seeds:

| opt step | control | zero `o` | Δ |
|---|---|---|---|
| 8 | 10.715 | 9.208 | −1.507 |
| 72 | 7.045 | 6.082 | −0.964 |
| 136 | 5.860 | 5.600 | −0.260 |
| 264 | 5.110 | 5.103 | −0.006 |
| 392 | 4.682 | 4.773 | +0.092 |
| 512 | **4.370** | **4.456** | **+0.086** |

Final per seed: control 4.368 / 4.378 / 4.365, zero `o` 4.471 / 4.457 / 4.442. The worst
control seed beats the best treatment seed by 0.077. The gap opened at ~270 steps and stopped
growing after ~420, so it is not closing at the cap. The run used the same tok/s (110.4–111.7
min for steps 64→512, against 110.1–110.4). Per-block act_max started lower (~14 against
~27 at a quarter of the run), as predicted, and ended lower (51–63 against 57–87).

The pre-registered readout, tokens to val CE 5.85, said **KEEP** (13.6M against 18.9M
tokens). 5.85 is crossed at step ~130 of 512, inside the window where zero `o` is ahead, so
the bar measured only the start. The criterion was met, and the decision not to adopt goes
against it, on the readout `final_val_ce` that the spec carried with no weight.

## What it means

A no-op start helps an untrained network and costs a trained one, at least at this horizon.
Under Muon the first update of a zero matrix is already full-size, so the early lead is the
bigram start. The late cost is not explained here.

The registered prediction was "faster over the first ~200 steps, parity by 512". The first
half held. The second half was wrong in the same direction as the depth-recurrence
predictions (#238/#242/#246): it credited the mechanism with more than it delivered.

Method lesson: a tokens-to-target bar set at a CE the control reaches in its first quarter
judges the start of the run, not the run. Put the target near the control's endpoint.
The referee also turned the three identical, probe-quantized token counts into σ = 0 and
"+∞σ". That needs fixing in the instrument; see the PR.

## Limitations

512 steps is a recipe-knob horizon. A base run is ~40× longer, and the sign could change
again, though nothing here suggests it. One shape (8 × 960), one optimizer (Muon).
