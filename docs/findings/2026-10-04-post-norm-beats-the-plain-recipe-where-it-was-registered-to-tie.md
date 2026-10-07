# POST_NORM beats the shipped plain recipe by 0.056 val CE at 512 steps and halves its peak activation

Status: confirmed at 512 opt steps (3 seeds); verdict INCONCLUSIVE by the registered criteria, adoption open on #494
Date: 2026-10-04
Spec: experiments/recipe/specs/494-postnorm-pair.toml
Commit: 2827a76 (PR #538)  Measured with: `python -m instruments.experiment experiments/recipe/specs/494-postnorm-pair.toml`

## Setup

The shipped plain recipe (8 × 960, Muon with the 2024 quintic, LR 6e-4, f32 residual
stream), 512 opt steps ≈ 67M tokens, seeds 0–2, one shared control. The only change:
`POST_NORM=1`, an RMSNorm on each residual branch's output before it is added back
(Gemma 2's sandwich norm). It had been judged only on the refiner toy (#238), as a rescue
for depth recurrence.

## Evidence

Held-out val CE, mean of 3 seeds:

| opt step | control | POST_NORM | Δ |
|---|---|---|---|
| 8 | 10.715 | 8.807 | −1.909 |
| 72 | 7.045 | 6.295 | −0.751 |
| 200 | 5.389 | 5.235 | −0.154 |
| 328 | 4.887 | 4.797 | −0.090 |
| 456 | 4.469 | 4.411 | −0.058 |
| 512 | **4.370** | **4.314** | **−0.056** |

Final per seed: control 4.368 / 4.378 / 4.365, POST_NORM 4.316 / 4.312 / 4.315. The worst
POST_NORM seed beats the best control seed by 0.049. The lead shrinks quickly early on,
then flattens: −0.067 at 424, −0.056 from 488 to the cap.

Peak per-block act_max over the run was 35–42 against 84–88, and the final value was 31–34
against 57–87. The cost is 1.4% wall-clock (111.8 min for steps 64→512, against 110.1–110.4).

## What it means

The registered criteria asked only "does it cost tokens" (KILL) and "does it tie" (KEEP),
because the registered prediction was "parity or a small loss". A clear win was not a
branch, so the referee says INCONCLUSIVE. The numbers are not inconclusive: the arm is
ahead at every probe, by many σ.

That prediction was wrong in the opposite direction from #361's. It credited the mechanism
with too little.

Adoption stays open on #494, as the spec's readouts committed: "POST_NORM is NOT adopted
on this pair alone". The open question is the horizon. The gap was still −0.056 at 512 and
no longer shrinking, but a base run is ~40× longer. The cheapest settlement is a longer
pair, ~2,000 steps, the architecture tier, since it changes what the model is, judged
against the control's endpoint CE.

## Limitations

512 steps, one shape, Muon with the 2024 quintic. It was not run together with Polar
Express (adopted from the same PR), so the two gains are not known to add.

## Follow-up (2026-10-07)

The architecture-tier pair answered the horizon question: at 2,000 steps the lead is −0.015, not −0.056, so most of it was the start. See `2026-10-07-post-norms-lead-is-mostly-the-start-and-its-act-max-cap-grows-in-value.md`.
