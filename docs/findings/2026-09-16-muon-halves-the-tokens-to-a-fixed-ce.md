# Muon halves the tokens to a fixed held-out CE at 148M in f16 — and costs less memory, not more

Status: open (KEEP, adoption gated on #287)
Date: 2026-09-16
Commit: 632c1a8
Runs: run_026_{adamw,muon_m100}_s{0,1,2} (six 512-step runs, ~3.3h each)
Spec: experiments/recipe/specs/026-muon-stage2-pair.toml
Measured with: `python -m instruments.experiment experiments/recipe/specs/026-muon-stage2-pair.toml`

## The claim

Muon — orthogonalized momentum (Newton-Schulz) on the 2-D weight matrices, AdamW on
the embedding, norms and biases — reaches held-out CE 5.85 in **half the tokens** of
the AdamW recipe every run of this repo has used, at dim 960 / 9 layers / 148M params,
in **f16 compute on a Turing card with no bf16**, and it does so while holding **less**
optimizer state than the AdamW it replaces.

| arm | tokens to CE 5.85 (M) | seeds | wall-clock to target | arena peak | val CE at the 512-step cap |
|---|---|---|---|---|---|
| AdamW (LR 1e-4) | **44.56** | 44.56 / 44.56 / 44.56 | 131.5 min | 4438 MiB | 5.707 / 5.700 / 5.709 |
| Muon (x100)     | **20.97** | 20.97 / 20.97 / 20.97 | 63.8 min | **4058 MiB** | 4.620 / 4.614 / 4.611 |

Matched pair, one variable: same seed ⇒ same init and same data order; only
`TRM_OPTIMIZER` (+ `MUON_LR_MULT`) differs. Δ = 23.59M tokens against a pre-registered
`min_delta` of 6.7M (15% of the control). Verdict: **KEEP**.

## What is novel here, and what is not

Muon's advantage over AdamW on token-efficiency is **settled** — the NanoGPT speedrun
line of work established it, and this result reproduces it rather than discovering it.
Three things around it are what this entry is for:

1. **It survives f16 on Turing.** The published Muon results run in bf16 on hardware
   this repo does not have. Here the matrix LR is 100x the shared schedule in pure f16
   with dynamic loss scaling, and it trains: no divergence, no NaN, 515 steps. The
   scaler does visibly work for it — the Muon arms take periodic `Non-finite loss/grad`
   skips near the end of the run (loss scale backing off 524288 → 131072 and climbing
   again) where the AdamW arms take none. Muon at x100 rides the edge of the f16
   exponent range; it does not fall off it, but a longer run should watch that margin.
2. **It is a memory *win*.** Muon keeps one momentum buffer for the 2-D matrices where
   AdamW keeps two moments, so peak arena fell 380 MiB (4438 → 4058) — about 8% of the
   4883 MiB budget on a 6GB card. On this box that is the difference between headroom
   classes, and it is the opposite of the "Newton-Schulz costs you something" intuition
   the stage-1 spec was drafted with.
3. **The wall-clock tax is 3%, not 9%.** Newton-Schulz was budgeted at ~9% per token at
   9 layers. Measured over 515 opt steps it is +3.0% (23.91s vs 23.21s per step), so the
   2.1x token win is a 2.06x wall-clock win and the KEEP survives its own readout.

## What this does NOT license

**The control's LR was never swept.** AdamW here runs at peak LR 1e-4, the value this
repo set once and never revisited, while the treatment's multiplier was tuned by the
stage-2a sweep (x25 36.0M, x50 27.5M, x100 21.0M, x200 diverged). Rule 4 says a win over
an undertuned baseline is a mirage. So this reads as *"Muon beats the recipe we have been
running"*, not *"Muon beats a tuned AdamW"* — #287 sweeps the control's peak LR, and the
base-run recipe should not adopt Muon until that number exists. A 2.1x gap is large
enough that a tuned AdamW is unlikely to close all of it; "unlikely" is not a measurement.

**The horizon is short.** 512 opt steps (67.1M tokens) is 0.3% of a base run. The
pre-registered caveat stands: a short horizon flatters the faster optimizer, and the gap
may narrow over a full budget. What the pair establishes is the early-training slope, on
a target (CE 5.85) that both arms clear comfortably.

**σ is undefined, and the margin is what decided it.** Tokens-to-target resolves no finer
than the validation probe interval (every 16 opt steps = 2.097M tokens), so all three
seeds of each arm landed in the same bucket and σ_pooled is exactly 0 — the referee
prints ±∞σ, which would decide a `beats` on any gap at all. The verdict rests on the
absolute margin instead (23.59M ≥ 6.7M), which `require = "all"` made mandatory, and the
gap is 11 probe intervals wide. The continuous readout agrees and is not quantized: val
CE at the cap separates 5.705 ± 0.005 from 4.615 ± 0.004. A σ bar on a metric quantized
this coarsely is decoration — drafted, not earned, and recorded here as a drafting flaw
rather than re-drafted after the fact. `instruments/audit.py` now says so out loud on
any spec in that shape.

## Relation to prior work

Follows [[2026-09-12-depth-recurrence-is-suppressed-not-exploited]] only in sequence, not
in substance: with the architecture bet retired, the live question is the recipe. Stage 1
of #26 (`026-muon-stage1-safety.toml`, INCONCLUSIVE) established the chain was safe to
spend card time on; this is stage 2. The registered prediction — "KEEP, by a wide margin"
— was **right**, the first registered prediction on this repo to be after four
consecutive wrong ones on depth recurrence, all wrong in the same direction.
