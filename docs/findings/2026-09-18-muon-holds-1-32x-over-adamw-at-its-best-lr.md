# Muon holds 1.32x over AdamW at its best LR: the lead narrowed as the baseline was tuned, and stopped narrowing

Status: open (KEEP; the base run's optimizer is Muon at x16.667 on a 6e-4 schedule, per the rule registered before the numbers)
Date: 2026-09-18
Commit: f59dd17 (trm/ identical to 57df54c, the commit the control arms trained at)
Runs: run_287_muon_m16.6667_lr0.0006_s{0,1,2} (three 512-step runs, ~3.5h each) against the reused run_287_adamw_lr0.0006_s{0,1,2}
Spec: experiments/recipe/specs/382-muon-on-6e4-pair.toml
Measured with: `python -m instruments.experiment experiments/recipe/specs/382-muon-on-6e4-pair.toml`

## The claim

At dim 960 / 9 layers / 148M params in f16 on a Turing card, Muon reaches held-out CE
5.85 in **1.32x fewer tokens** than AdamW at the best peak LR we have found for it
(6e-4), with every parameter group of both arms at its best-known LR.

| arm | tokens to CE 5.85 (M) | step reached | s/opt step | arena peak | max activation | val CE at the cap |
|---|---|---|---|---|---|---|
| AdamW, peak 6e-4 | 20.97 / 22.28 / 22.28 | 160 / 170 / 170 | 22.9–23.2 (clean, stage 3) | 4,438 MiB | 68 / 74 / 77 | 4.723 / 4.761 / 4.751 |
| Muon, x16.667 on 6e-4 | **17.04 / 15.73 / 17.04** | 130 / 120 / 130 | 24.1–24.3 | **4,058 MiB** | **59 / 47 / 59** | **4.195 / 4.200 / 4.195** |

Matched pair: same seed ⇒ same init and same data order, same commit, same 64-row probe
every 8 opt steps, same 6e-4 schedule; only `TRM_OPTIMIZER` (+ `MUON_LR_MULT`) differs.
Δ = 5.24M tokens against a pre-registered `min_delta` of 3.3M, +6.9σ. Verdict: **KEEP**.

## The chain this closes

The same question was asked four times because the baseline kept moving, and every
step was a matched pair of its own:

| pair | Muon | AdamW | Muon's lead |
|---|---|---|---|
| [[2026-09-16-muon-halves-the-tokens-to-a-fixed-ce]] | x100 on 1e-4 | 1e-4 | 2.1x (4-row probe) |
| [[2026-09-17-muon-still-wins-against-a-tuned-adamw]] | x33.3 on 3e-4 | 3e-4 | 1.42x |
| this entry | x16.667 on 6e-4 | 6e-4 | **1.32x** |

with [[2026-09-17-a-3x-peak-lr-beats-muons-win-over-the-same-control]] and
[[2026-09-18-the-lr-curve-flattens-6e-4-beats-3e-4-by-1-24x]] moving AdamW in between.
Tuning the baseline took Muon's lead from 2.1x to 1.42x, then only to 1.32x, as AdamW's
own LR curve flattened. The lead that survives a tuned baseline is about a third.

**Muon gained from the same change AdamW did.** Muon on the 6e-4 schedule (16.60M mean)
beat Muon on the 3e-4 schedule (18.79M), with the matrix peak held at 1e-2 in both: the
higher LR on its AdamW partition (embedding, norms, biases) helped it too. That is why the
registered prediction, cross-table arithmetic that held Muon fixed at its 3e-4 reading,
came in low at 1.15–1.20x.

## The cost side

- **Wall-clock:** +4.8% per opt step for Newton-Schulz (24.2 s against AdamW's clean
  22.9–23.2 s on the same code; the control's own log reads 26–28 s because HDD archiving
  ran beside it). The token win nets ~1.25–1.3x less time to the target.
- **Memory:** 380 MiB less arena peak, measured the same in all four pairs. Muon keeps one
  momentum buffer for the matrices where AdamW keeps two moments.
- **f16 margin:** scaler skips and max |logit| are indistinguishable between arms. At 6e-4
  Muon runs *cooler* activations than AdamW (47–59 vs 68–77), reversing the 3e-4 reading,
  where Muon ran slightly hotter (45 vs 41).

## What this does NOT license

- **Horizon.** 512 opt steps (67.1M tokens), 0.3% of a base run. A short horizon flatters
  the faster optimizer, and the chain above shows the lead is sensitive to everything
  around it.
- **Muon's multiplier was never re-swept.** The x100 on 1e-4 sweep set the matrix peak at
  1e-2, and every later pair held that fixed. AdamW got two more LR points; Muon's matrix
  LR got none.
- **These pairs trained on the start of the mixture ramp** (#362): ~web-only data, while
  the base run ends at 65% code and math.
- **Seeds share the data stream** (#378), so σ is init variance, not data variance.
  **Weight decay is LR-coupled** (#360).
- **Prediction record:** KEEP at 1.15–1.20x registered; 1.32x measured. The fourth
  consecutive miss on this line in the timid direction, after four in the credulous
  direction on depth recurrence.
