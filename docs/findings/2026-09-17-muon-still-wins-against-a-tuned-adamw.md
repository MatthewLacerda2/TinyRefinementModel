# Muon's win survives a tuned baseline: 1.42x fewer tokens against AdamW at its own best LR

Status: open (KEEP; the base-run optimizer decision now rests on this, not on #26 stage 2)
Date: 2026-09-17
Commit: f59dd17 (trm/ identical to 57df54c, the commit the control arms trained at)
Runs: run_287_muon_m33.3333_lr0.0003_s{0,1,2} (three 512-step runs, ~3.5h each) against the reused run_287_adamw_lr0.0003_s{0,1,2}
Spec: experiments/recipe/specs/026-muon-stage3-tuned-control.toml
Measured with: `python -m instruments.experiment experiments/recipe/specs/026-muon-stage3-tuned-control.toml`

## The claim

Muon — orthogonalized momentum (Newton-Schulz) on the 2-D weight matrices, AdamW on the
embedding, norms and biases — reaches held-out CE 5.85 in **1.42x fewer tokens** than
AdamW at the peak LR that beat every other peak we have tried, at dim 960 / 9 layers /
148M params in f16 on a Turing card. Every parameter group is at its best-known LR on
both arms: Muon's matrices at the 1e-2 peak its own sweep chose, everything else at the
3e-4 peak #287 chose.

| arm | tokens to CE 5.85 (M) | step reached | s/opt step | wall-clock to target | arena peak | val CE at the cap |
|---|---|---|---|---|---|---|
| AdamW, peak 3e-4 | 26.21 / 27.53 / 27.53 | 200 / 210 / 210 | 22.9–23.2 | 76–79 min | 4,438 MiB | 5.093 / 5.107 / 5.099 |
| Muon, x33.3 on 3e-4 | **19.01 / 19.01 / 18.35** | 145 / 145 / 140 | 24.2 | **55–57 min** | **4,058 MiB** | **4.373 / 4.382 / 4.379** |

Matched pair, one variable: same seed ⇒ same init and same data order; only `TRM_OPTIMIZER`
(+ `MUON_LR_MULT`) differs. Δ = 8.30M tokens against a pre-registered `min_delta` of 4.0M,
+13.9σ. Verdict: **KEEP**.

## Why this run had to happen, and what it changes

[[2026-09-16-muon-halves-the-tokens-to-a-fixed-ce]] measured Muon against AdamW at peak
1e-4 and read 2.1x. [[2026-09-17-a-3x-peak-lr-beats-muons-win-over-the-same-control]] then
showed 1e-4 was undertuned by 2.25x, which put rule 4 — a win over a weak baseline is a
mirage — squarely against stage 2's KEEP. This run replaces the baseline.

**Both numbers are real and they mean different things.** 2.1x is Muon against the recipe
this repo had been running. 1.42x is Muon against a tuned AdamW, and it is the only one of
the two that licenses adopting Muon on the merits.

**The two earlier results could not simply be divided.** Stage 2 scored its probe on 4
held-out rows every 16 opt steps; #287 on 64 rows every 8. The same AdamW-1e-4 recipe read
44.6M tokens under the first and 60.3M under the second, so the yardstick itself moved by
35%. The registered prediction here came from rescaling stage 2's Muon reading by that
ratio (21.0 × 60.3/44.6 ≈ 28M, i.e. parity) and it was wrong by 9M tokens: **a probe
offset is not a constant along a training curve.** That is the reusable lesson — a changed
measurement cannot be corrected for after the fact, it has to be re-run.

## The cost side, since a token win has to survive it

- **Wall-clock +4.8% per step** (24.2 s vs 23.0 s), higher than stage 2's +3.0% because the
  control is now faster per step too. The token win still nets **1.39x less time** to the target.
- **Memory is a win, not a cost:** 4,058 vs 4,438 MiB peak. Muon holds one momentum buffer
  for the matrices where AdamW holds two moments — 380 MiB, ~8% of the 4,883 MiB arena on a
  6GB card.
- **f16 margin unchanged:** ~200 loss-scaler skips per run on both arms (routine upward
  probing, not divergence), streaks >1 at 2/4/2 vs 1/4/1, max activation 45 vs 42.

## What this does NOT license

- **The horizon is 512 opt steps**, 0.3% of a base run. A short horizon flatters the faster
  optimizer; what this measures is the early-training slope.
- **The baseline may move again.** #287 stage 2 (peak 6e-4) is running now. If 6e-4 beats
  3e-4, this control improves and the 1.42x narrows — the same trap this entry exists to
  close, one rung up. The base-run decision should wait for that number.
- **Seeds share the data stream** (#378): `DATA_SEED` moves a <1,025-token offset and the
  mixture draws, so σ here is init variance, not data variance.
- **Weight decay is LR-coupled** (#360) on both arms; the matrices' decay is unchanged from
  stage 2 since their LR is.

## Relation to prior work

Third leg of #26, after the stage-1 CPU safety gate (INCONCLUSIVE) and stage 2's KEEP.
Registered predictions on this repo now read: four on depth recurrence, all wrong by
crediting the mechanism too much; #287 stage 1, wrong by timidity; this one, wrong by
timidity again. The direction of the error has flipped, and the flip is worth watching.
