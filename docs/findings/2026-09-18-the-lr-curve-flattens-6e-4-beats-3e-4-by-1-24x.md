# The LR curve flattens: 6e-4 beats 3e-4 by 1.24x, where 3e-4 beat 1e-4 by 2.25x

Status: open (KEEP; AdamW's peak for the base-run recipe is 6e-4, and #26 stage 3's control is stale)
Date: 2026-09-18
Commit: f59dd17 (trm/ identical to 57df54c, the commit the control arms trained at)
Runs: run_287_adamw_lr0.0006_s{0,1,2} (three 512-step runs, ~3.7h each) against the reused run_287_adamw_lr0.0003_s{0,1,2}
Spec: experiments/recipe/specs/287-stage2-6e4-pair.toml
Measured with: `python -m instruments.experiment experiments/recipe/specs/287-stage2-6e4-pair.toml`

## The claim

At dim 960 / 9 layers / 148M params in f16 on a Turing card, AdamW's peak LR is still
below its optimum at 3e-4. Peak 6e-4 — GPT-2-small's own value at a comparable size —
reaches held-out CE 5.85 in **1.24x fewer tokens**, on every seed, with no more work for
the f16 loss scaler.

| peak LR | tokens to CE 5.85 (M) | step reached | val CE at the cap | max activation | scaler skips |
|---|---|---|---|---|---|
| 1e-4 (the shipped value) | 60.29 / 60.29 / 60.95 | 460 / 460 / 465 | 5.835 / 5.838 / 5.843 | — | 202 / 200 / 201 |
| 3e-4 | 26.21 / 27.53 / 27.53 | 200 / 210 / 210 | 5.093 / 5.107 / 5.099 | 41 / 41 / 45 | 201 / 198 / 195 |
| 6e-4 | **20.97 / 22.28 / 22.28** | 160 / 170 / 170 | **4.723 / 4.761 / 4.751** | **68 / 74 / 77** | 194 / 199 / 200 |

Matched pair against the 3e-4 arms: same seed ⇒ same init and same data order, same
commit, same 64-row probe every 8 opt steps; only `PEAK_LR` differs. Δ = 5.24M tokens
against a pre-registered `min_delta` of 4.0M, +6.9σ. Verdict: **KEEP**. (The 1e-4 row is
from [[2026-09-17-a-3x-peak-lr-beats-muons-win-over-the-same-control]], the same
apparatus, and is shown for shape, not as this pair's control.)

## What is worth keeping here

1. **The curve is flattening, which is what tells us where we are.** 1e-4 → 3e-4 bought
   2.25x; 3e-4 → 6e-4 bought 1.24x for the same 2x step in LR. The optimum is at or above
   6e-4, and the remaining headroom is small. That shape — not either number alone — is
   what says the sweep is nearly done.
2. **What the higher LR costs, and what it does not.** The registered worry was the f16
   loss scaler: at 6e-4 it takes 194–200 skipped micro-steps per run, the same routine
   probing as 3e-4 (195–201), and max |logit| is flat at ~22. The global gradient clip
   bites *less* (63–66% of logged steps vs 76–80%). The one readout that moved is the
   **largest activation, which nearly doubled** (41–45 → 68–77). At this horizon that is
   still three orders of magnitude inside f16's 65,504 ceiling, but it is the quantity a
   1e-3 arm or a base-length run would be decided by (#368), and it is the reason this
   entry does not simply say "the LR is free".
3. **A verdict can go stale the same week it lands.** [[2026-09-17-muon-still-wins-against-a-tuned-adamw]]
   measured Muon against AdamW at 3e-4 and read 1.42x. Its control is now this arm.
   Against these numbers Muon's lead would be ~1.16x, but that is an unmatched
   cross-spec comparison and is **not** a result: Muon on a 6e-4 schedule
   (`MUON_LR_MULT` 16.667, matrix peak unchanged at 1e-2) has to be run against these
   arms. Three consecutive pairs have each moved the baseline of the one before it.

## Limitations

- **Horizon.** 512 opt steps (67.1M tokens) is 0.3% of a base run. A higher peak can win
  early and lose late; nothing here measures that.
- **Wall-clock is not comparable between the two sets.** The 6e-4 arms read 26.0–27.9 s
  per opt step against the control's 22.9–23.2, but the arithmetic is identical — these
  arms ran while finished checkpoints were copied to the HDD in the background. The token
  metric is unaffected. Arena peak is 4,438 MiB on every arm.
- **LR-coupled weight decay** (#360): the 6e-4 arm also got 2x the decay, a ~0.3% weight
  shrink over the run.
- **Seeds share the data stream** (#378), so σ is init variance, not data variance.
- **Prediction record.** Registered: INCONCLUSIVE leaning KEEP, with a real chance of
  KILL. Wrong, and timid — the third in a row on this line to underestimate the change,
  after four in the other direction on depth recurrence.
