# Raising AdamW's peak LR 1e-4 → 3e-4 cuts the tokens to a fixed CE by 2.25x — more than Muon's win over the same control

Status: open (KEEP; reframes #26's KEEP, which was measured against the 1e-4 control)
Date: 2026-09-17
Commit: 57df54c
Runs: run_287_adamw_lr{0.0001,0.0003}_s{0,1,2} (six 512-step runs, ~3.3h each)
Spec: experiments/recipe/specs/287-peak-lr-pair.toml
Measured with: `python -m instruments.experiment experiments/recipe/specs/287-peak-lr-pair.toml`

## The claim

At the shipped config (plain, dim 960, 9 layers, 148M params, f16 compute with dynamic
loss scaling on an RTX 2060), AdamW with the whole schedule scaled to a 3e-4 peak reaches
held-out CE 5.85 in **2.25x fewer tokens** than the 1e-4 peak this repo has used since it
started, on every seed, and it does so without working the f16 loss scaler any harder.

| arm | tokens to CE 5.85 (M) | opt step reached | wall-clock to target | val CE at the 512-step cap | non-finite skips |
|---|---|---|---|---|---|
| peak 1e-4 | 60.29 / 60.29 / 60.95 | 460 / 460 / 465 | 177–178 min | 5.835 / 5.838 / 5.843 | 202 / 200 / 201 |
| peak 3e-4 | **26.21 / 27.53 / 27.53** | 200 / 210 / 210 | **76–79 min** | **5.093 / 5.107 / 5.099** | 201 / 198 / 195 |

Matched pair: same seed ⇒ same init and same data order; only `PEAK_LR` differs (init =
peak/10, end = peak/100, same warmup and cosine). Δ = 33.4M tokens against a 6.7M bar.
Arena peak 4438 MiB and ~197–199 min per run on every arm. Validation on the 64-row probe
every 8 opt steps (1.05M tokens).

## What is novel here, and what is not

That 1e-4 is low for a ~124–150M model is **settled** (GPT-2-small trains at 6e-4). What
this entry records is the size of it *on this apparatus*, and what that does to a result
already on the books:

1. **The LR fix alone is bigger than the Muon win.** #26 stage 2 read Muon at 2.1x fewer
   tokens than this same 1e-4 control; a 3x AdamW LR gives 2.25x. The two measurements used
   different probes (4 rows every 16 steps vs 64 rows every 8), so the ratios are not a
   head-to-head — but they are enough to say #26's KEEP was largely a win over an
   undertuned baseline (rule 4). Whether Muon beats a *tuned* AdamW is unmeasured, and needs
   its own matched pair.
2. **f16 did not charge for it.** The registered worry was that a 3x LR would thrash the
   dynamic loss scaler on a card with no bf16. It did not: ~200 skipped micro-steps per run
   on both arms, the scaler's routine upward probing, the same count #26's AdamW and Muon
   arms show. (The spec's readouts note said AdamW took no skips in #26; that was wrong.)

## Limitations

- **Short horizon.** 512 opt steps (67.1M tokens) is the early-training slope. A higher
  peak can win early and lose late; the gap on a full base-run budget is not measured.
- **The control was near the cap.** 1e-4 reached 5.85 only at steps 460–465 of 512.
- **Quantized metric.** Tokens-to-target resolves to 1.05M tokens; #351 adds a +0–4 step row
  offset to both arms alike. σ ≈ 0.6M, the margin is 33.4M. The continuous readout agrees
  (val CE at the cap 5.84 vs 5.10).
- **LR-coupled weight decay** (#360): the 3e-4 arm also got 3x the decay, a 0.15% vs 0.05%
  weight shrink over the run. Negligible, and standard AdamW behaviour.
- **Only 3e-4 was tried.** The optimum may be higher; 6e-4 is now eligible for its own spec.

## Relation to prior work

Reframes [[2026-09-16-muon-halves-the-tokens-to-a-fixed-ce]] (#26 stage 2, PR #308),
whose "What this does NOT license" section named exactly this risk. The registered
prediction here (KEEP at ~1.5x, real weight on INCONCLUSIVE) was **wrong in the timid
direction** — the opposite of the four depth-recurrence predictions, which all credited
the mechanism too much.
