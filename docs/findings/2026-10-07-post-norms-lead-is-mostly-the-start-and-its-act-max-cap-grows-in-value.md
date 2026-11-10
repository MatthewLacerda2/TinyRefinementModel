# POST_NORM's 512-step lead was mostly the start: at 2,000 steps it is −0.015 val CE, while the control's act_max grows past 600

Status: confirmed at 2,000 opt steps (3 seeds); INCONCLUSIVE by the registered bar; not adopted, the decision stays on #494
Date: 2026-10-07
Spec: experiments/recipe/specs/494-postnorm-long-pair.toml
Commit: a7f4830 (PR #551)  Measured with: `python -m instruments.experiment experiments/recipe/specs/494-postnorm-long-pair.toml`

## Setup

The shipped plain recipe: 8 × 960, Muon with Polar Express, LR 6e-4 on a schedule that
completes in-run, f32 residual stream, f16 compute, batch 2. The pair ran 2,000 opt steps
≈ 262M tokens, seeds 0–2. Its only change is `POST_NORM=1`, an RMSNorm on each residual
branch's output (Gemma 2's sandwich norm). This is the architecture-tier follow-up to the
512-step pair (`2026-10-04-post-norm-beats-the-plain-recipe-where-it-was-registered-to-tie.md`).

The registered target, 3.55, was an estimate the control never reached; the control ended
at 3.592. Using the spec's own fallback, it moved to 3.70 before any POST_NORM arm had a
result (commit a7f4830).

## Evidence

Held-out val CE, mean of 3 seeds:

| opt step | control | POST_NORM | Δ |
|---|---|---|---|
| 8 | 10.703 | 8.765 | −1.938 |
| 104 | 6.138 | 5.727 | −0.411 |
| 304 | 4.597 | 4.522 | −0.074 |
| 504 | 4.206 | 4.170 | −0.036 |
| 1000 | 3.940 | 3.915 | −0.024 |
| 1400 | 3.829 | 3.809 | −0.019 |
| 1760 | 3.698 | 3.681 | −0.017 |
| 2000 | **3.593** | **3.578** | **−0.015** |

- **Final per seed:** control 3.5924 / 3.5922 / 3.5943, POST_NORM 3.5776 / 3.5763 / 3.5787 (σ 0.0012 in each arm). The gap is ~13σ, and every POST_NORM seed beats every control seed.
- **Tokens to 3.70:** 230.7M for every control seed, against 226.5 / 226.5 / 225.4M for POST_NORM. That is Δ 4.5M tokens, 2% fewer, so the registered 11M bar (5%) is not met: **INCONCLUSIVE**.
- **Wall-clock:** steps 1600→2000 took 98.7–99.1 min for POST_NORM against 97.4–97.5 min for the control, +1.6%.

**Per-block act_max:**

| | peak | final |
|---|---|---|
| control | 653 / 601 / 798 | 601 / 350 / 505 |
| POST_NORM | 72 / 78 / 76 | 63 / 67 / 65 |

The control reads ~78–86 a quarter of the way in (step ~500), and its peak ends 7–9× above that. POST_NORM goes from ~30 to ~65 over the same span.

## What it means

**Horizon.** Most of the 512-step lead came from the start, and the 512-step schedule froze
it: in that pair the LR decays within 512 steps, so the control has no time to catch up.
At step 504 of the long pair, with the LR still high, the gap was −0.036 rather than
−0.056. From step ~1,400 to 2,000 it held at −0.015 to −0.019, which looks like a small
persistent gain rather than a gap still closing. Whether it holds at 5B tokens, ~19× longer
again, is not known.

**The second axis.** With the f32 residual stream (#357), act_max no longer threatens the
stream itself. It is still the input scale every pre-norm divides by, and the control's
kept growing to the cap (quarters: ~86, ~240, ~370–480, ~500–600) with no sign of flattening. A base run is ~38k steps. POST_NORM caps
that growth at ~70. Whether an uncapped act_max costs anything on a long run is #536's
question (read each branch's f16 output), not this pair's.

**Prediction check.** The registered prediction was "still ahead, by less than at 512,
−0.02 to −0.04 at the cap: KEEP". The direction was right but the size was overstated
(−0.015), and so was the verdict. The three init/normalization pairs (#361, #494 at 512, this
one) all misread the same thing, how much of an early lead survives the rest of the run.
#361 and this pair overstated what survives. The 512-step pair said "tie" and measured a
lead, but that lead was mostly the start.

## Limitations

- **Coverage:** one shape (8 × 960), one optimizer, 2,000 steps.
- **The act_max trend:** extrapolated from 2,000 steps, not measured further.
- **Polar Express:** both arms carry it, so its interaction with POST_NORM is inside both numbers, not separated.
