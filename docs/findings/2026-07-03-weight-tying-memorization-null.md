# Weight-tying shows no measurable memorization penalty and no generalization edge over distinct blocks at toy scale — both pre-registered criteria came back null

Status: negative/inconclusive (pre-registered criteria not met; no claim kept)
Date: 2026-07-03
Commit: 0970e6bde25269b73040b9331a196bb4c6b35eba  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python ablation_harness.py --task memorize --mem-pairs {8192,16384,32768} --dim 16 --arch {refiner,vanilla} --depths 4 --steps 600 --seed {0,1,2}` and `--task statetrack --arch {refiner,vanilla} --depths 4 --steps 2500 --seed {0,1,2}` (CPU, FORCE_F32_COMPUTE=1)

## Setup

The #35 hypothesis: at matched compute, the weight-tied CausalRefiner memorizes
less (fewer parameters available for lookup storage) and generalizes more than a
K-distinct-block vanilla transformer. Protocol pre-registered on the issue
before any run: matched pairs (same seed ⇒ same pools), depth 4, 3 seeds,
toy-local noise floor σ = per-arm seed spread, keep bars at ≥2σ_pooled. A pilot
located dim 16 as the width where memorize capacity binds (recall off ceiling
at N = 16384).

## Evidence

**Memorization** — recall over the trained dictionary, dim 16, mean ± σ over
seeds {0,1,2}:

| N | refiner (params) | vanilla (params) | vanilla − refiner | 2σ_pooled |
|---|---|---|---|---|
| 8192 | 0.996 ± 0.001 (0.14M) | 0.997 ± 0.002 (0.16M) | +0.001 (at ceiling) | — |
| 16384 | 0.880 ± 0.024 (0.28M) | 0.872 ± 0.024 (0.29M) | **−0.008** | 0.049 |
| 32768 | 0.531 ± 0.045 (0.54M) | 0.510 ± 0.053 (0.55M) | **−0.021** | 0.098 |

Criterion 1 (vanilla ahead by ≥2σ at some off-ceiling N): **not met** at any N —
the deltas sit well inside the noise floor, and their sign actually leans
refiner.

**Why the probe cannot show the hypothesized effect:** at the widths where
capacity binds, the token embedding dominates the parameter count (at N = 32768,
the 0.52M-param embedding is ~97% of both arms), and both arms share that
embedding + tied head. Storage for a key→value lookup lives in the embedding,
not the trunk, so trunk weight-tying is invisible to this instrument. The
memorize probe measures *embedding* capacity; separating trunk architectures
would need a task whose storage cannot fit in the embedding (e.g. pair-keyed
lookups over token combinations).

**Generalization** — statetrack held-out accuracy, dim 96:

| arm | params | acc (seeds 0/1/2) | mean ± σ |
|---|---|---|---|
| refiner | 0.36M | .982 / .935 / .965 | 0.961 ± 0.024 |
| vanilla | 0.67M | .927 / .945 / .970 | 0.947 ± 0.022 |

Criterion 2 (refiner ahead by ≥2σ): **not met** — Δ = +0.014 against a 2σ bar
of 0.046. What survives is parity-at-half-params: the shared block matches
distinct blocks with 54% of their parameter count.

**A live noise-floor lesson.** The single-seed #34 smoke showed refiner +5.6
accuracy points on this exact configuration (seed 0: .982 vs .927) and was
reported as a "clear" direction. Across three seeds the gap is +1.4 points —
inside the noise. This is rule 1 of the working agreement demonstrated on our
own numbers: the seed-0 comparison happened to pair the refiner's best seed
with vanilla's worst.

## Limitations

Toy scale, one depth (4), 3 seeds, capacity read at one width. A trunk-storage
probe (embedding-proof task), a depth/param-budget sweep, or LM-scale replay
could still surface the trade — but the claim as stated in #35 is not supported
here, and per the pre-registration no part of it is kept.
