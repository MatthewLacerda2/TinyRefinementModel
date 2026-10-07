# A warmup-era gradient clip reaches Muon's matrices while it binds, and enlarges AdamW's steps ~1.6x for thousands of steps after it stops

Status: confirmed (optimizer replay at two model sizes; the effect on loss is not measured)
Date: 2026-09-21
Spec: experiments/recipe/specs/447-clip-under-muon.toml
Commit: b57ed9d (pre-registration)  Measured with: `python -m instruments.experiment experiments/recipe/specs/447-clip-under-muon.toml --no-gpu-lock`

## Setup

The shipped chain is `clip_by_global_norm(1.0)` → Muon on the 2-D matrices, AdamW
on everything else (in the plain model: the tied embedding/output head and the
norms). Both halves are invariant to a *uniform* rescale of the gradient, so the
clip cannot cap a step's size; it can only change how much each step weighs inside
the moment estimates.

`experiments/recipe/clip_under_muon.py` replays one gradient sequence into two
optimizer states, the clip in front of one and nothing in front of the other, and
compares the updates each emits at every step, per partition. Directions come from
a small plain model on real FineWeb-Edu tokens; every step's global norm is set to
the **base run's recorded pre-clip norm** (`run_20260920_191351`: ~10 → 3 in the
first 250 opt steps, 2.0 by 512, 0.56 after 1,000; the clip binds on every step
until ~512, on 28% to 1,000, never after). A null arm (clip 1e9) must read exactly 1.

## Evidence

Confirming run, dim 128 / 4 layers, seeds 0/1/2, median over each band:

| steps | clip on | Muon update cosine | AdamW update-norm ratio, clipped / unclipped |
|---|---|---|---|
| 0–100 | 100% | **0.898 / 0.877 / 0.872** | — |
| 100–250 | 100% | 0.967 / 0.972 / 0.969 | — |
| 512–1000 | 29% | 0.998 | 1.765 / 1.756 / 1.768 |
| 1000–1300 | 0% | 1.000 | **1.619 / 1.610 / 1.612** |

Null arm: exactly 1.0 at every point on every seed. Verdict KEEP — Muon cosine
Δ −0.118 at −12.3σ; AdamW ratio Δ +0.614 at +190.7σ.

The exploratory run at dim 64 (seen before the spec, so the prior rather than the
result) gave the same picture: Muon 0.847–0.885, AdamW 1.633–1.669. On seed 0 run to
4,000 steps, the AdamW ratio decays **1.62 → 1.43 → 1.25 → 1.11** over
1000–1500 / 1500–2000 / 2000–3000 / 3000–4000. A smooth-trend profile with no
step-to-step noise gives the same numbers: the effect follows the norm's trend.

**Mechanism.** Muon's momentum (β = 0.95) remembers ~20 steps: while the clip binds,
it reweights that window and turns the update, and within ~20 steps of the clip
letting go the two arms agree again. AdamW's second moment (β₂ = 0.999) remembers
~1,000: the unclipped arm's v is inflated by the norm-10 gradients of the first few
hundred steps, which depresses m/√v long after those gradients stopped. The clip
keeps v from ever seeing them, so the clipped arm steps larger — for about three to
four v-lifetimes.

**Prior on the predictor.** Before the exploration I predicted Muon cosine ≥ 0.995
and AdamW norms "a few percent" apart (#447). Both were wrong in the direction of
crediting the clip too little — the opposite of the depth-recurrence pattern.

## What it means

In the base run, "CLIP_NORM is inert after warmup" is true of the clip and false of
its consequences: it raises the effective step of the tied embedding and the norms
by ~1.6x at step 1,000 and ~1.1x at step 4,000 — about the first tenth of the run.
It is also not independent of Adam's β₂ (#359): at β₂ = 0.95, v remembers ~20 steps
and the tail disappears. A pair on either knob alone is confounded by the other.

A 512-step pair of `CLIP_NORM` values would live entirely in the binding era and
miss the tail; it does not inherit the shape of a base run.

## Limitations

Update geometry, not loss: this measures what the clip does to the steps, given the
base run's norm profile, at dim 64 and 128. Whether 1.6x larger embedding steps
early on help or hurt is a GPU pair. The norm profile between logged points (every
5 opt steps) is interpolated; the smooth control bounds that choice. The 4,000-step
tail holds the norm flat past the last logged step.
