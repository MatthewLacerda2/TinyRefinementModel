# Dense depth sweep corrects the d8 read: depth plateaus by ~d6, it does not keep climbing

Date: 2026-06-19. Model: the 77.65M `refiner` (CausalRefiner), same Stage 2
fine-tune protocol as `2026-06-18-plan-a-depth-transfer.md` (state tracking,
seq=48, n_states=8, n_gen=4, steps=3000, batch=32, lr=3e-4, seed=0,
`FORCE_F32_COMPUTE=1`). Instrument: `tools/eval_refiner_depth_finetune.py`.
Raw log: `runs/depth_sweep_gapfill_20260619_020601.log` (gitignored).

## Why this run exists
Stage 2 (`2026-06-18`) only measured depths 1, 2, 4, 8 — four dots. From those it
read two conclusions that turned out to be artifacts of the sparse sampling:
(a) "both arms are still climbing at d8, so depth may pay past 8," and
(b) "the pretrained arm climbs monotonically." This run fills depths 3, 5, 6, 7 for
both inits so the *shape* of the curve is measured, not interpolated.

## The dense 1→8 curve (new points in **bold**)

| depth | scratch acc (CE) | pretrained acc (CE) |
|------:|:-----------------|:--------------------|
| 1 | 0.5797 (1.0355) | 0.5620 (1.0996) |
| 2 | 0.6155 (0.9753) | 0.6512 (0.8897) |
| 3 | **0.6485 (0.9045)** | **0.6950 (0.7832)** |
| 4 | 0.6017 (1.1123) | 0.7301 (0.6864) |
| 5 | **0.7308 (0.6980)** | **0.8008 (0.5220)** |
| 6 | **0.8035 (0.5214)** | **0.7792 (0.5931)** |
| 7 | **0.8384 (0.4749)** | **0.8331 (0.4850)** |
| 8 | 0.7795 (0.5995) | 0.8048 (0.5089) |

## What the dense fill changes

- **Depth is NOT "still climbing at d8" — it plateaus by ~d6.** Both arms rise
  steeply through d5–d6 to ~0.80, then flatten. Both in fact **peak at d7**
  (scratch 0.838, pretrained 0.833) and *drop* at d8. The earlier "still climbing"
  read was a connect-the-dots illusion: d8 happened to sit above d4, but the
  intervening d5/d6/d7 are all higher than d8. There is no upward trend past the
  knee to chase. **This removes the motivation for a d>8 retrain.**

- **"Pretrained climbs monotonically" was also an artifact.** With the gaps filled,
  pretrained wobbles too (d5 0.801 → d6 0.779). Neither curve is monotonic at the
  top; the d6–d8 region is seed-level wobble around a shared ~0.80–0.84 ceiling.

- **What survives, sharpened: pretraining shifts the curve left, it does not raise
  the ceiling.** Pretrained leads clearly through d5 (d4: 0.730 vs 0.602; d5: 0.801
  vs 0.731) — more accuracy per loop — then scratch catches up at the d6–d7 plateau
  and the two converge. So general pretraining buys the **same ceiling with fewer
  loops**, consistent with (and stronger than) the 06-18 "exploits depth better"
  claim, now correctly framed as *earlier*, not *higher*.

- **The scratch d4 dip is real and is not the only wobble.** Scratch dips at d4
  (0.602, below its own d3 0.649) and again at d8 (0.780, below d7 0.838). The d4
  dip reproduces the 06-18 observation; the top-end wobble is new and argues the
  full-scale per-depth noise floor is wider than the tiny-ablation ±0.015.

## Caveats
- Single seed per (init, depth) cell, as in Stage 2. The *shape* claim (steep rise
  to ~d6, then flat) is robust because it holds across eight depths and both inits;
  the individual ±0.03 wobbles at the top are not individually trustworthy.
- Toy state-tracking task, not a downstream language task — the still-open
  downstream-language depth question (issue #32) is unchanged by this.

## Consequence for the roadmap
The "scale depth past 8?" question is now answered **no** on the evidence we have:
the curve is flat by d6 and d8 is already past the knee. Freed VRAM (bf16 opt-state
#18, r50k tokenizer #21) should fund **width and data (#22)**, not more refinement
depth. `MAX_STEPS_LIMIT=8` stays; no `max_depth` rebuild is warranted.
