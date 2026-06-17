# Plan A depth recurrence earns its compute on state-tracking (#16 proof signal)

Date: 2026-06-16. Instrument: `ablation_harness.py` (tiny CausalRefiner, dim=96,
2 encoder layers, trained from scratch per depth on depth-sensitive toy tasks;
2500 steps, seed=0, gate_bias=0.0, lr=2e-3, train/test seq=24).

## Why this run exists
The from-scratch fineweb run showed Plan A (CausalRefiner) **tied** the
UniversalReasoner baseline on held-out CE at matched compute (~5.45 at opt-step
1850). That is the *expected* result, not a failure: recurrent-depth wins live on
tasks requiring multi-step inference, not on raw web-text perplexity. So the proof
gate (#16) was always going to be decided on a yardstick where depth provably has
to do work — which is what the ablation harness tests.

## Result

| task       | d1     | d2     | d4     | d8     | shape |
|------------|--------|--------|--------|--------|-------|
| parity     | 0.6817 | 0.6717 | 0.7169 | 0.6903 | flat (~0.70) |
| cumsum5    | 0.9833 | 0.9833 | 0.9908 | 0.9498 | ceiling at d1; d8 regresses |
| statetrack | 0.8236 | 0.8459 | 0.9236 | 0.9769 | **monotonic climb to near-solved** |

statetrack CE falls in lockstep with the accuracy climb: 0.413 → 0.367 → 0.252 →
0.066. Depth-1 → depth-8 is **+15.3 accuracy points** on the one task with both
headroom (not floored) and structure (not solved by a single block application).

## Interpretation
- **statetrack is the discriminating task and depth wins on it cleanly.** A single
  causal block already has full receptive field, so it can aggregate a running sum
  (cumsum5 solved at d1) — but state tracking requires *composing* updates across
  positions, and that composition is what the looped block buys. The monotonic
  staircase is depth recurrence doing real sequential work.
- **parity is uninformative, not a counter-example.** Parity (mod-2) is a documented
  hard case for attention models (Hahn 2020 and follow-ups): they struggle to learn
  it and do not length-generalize. The flat ~0.70 across all depths reflects that
  pathology, not a failure of depth.
- **cumsum5 has no headroom for depth to show** — depth-1 already scores 0.983.
- **The depth-8 collapse is real but task-dependent**: cumsum5 regresses at d8
  (0.991 → 0.950) while statetrack does not (climbs to 0.977). Deep recurrence can
  destabilize optimization in some regimes (cf. the `plan_a_model.py` gate comment
  and `--gate-bias` retention knob). Understand this before scaling depth.

## Seed reproduction (noise-floor check)
The statetrack staircase was re-run at seeds 1 and 2 (same config):

| seed | d1     | d2     | d4     | d8     |
|------|--------|--------|--------|--------|
| 0    | 0.8236 | 0.8459 | 0.9236 | 0.9769 |
| 1    | 0.7927 | 0.8704 | 0.9703 | 0.9523 |
| 2    | 0.8040 | 0.8800 | 0.9820 | 0.9741 |
| mean | 0.8068 | 0.8654 | 0.9586 | 0.9678 |

The depth-1 spread across seeds is ±0.015 (0.793–0.824) — that is the noise floor.
The mean depth-1 → depth-4 gain is +0.152, ~10x that spread. The climb reproduces
in all three seeds; the d4→d8 step is where seed noise lives (d8 peaks for seed 0,
dips mildly for seeds 1-2), so the useful depth on this task is ~4 with 8
occasionally softening — the same mild collapse cumsum5 showed more strongly.

## Caveat that remains
**Mechanism, not transfer.** This proves the looped block can perform sequential
computation depth-1 cannot, at tiny scale on toy tasks, robustly across seeds. It
does **not** establish that the 77.65M fineweb model converts that capacity into
downstream quality on real text — that is a separate, still-open question.

## Verdict
Plan A clears the #16 proof gate's intent: depth recurrence earns its compute on
the task that demands it, reproducibly across seeds and well above the noise floor.
Next: (1) **transfer** — does depth help the 77.65M model on a downstream task that
needs multi-step inference (not perplexity)?; (2) probe the depth-8 collapse /
`--gate-bias` interaction before committing to deep recurrence at scale.
