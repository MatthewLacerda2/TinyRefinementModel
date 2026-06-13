# Compressed latent memory carried across windows fails to train: the cross-window "hunch" gives no next-window benefit and deeper reasoning slightly hurts

Status: confirmed
Date: 2026-06-13
Commit: 5f17340  Run: runs/run_20260611_234058 (checkpoint step 384639 ≈ opt step 3005)  Measured with: `PYTHONPATH=. venv/bin/python tools/eval_depth_curve.py --batches 16 --skip 3000000`

## Setup

Universal Reasoner, 79.6M params (dim 512, 4 encoder + 4 decoder blocks, one
weight-shared reasoning block iterated over 32 latent slots), trained on the
fineweb-edu curriculum mixture for ~3000 optimizer steps (~200M tokens) on a
single RTX 2060. Since the 2026-06-11 causality fix [[slot-future-leak]] the
reasoning loop's only pathway to influence predictions is the cross-window
hunch cache: the loop runs on window 1 and writes a compressed latent summary
that the decoder reads when it starts window 2. The depth curve measures exactly
this pathway — window-2 held-out CE as a function of window-1 reasoning depth,
with a no-hunch ("fresh") control. 16 held-out batches, evaluation data past the
trained range (skip 3,000,000 samples).

## Evidence

The curve is flat at the no-hunch baseline and, where it moves at all, moves the
wrong way:

| window-1 condition | window-2 CE (all) | hard-quartile CE |
|---|---|---|
| fresh (no hunch)   | 5.2360 | 9.7229 |
| depth 1            | 5.2347 | 9.7207 |
| depth 2            | 5.2339 | 9.7207 |
| depth 4            | 5.2350 | 9.7270 |
| depth 8            | 5.2403 | 9.7360 |

Having any hunch versus none buys 0.0013 nats — within measurement noise of
zero. The benefit does not grow with depth; it declines monotonically across all
eight depths, ending at a net loss of -0.0043 nats (all tokens) and -0.0131
(hard quartile) at depth 8 versus fresh. The monotone climb across eight points
is too consistent to be sampling noise: deeper reasoning produces a hunch that
mildly *degrades* the next window. (Plot: 2026-06-13-cross-window-hunch-depth-curve.png.)

The training telemetry corroborates the mechanism. The forget/blend gate
collapses to a degenerate near-constant regime almost immediately and never
recovers:

| opt step | forget_density | avg_forget_cost |
|---|---|---|
| 0    | 0.2011 | 0.8747 |
| 500  | 0.0055 | 0.0228 |
| 1000 | 0.0008 | 0.0037 |
| 1500 | 0.0003 | 0.0014 |
| 3000 | 0.0004 | 0.0018 |

By ~step 750 the dynamic blend-fresh-with-memory mechanism is effectively off.
The gradient eliminated the cross-window memory pathway early; subsequent
training did not revive it. This is despite the backbone training healthily over
the same window (train CE 9.83 -> 4.77, clean power law, validation tracking) —
the model learned to predict well by leaning on within-window attention and
ignoring the carried hunch.

Contrast with the pre-causality-fix reading, where extra reasoning steps
appeared to lower CE: that signal was the future-token leak [[slot-future-leak]],
not refinement. Once the leak is closed, the refinement benefit is gone — direct
evidence that the earlier apparent gains were leak bandwidth, not thinking.

## Limitations

Single run, single scale (79.6M), single architecture, 3000 optimizer steps
(~200M tokens) — not the full planned 8k. 16 eval batches; the per-condition CE
differences are ~0.001-0.005, so the all-tokens magnitudes sit near the noise
floor and the load-bearing evidence is the monotone trend plus the forget-gate
collapse rather than any single delta. What would strengthen it: more eval
batches; repeating at a second scale; checking whether the gate ever re-activates
under a longer run or a different gate parameterization. What would kill it: a
routing or incentive change (e.g. a loss weighting early-in-window tokens) that
makes the same hunch demonstrably predictive. The result speaks to *this*
cross-window-compressed-memory pathway; it does not test within-window causal
depth recurrence, which the current architecture cannot perform (the slot pooling
that would enable it is exactly what leaked).

## Relation to prior work

Not uncharted. Caching state across segments is Transformer-XL (Dai et al. 2019,
full hidden-state KV recurrence, not compressed). Summarizing a segment into a
few memory tokens passed to the next is the Recurrent Memory Transformer family
(Bulatov et al. 2022) — the closest analogue to our hunch, and reported as
finicky to train with task-dependent gains. Compressive Transformer (Rae et al.
2019) compresses old memory with similarly mixed results. The consensus is that
compressed cross-segment memory is hard to train because within-segment
prediction dominates the loss and starves the memory pathway of gradient — which
is precisely the forget-gate collapse seen here. The well-supported alternative
(Universal Transformers, Dehghani et al. 2018; the 2024-25 recurrent-depth /
latent-reasoning line) loops a shared block *causally over the current positions*,
feeding refinement back into the prediction it refines — the pathway this
architecture severed when it closed the leak, and the target of the Plan A
redesign.
