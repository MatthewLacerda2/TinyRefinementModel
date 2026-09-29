# The first finished base model beats GPT-2 124M on its own text by 0.44 nats and misses the LAMBADA gate by 0.12

Status: confirmed (the gate verdict); the cause (the data mix) is a reading, under test in #489
Date: 2026-09-28
Spec: experiments/base/specs/001-plain-base.toml
Commit: 6ec8108 (the run)  Measured with: `python -m instruments.yardstick.eval_yardstick` on `runs/run_20260920_191351/checkpoints` step 2445311

## Setup

`run_20260920_191351`: the plain 8 × 960 transformer (136.9M parameters), Muon, WSD, 5.008B
tokens, 190.6 h on the RTX 2060. It is the project's first base model trained to its
budget. Its mix ramped to an end mix of FineWeb-Edu **score ≥ 4** 35%, codeparrot 40% and
finemath 25%. It held no general web and no narrative.

## Evidence

The gate on the full LAMBADA set gave **0.2036** accuracy and ppl 301, against GPT-2-small's
0.3256 / 40.06. That is 0.122 below, past the 0.05 KILL margin, so the verdict is **KILL**.

The same weights and GPT-2 124M, scored on the same rows with the same 512-token window and
the same targets (end-of-text excluded where our probe excludes it):

| text | ours | GPT-2 124M | gap |
|---|---|---|---|
| FineWeb-Edu held-out (64 rows, our val probe) | **2.873** | 3.310 | ours 0.44 nats better |
| FineWeb open web (speedrun val shard, first 2^18 targets) | 3.906 | **3.427** | ours 0.48 nats worse |
| LAMBADA (narrative fiction, last word) | 0.204 acc | **0.326** acc | ours 0.12 worse |

The model is ahead of GPT-2 on the text it trained on, and falls further behind the further
a text moves from that mix. On the open web, the gap was 0.91 nats at a mid-run reading and
0.48 at the end: the decay closed about half of it.

Other observations:
- The LAMBADA subsample climbed slowly over the milestones (0.078 → 0.133 → 0.143 → 0.173 → 0.178).
- Fixed-prompt transcripts stay on topic and write fluent didactic and historical prose. They
  drift toward exercises and math, and they loop on narrative prompts.

## What it means

The pipeline is not what the gate says is broken. A broken model does not beat GPT-2 by
0.44 nats anywhere. The miss reads as a **distribution gap**, which is the eval tool's own
wording for a healthy own-distribution ppl beside a failed LAMBADA. The base spec's KILL
branch lists data first among the places to look, and the mixture is the suspect it names.

That reading is not yet a result. Its matched test is the #489 pair: the same checkpoint,
decayed on the base run's end mix or with 25 points moved to general web, judged on
LAMBADA.

Two corrections to what was believed before the run:
- The registered LAMBADA prediction (0.29) was too optimistic by 0.09. This line's earlier
  recipe predictions had all been too timid.
- The codeparrot held-out probe is contaminated by near-duplicates in training (#485), so
  the model's code CE (0.81) is not a generalization number.

## Relation to prior work

It is widely known that models trained on narrow, filtered data do well in-domain and
poorly out of it. What this entry records is where this recipe lands: an educational ≥ 4
slice plus 65% code and math at 5B tokens puts a 137M model ahead of GPT-2 in-domain and
behind it on the open web and on LAMBADA. The numbers above measure that.
