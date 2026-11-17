# The plain recipe on the SmolLM mix reaches 0.267 on LAMBADA at 7.5B tokens: 0.063 above run 001, 0.059 below the gate

Status: confirmed as registered (the gate verdict, one seed, 1.4 standard errors past the KILL line); why it still misses is not established
Date: 2026-11-17
Spec: experiments/base/specs/002-smol-mix-base.toml
Commit: 1ee0232 (the run)  Measured with: `python -m instruments.yardstick.eval_yardstick` on `runs/run_20261107_011942/checkpoints` step 3666623 (the supervisor's final score)

## Setup

`run_20261107_011942` (#597): the plain 8 × 960 transformer, Muon, WSD with the last 20%
decayed, 7.509B tokens (opt step 57,291), 227.4 h of training on the RTX 2060. The mix
was constant: FineWeb-Edu-Dedup 70%, Stack-Edu Python 20%, FineMath 3+ 10%. Against run
001 it changes the data, the token count (1.5×) and five recipe adoptions plus one
precision fix, all listed in the spec's hypothesis.

## Evidence

The full LAMBADA set gave **0.2668** accuracy and ppl **101.3**, against GPT-2-small's
0.3256 / 40.06. That is 0.059 below, past the 0.05 KILL margin by 0.009, so the verdict
is **KILL**. The registered prediction was 0.26 (range 0.22–0.31).

The verdict sits close to its line. The binomial standard error of an accuracy near 0.27
on 5,153 examples is 0.0062, and the #489 arms (the same recipe, three seeds each) spread
0.002–0.007. One seed, 0.009 past the line: a second seed could land in the band between
KILL and KEEP. The spec registers absolute margins, so the verdict stands as registered.

| | LAMBADA acc | LAMBADA ppl | open-web FineWeb CE |
|---|---|---|---|
| run 001, final (5.0B tokens) | 0.2036 | 301 | 3.906 |
| #489 general-web arm (001 branched and decayed) | 0.2379 | — | 3.692 |
| **run 002, final (7.5B tokens)** | **0.2668** | **101.3** | **3.521** |
| GPT-2 124M | 0.3256 | 40.06 | 3.427 |

The open-web column is the speedrun validation shard's first 2^18 targets at a 512-token
window (`instruments/yardstick/fineweb_val.py`). The 001 entry describes the same rows; 001's
own yardstick files hold no open-web record, so that match rests on its text.

Other observations:
- The LAMBADA subsample (1,000 examples) over the milestones read 0.097 → 0.134 → 0.146 →
  0.170 → 0.177 → 0.191, the last at 4.1B tokens and full learning rate. The final score
  is 0.076 above that last milestone; the decay began at 6.0B tokens.
- Held-out web loss (the corpus tail, #596) went from about 3.35 at the decay's start to
  3.10 at the end. It is not comparable with 001's 2.873, which read another slice of
  another corpus.
- That curve also fell in a step: 0.056 between two consecutive probes (3.229 at opt step
  52,160, 3.173 at 52,225), 0.075 by the next, and it stayed down. The code and math
  probes did not move, and no restart sits there. Training loss is too noisy to say.
  Single-probe drops of 0.05–0.08 also sit at 14,850, 28,480, 36,290 and 38,020.
  Unexplained.

## What it means

Run 002 is the best base model here and it still fails the gate by the spec's margin, so
by the base-model bar the next place to look is still data, learning rate, tokenizer or
eval, before any architecture verdict is read off this base.

001 → 002 is a comparison of two differently-built runs: it shows the size of the move
(+0.063 accuracy, a third of the perplexity, the open-web gap to GPT-2 down from 0.48 to
0.09 nats) and names none of the changes as its cause.

What it does not settle: how much of the remaining 0.059 is tokens and how much is the
kind of text. The model is within 0.09 nats of GPT-2 on the open web and still 0.059
behind on LAMBADA, which is narrative fiction, and none of the three buckets is a fiction
corpus (their fiction share was not measured). The open-web number flatters, though:
FineWeb is the family 70% of this mix was filtered from, GPT-2 trained on other text, and
no overlap reading between the shard and `runs/data-dedup` was taken
(`fineweb_val.overlap` exists). So the contrast hints at kind of text and proves nothing.

This line's prediction was right for the first time (0.26 registered, 0.267 scored),
after 001's two were too optimistic.

## Relation to prior work

SmolLM2-135M reaches 0.4289 on a mix of this kind at roughly 270× the tokens (the spec's figure). What
this entry records is where the same kind of mix lands at 7.5B tokens on this recipe.
