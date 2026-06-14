---
name: Ablation / experiment proposal
about: Propose an experiment with hypothesis, metric, and success/fail criteria agreed before any compute is spent.
title: "[ablation] "
labels: ["experiment"]
---

## Hypothesis
<!-- One sentence. What do you expect to change and why? Cite prior art / docs if this is documented ground. -->

## What is being varied
<!-- The single knob under test (arch change, schedule, data mix, ...). Everything else held fixed. -->

## Task & dataset
<!-- Which task/data slice, which split, sequence length, etc. -->

## Metric
<!-- The ONE primary metric this is judged on (e.g. val loss, eval accuracy). Secondary metrics optional. -->

## Success / fail criteria (agreed in advance)
<!-- Be concrete and decide BEFORE the run. -->
- Keep if:
- Reject if:

## Baseline
<!-- What is this compared against? Link the run / commit / finding that defines the baseline. -->

## Compute estimate
<!-- Approx wall-clock on the RTX 2060, VRAM headroom, checkpoint size. Flag anything multi-hour for sign-off. -->
- Est. wall-clock:
- Peak VRAM:
- Needs sign-off before launch? (yes / no)

## Risk / known dead-ends
<!-- Documented to fail for general models, or genuinely untried? Note what could invalidate the result. -->
