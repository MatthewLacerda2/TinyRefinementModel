<!--
Copy this file to docs/registry/<run-id>.md for any champion or notable challenger.
Fill every field — the card is the master copy that regenerates and explains the model.
Delete this comment in the copy.
-->

# Model card — `<run-id>`

**One line:** <why this model is worth keeping — champion / milestone / notable negative>

## Recipe (what regenerates it)

| Field | Value |
|---|---|
| Commit SHA | `<git rev-parse HEAD at launch>` |
| `MODEL_ARCH` | `refiner` \| `reasoner` |
| Config snapshot | `<LATENT_DIM, NUM_HEADS, depth, MAX_SEQ_LEN, … or link to the config.py at that SHA>` |
| Params | `<count>` |
| Tokenizer | `r50k_base` (VOCAB_SIZE `<n>`) |
| Dataset manifest | `<prefill version + per-source token counts + filter settings, e.g. fineweb≥4 4.0B / codeparrot 4.0B / finemath 2.5B / ultrachat 0.26B>` |
| Seed(s) | `<DATA_SEED + model seed>` |
| Optimizer / schedule | `<AdamW, lr, warmup, decay, bf16-mu? …>` |

## Result (how it did)

| Metric | Value | Noise floor (seed σ) |
|---|---|---|
| Held-out perplexity | `<…>` | `<…>` |
| LAMBADA last-word acc | `<…>` | `<…>` |
| GPT-2-small yardstick | `<above / below / matches>` | — |
| <toy-task / depth-curve, if relevant> | `<…>` | `<…>` |

## Cost

| | |
|---|---|
| Peak VRAM | `<GB of 6>` |
| Wall-clock | `<h>` |
| Tokens seen | `<N>` |

## Weights (the regenerable cache)

| | |
|---|---|
| Checkpoint path (live) | `runs/<run-id>/checkpoints` |
| Archive path (HDD) | `<…>` |
| `sha256` | `<…>` |

## Notes

<what was notable, what to compare it against, links to the findings/issues it came from>
