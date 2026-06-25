# Model registry

A curated record of models worth keeping, so we can compare, fine-tune from, or
regenerate them later. The guiding idea (see `CLAUDE.md` → "The model registry &
reproducibility"):

**Weights are a cache; the recipe is the master copy.** If a run is deterministic —
same commit, config, seed, tokenizer, data manifest — then losing the weights costs
*compute to regenerate*, not *knowledge*. So the thing we protect in git is small: the
**model card**. The weights themselves are a regenerable convenience.

## Two tiers

- **Master (in git, tiny):** one model card per kept model — `MODEL_CARD_TEMPLATE.md`
  copied to `<run-id>.md` here. This is what makes the model reproducible and comparable.
- **Cache (gitignored, regenerable):** the weights/checkpoint, with a `sha256` recorded
  in the card so corruption is detectable. Live weights sit on the SSD under `runs/`;
  archive a champion's weights to the 1TB HDD once the run is done.

## What earns a card

Curate, or the registry fills with noise. Add a card only for:
- a **champion** — the current best on the agreed yardstick (match-GPT-2-small, then
  whatever supersedes it), or
- a **notable challenger** — a model whose result is worth being able to reproduce
  (a milestone, or a clean negative worth re-running).

Not "every run that finished."

## Comparisons: causal vs observational

A card lets you re-run and *compare* a stored model. Keep the two kinds of comparison
straight:
- **Matched one-variable pair** (same seed/data, one knob changed) → can claim a cause.
  This is a real ablation; it usually doesn't even need the registry, just two runs.
- **Cross-model comparison** (different params/data/arch) → observational only. A
  sanity/regression check — "are we in the ballpark" — never proof an idea works.

The registry mostly serves the second kind, plus warm-starting: a stored base model lets
a new idea **fine-tune or branch from it** instead of pretraining from scratch.
