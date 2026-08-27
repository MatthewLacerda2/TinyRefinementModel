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

## Loading an archived model

This is the section that makes the registry useful rather than decorative: picking up an old
model months later, to compare against a new one or to warm-start from.

### Archive layout

Every archive under `/mnt/d_drive/TRM_cold/weights/<name>/` looks like this:

```
<name>/
  <step>/              <- NUMERIC. this is the whole trick, see below
    model/  optimizer/  monitor_state/  step/  _CHECKPOINT_METADATA
  SHA256SUMS           <- per-file hashes; the card records sha256 OF THIS FILE
  run_metadata.json    metrics.csv      (+ whatever else the run produced)
```

**The step directory must keep its numeric name.** Orbax's `CheckpointManager` discovers
checkpoints by scanning for numerically-named subdirectories. An archive whose step dir is
called anything else — `checkpoint/`, `weights/`, `final/` — returns `all_steps() == []` and
looks empty and corrupt, with no error explaining why. The three July 2026 archives were
stored that way and were silently unloadable until relaid out on 2026-08-25.

### Load it

```bash
# 1. verify the archive is intact before trusting it
cd /mnt/d_drive/TRM_cold/weights/<name> && sha256sum -c SHA256SUMS

# 2. the archive path IS a checkpoint path — pass it straight in.
#    MODEL_ARCH must match the arch the card names: the two arches have
#    different param trees, so restoring a refiner into a reasoner skeleton
#    fails on the structure, not on anything informative.
MODEL_ARCH=refiner PYTHONPATH=. python -m instruments.dump_transcripts \
    --checkpoint-path /mnt/d_drive/TRM_cold/weights/<name> --device gpu
```

Any tool that takes `--checkpoint-path` works the same way, since they all resolve through
`trm/runtime/restore.py`.

### Treat the HDD as dumb blob storage

Per `CLAUDE.md`, don't train off the cold tier and don't rely on its permissions or symlinks
(it is a non-POSIX mount — note the `777` on everything). Reading a checkpoint to restore
from is fine. Writing a run there is not: **copy back to the SSD first.**

### Before archiving a new one

1. Copy the checkpoint dir keeping its numeric name.
2. Copy `run_metadata.json`, `metrics.csv`, and `worktree.patch` if the run was dirty — the
   patch is part of the recipe when the SHA alone doesn't reproduce the tree.
3. Generate the manifest, from inside the archive so paths stay relative:
   ```bash
   find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
   ```
4. Record `sha256sum SHA256SUMS` in the card, and confirm `CheckpointManager(<archive>)
   .all_steps()` is non-empty **before** you rely on it. An archive nobody has ever loaded is
   a backup nobody has ever tested.
