---
name: instruments
description: Which of this repo's instruments answers which question about the model, a run, the card or the corpus, with the command to run it; and the contract a new instrument must meet. Use before measuring speed, memory, quality or a run's health, before reading a checkpoint, or before writing a new instrument.
---

# The instruments

Every instrument's module docstring is its manual (flags, caveats, what it refuses).
This file is the index: find the question, then read that docstring. Run them as
modules from the repo root with `venv/bin/python -m instruments.<name>`.

## Which one

**Is this result real?**
| question | instrument |
|---|---|
| verdict of a pre-registered spec | `verdict` (library, called by `experiment`), see the `experiment` skill |
| can that verdict be trusted (floor, seeds, σ, commit order) | `audit` — `make audit`, `make audit ALL=1` |
| what a pair did beside its verdict | `pair_report <spec> [--runs DIR]` |
| how noisy the validation probe itself is | `probe_sigma --checkpoint-path …` |
| the base run's verdict, yardstick journal and model card | `base_run score / verdict / card` |
| has the base run stalled | `stall --spec … --run runs/<run>` |

**How good is a checkpoint?**
| question | instrument |
|---|---|
| LAMBADA vs GPT-2-small (the gate) and SmolLM2 (the neighbour) | `yardstick.eval_yardstick [--checkpoint-path …]` |
| score any HF model on the same yardstick | `yardstick.score_hf --model …`; calibration: `yardstick.calibrate_gpt2` |
| can it write Python (pass@k per difficulty) | `python_world --checkpoint-path … ` / `--reference` |
| what it actually says on fixed prompts | `dump_transcripts` (card must be free for `--device gpu`) |
| what mean held-out CE averages away | `heldout_profile --checkpoint-path … --step N` |
| every offline diagnostic for one checkpoint | `milestone_report [--ckpt DIR] [--quick]` |

**How is a run doing?**
| question | instrument |
|---|---|
| weight, cost and health in the terminal, plus figures | `make report RUN=run_…` (`report` + `plots`) |
| the CE curve and where it crosses a target | `curves` (library); metrics.csv is read through `runlog` |
| values with a fixed correct range (integrity checks) | `invariants` (library) |

**Speed and memory** (VRAM comes from the allocator's `memory_stats()`, never nvidia-smi)
| question | instrument |
|---|---|
| throughput of the real loop | `profile_step --pace`; without `--pace`, a trace of where the time goes |
| where a grad step's VRAM goes | `mem_profile [--top 15]` |
| does a config fit the card, with what headroom | `make gate` (`vram_headroom_smoke [--layers N]`) |
| does the real f16 path train / overfit one batch | `smoke_gpu`, `overfit_smoke` |
| parameter count without building the model | `model_stats` (library; `report --model-only`) |

**Inside the model**: `latents` (the residual-stream trajectory, block by block),
`trajectory_figures` (pictures of it), `embedding_geometry` (how the embedding organizes).

**Data**: `corpus_overlap heads|probe <bucket>` (where 32-token windows repeat).

**Reproduce the past**: `timemachine list | reconstruct <run-id> | fork <run-id> <name>`
revives a stored weight in the exact commit and environment that trained it.

**Repo and CI**: `queue [--cloud]` (what to work on next), `make ready PR=N` /
`make ci-wait PR=N` (`ci_wait`), `changed` (what a diff touches, used by
`make test-affected` and `make audit`).

## Writing a new one

- It lives in `instruments/` for good and never imports `experiments/` (a test enforces this).
- It prints a **plain, reproducible report**: what it measured, the result, how fast,
  and what it cost in CPU, GPU, RAM and VRAM. Numbers someone can compare across runs,
  never prose about them.
- A harness the runner reads prints `RESULT {...}` lines (`results.py`). Read metrics
  through `runlog`, shared helpers come from `_common.py`, and specs are parsed with `spec_file`.
- The docstring opens with the question it answers and a usage line.
- Its wiring tests go in `tests/apparatus/` if it serves one research line, in
  `tests/core/` if it is permanent (`tests/README.md`).
