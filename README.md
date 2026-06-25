# Tiny Refinement Model

This repository is built to be driven by the **Claude Code CLI**. I SSH into a single
machine — one **RTX 2060 (6 GB VRAM, Turing, no bfloat16)** — and Claude does the work
there.

Claude today is about as capable as a strong AI researcher — but not yet capable of
originating truly novel ideas; at its best it proposes the ideas a good researcher would.
That alone is enormously useful: a tireless collaborator that reasons carefully about a
problem, works programmatically, and pushes back when I forget something or argue against
what the literature has already settled. So the leverage is in the environment, not the
model. The point of this repo is to build the tools, guardrails, and infrastructure that
let Claude work autonomously, in a loop, around the clock — while I supply the ideas,
which are vetted and revised before they enter the pipeline.

## Why this exists

I didn't start this to rebuild an ordinary language model — that ground is well covered
and adds nothing new. The aim is a genuine contribution: to find out how much real
reasoning ability can be pressed into a very small model on hardware anyone can afford,
instead of buying capability with scale.

## The pipeline

Work runs as a pipeline. I come up with ideas; the ones that survive review become
**GitHub issues**, each a single closeable unit. From there an idea takes the same path
every time — a smoke test, then an ablation, then a full training run only if it earns
one — under the conventions below. The issues hold the live state of the project; what
gets learned is written down as findings; the narrative and the graveyard of discarded
ideas live in the roadmap. The repository itself is the environment that makes this loop
safe to run, and its layout comes next.

## Repository architecture & conventions

- **`CLAUDE.md`** — the working agreement Claude reads first: how we decide a result is
  real, the conventions here, and how to navigate the rest.
- **`config.py`** — the single source of truth for every architecture and training
  constant, the float16 compute policy, and the architecture selector.
- **Tracking is split by purpose.** GitHub issues hold per-item state. `docs/ROADMAP.md`
  holds the narrative and the graveyard of killed ideas. `docs/findings/` holds dated
  results, one conclusion each. `docs/registry/` holds model cards for kept models.
  Working plans stay local and gitignored.
- **Labels.** A *type* — `architecture` (the repo/environment), `tools` (research-support
  code), `ideas` (things to try on the model), `optimization` (cheaper code, same model),
  `documentation`. A *lane* — `cpu` (runs alongside a GPU job), `gpu` (the single card, a
  serial queue), `blocked` (unmet dependency). Plus `bug`.
- **Tests** live in `tests/` and run on CPU by default; CI runs them on every push and
  pull request.
- **`ablation_harness.py`** is the proof instrument — it trains the real architecture at
  tiny scale on toy tasks where depth has to do work.
- **Storage tiers.** The SSD holds live runs and the tokenized corpus under `runs/`; a
  1 TB HDD is the cold archive for finished runs and champion weights.

## The model

The model is the **`CausalRefiner`** (`plan_a_model.py`). Tokens are embedded and passed
through a stack of causal transformer blocks (RoPE positions, RMSNorm on the queries and
keys, a SwiGLU MLP, pre-norm residuals). A single **shared** block is then looped over
those representations several times — the *refinement depth* — each pass adding a
per-step time signal and blending its output into the running state through a gate. The
loop runs under a causal mask, so position *t* only ever attends to positions ≤ *t* and
depth refines a prediction without seeing future tokens. The number of refinement steps
is sampled randomly during training and fixed at inference. A tied LM head reads the
final state.

A second mode, selected with `MODEL_ARCH=reasoner` (`model.py`), is a vanilla
random-depth transformer kept as a control baseline.

Everything runs in float16 on the RTX 2060 (Turing has no bfloat16 tensor cores). The
tokenizer is `r50k_base`. Exact dimensions, depth limits, and the rest of the constants
live in `config.py`.
