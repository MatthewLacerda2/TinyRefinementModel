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
- **`trm/config.py`** — the single source of truth for every architecture and training
  constant, the float16 compute policy, and the architecture selector.
- **Tracking is split by purpose.** GitHub issues hold per-item state. `docs/ROADMAP.md`
  holds the narrative and the graveyard of killed ideas. `docs/findings/` holds dated
  results, one conclusion each. `docs/registry/` holds model cards for kept models.
  Working plans stay local and gitignored.
- **Labels.** A *type* — `architecture` (the repo/environment), `tools` (research-support
  code), `ideas` (things to try on the model), `optimization` (cheaper code, same model),
  `documentation`. A *lane* — `cpu` (runs alongside a GPU job), `gpu` (the single card, a
  serial queue), `blocked` (unmet dependency). Plus `bug`.
- **Four trees, one home per file.** `trm/` is the library and its operations, and lives
  forever; `experiments/<line>/` holds one research line and is deleted with it;
  `instruments/` holds permanent measuring gear; `tests/` holds the guards. Nothing sits
  in the repo root, entry points run as `python -m trm.train.start`, and a test enforces
  both that and the dependency direction between the trees.
- **Tests** live in `tests/` and run on CPU by default; CI runs them on every push and
  pull request, alongside `ruff` and `vulture` (dead code).
- **Experiments are pre-registered.** Each one is a spec in `experiments/<line>/specs/*.toml`
  that fixes the arms, the seeds, and the keep/kill bars before anything runs;
  `python -m instruments.experiment <spec>` runs it and `instruments/verdict.py` referees it.
  The real-model workhorse is `experiments/recipe/tokens_to_ce.py`: the shipped config trained
  for a few hours per seed, scored by how many tokens it takes to reach a fixed held-out loss.
- **Storage tiers.** The SSD holds live runs and the tokenized corpus under `runs/`; a
  1 TB HDD is the cold archive for finished runs and champion weights.

## The model

The model is a **plain causal transformer**, `PlainTransformer` (`trm/model/plain.py`).
Tokens are embedded and passed through a stack of distinct causal transformer blocks
(RoPE positions, RMSNorm on the queries and keys, a SwiGLU MLP, pre-norm residuals), and
a tied LM head reads the final state. There is no loop and no depth dial: every token
gets the same amount of compute.

It replaced depth recurrence, which was the bet until September 2026. That design, the
**`CausalRefiner`** (`trm/model/refiner.py`), looped one shared block over the token
representations several times under a causal mask, with a per-step time signal and a
gate. It works on toy tasks that need sequential composition, but on language the trained
model learned to switch the loop off
(`docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md`). It stays
selectable with `MODEL_ARCH=refiner`, because the 4B-token champion checkpoint uses it.

A third mode, selected with `MODEL_ARCH=reasoner` (`trm/model/reasoner.py`), is a vanilla
random-depth transformer kept as a control baseline.

At the default shape it has 9 blocks of width 960 with 15 attention heads, a 512-token
window, and about 148M parameters.

Everything runs in float16 on the RTX 2060 (Turing has no bfloat16 tensor cores). The
tokenizer is `r50k_base`. Exact dimensions and the rest of the constants live in
`trm/config.py`.

## The training recipe

The defaults in `trm/config.py` train with **Muon at a 6e-4 peak** (its matrix partition at
1e-2). Until September 2026 they were AdamW at 1e-4, the value this project started with;
the recipe pairs of that month replaced it:
AdamW's best peak here is at least 6e-4, and **Muon** (orthogonalized momentum on the
weight matrices, AdamW on the embedding and norms) beats even that, reaching a fixed
held-out loss on 1.32x fewer tokens with 380 MiB less memory
(`docs/findings/2026-09-18-muon-holds-1-32x-over-adamw-at-its-best-lr.md`, which links
the chain of pairs behind it).
