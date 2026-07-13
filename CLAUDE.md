# Working agreement & research doctrine

This file is the whole context: how we work together, how we know a result is real,
how the repo is laid out, and how to dig into what we already learned. It is meant to
be small enough to read in one sitting — if it ever stops fitting in your head, that's
the signal to split a piece of it into a skill, not to let it sprawl.

## How we work

Prefer plain language that explains what we — or the code — are *doing*, not highly
technical decoration. The user is trying to architect intelligence, not ornament an
implementation.

First structure a good architecture and write code that is readable and organized.
Then tighten it — denser, more compact — but compactness serves readability, it is not
the finish line. If the clearest version of something isn't the densest, leave it
clear. Don't end on clever one-liners nobody can debug later.

Push back when it's earned:
- If a feature or addition doesn't move the model's final performance, say so and say
  why it isn't pulling its weight.
- If an idea contradicts what the literature has settled, flag it immediately. But
  *calibrate*: push hard on **documented dead-ends**, stay curious about **genuinely
  untried** ground. Research means trying what the literature hasn't settled — don't
  suppress a novel idea just because it's unproven. The line is "documented to fail"
  versus "simply not yet tried."

Align before building. The user must have a clear, defined idea of what he's trying to
say. If the idea isn't yet clear — to him or to you — **stop**: don't plan, don't
implement. Get the idea defined for both of you first. Alignment of understanding comes
before everything downstream.

Favor smoke tests and ablations, preferably tiny so we test fast and know exactly what
works and what doesn't. Knowing — pinning down what holds and what breaks — beats a
speculative model improvement every time.

## How we know something worked

The user's instinct is to experiment a lot: try every idea several ways, pull pieces
out to see what happens. That instinct is right — and it is exactly the thing that
manufactures false wins if it isn't disciplined. Trying one idea ten ways hands you a
"winner" by chance alone. So the cure is never *fewer* experiments; it's **cheap,
controlled, pre-committed** experiments. These five rules are how a result earns belief:

1. **Clear the noise floor.** A single-seed number is not a result. We measure
   seed-to-seed variance first (what "no real difference" looks like), and a delta only
   counts if it clears that floor. Report the spread, not just the point estimate. Our
   own past deltas have sat *inside* the noise (0.001–0.005 nats) — that is what an
   unguarded comparison looks like.

2. **One variable per experiment, matched pairs.** An ablation only attributes cause if
   exactly one thing changes and everything else is held fixed — **same seed, same data
   order**, same config but the one knob. Pull a piece out, keep the rest identical,
   compare. An unmatched comparison tells you nothing; a matched one costs more (you
   train the control too) and is worth every minute.

3. **Pre-register the kill criterion.** Before the run, write down the threshold that
   would make you *keep* or *drop* the idea — phrased against the noise floor (e.g.
   "must beat its matched control by ≥2σ on the toy task, or it's dead"). You're not
   predicting the result; you're tying your own hands so enthusiasm can't move the
   goalpost after you see the number.

4. **Earn the comparison with a strong baseline.** A win over a weak or undertuned
   baseline is a mirage. Before any architecture bet is judged, the vanilla control it's
   measured against must itself be trained properly. (See the base-model bar below.)

5. **Record every result — where it belongs.** Every experiment's outcome gets written
   down, win or lose; novelty decides the home:
   - **Not novel** (literature settles it; we confirmed it holds here): the PR is the
     record — what ran, the numbers, the verdict. No findings file. If it killed or
     gates something, one tombstone line in the ROADMAP graveyard linking the PR —
     future sessions grep the repo, not closed PRs.
   - **Novel** (a combination, technique, or result the literature doesn't cover —
     failures included; a novel negative is exactly the moat): a dated `docs/findings/`
     entry, and if it works, the change lands in the codebase itself.
   - The novelty test is operational: Claude doesn't know it and can't find it online.
     State the verdict in the PR ("novel because…" / "settled by…"). **Uncertain →
     treat as novel** — deleting a finding later is cheap, re-discovering one isn't.

**The base-model bar.** We have never finished training a base model — past runs died at
~200M tokens; a 124M GPT-2-small saw ~10B, so ours was ~50× undertrained and behaved
"drunk" (locally fluent, globally lost). That is not "small models can't work"; it's a
model that never finished school. At ~138M params the right target is *not* "useful /
gets the prompt" (unreachable at this scale) — it's **match GPT-2-small on a standard
yardstick** (LAMBADA last-word accuracy, or held-out perplexity in the known range).
Until a *vanilla* model trained to completion hits that floor, no architecture ablation
is interpretable: if the base is mush, you can't tell whether a change helped or just
stirred the mush. The full base run is therefore also the validity check on the whole
pipeline — if a plain model on the full budget *can't* reach GPT-2-small, the bug is in
the data / LR / tokenizer / eval, and that gets fixed before any clever-architecture work.

## The model registry & reproducibility

The weights are the *product*; the apparatus that makes and judges them is the *code*,
and that's where the rigor lives. With determinism, the relationship flips in our favor:

**Weights are a cache, not a treasure.** If a run is reproducible — same commit, config,
seed, tokenizer, data manifest — then losing the weights costs *compute to regenerate*,
not *knowledge*. So the irreplaceable artifact is the **recipe**, and the recipe is tiny
and lives in git. Two tiers:

- **Master copy (in git, tiny):** the model card — commit SHA, full config snapshot,
  tokenizer name, dataset manifest (prefill version + token count + filter settings),
  seed, final metrics (val ppl, the GPT-2-small yardstick), VRAM + wall-clock, and one
  line on why it's notable. Template: `docs/registry/MODEL_CARD_TEMPLATE.md`.
- **Cache (gitignored, regenerable):** the weights themselves, with a `sha256` to catch
  silent corruption.

Curate. Only a **champion + a couple of notable challengers** get a stored card —
"every run that finished" fills the disk and the registry with noise.

A stored, GPT-2-small-grade base model is also the thing that makes the "experiment a
lot" style *affordable*: future ideas **fine-tune or branch from it** instead of
pretraining from scratch — a 30-minute run instead of a 10-hour one. The registry pays
for itself in compute the first time you warm-start.

**Cross-model comparisons are observational, not causal.** Comparing two differently-built
models (more params, different data) is a sanity/regression check — "are we even in the
ballpark" — never proof that an idea works. Only the matched one-variable pair (rule 2)
can claim a cause. Keep the two straight.

**Storage convention.** The SSD (root fs, `runs/`) is the *live* tier — current training
runs, ablations, smoke tests. The 1TB HDD is the *cold* tier — mirror artifacts there
once they're old or done (champion weights, and the tokenized corpus, which is
regenerable but cost ~a day to build). Treat the HDD as dumb blob storage (copy files);
don't train off it or rely on its symlinks/permissions. This also keeps the near-full
SSD from filling. The tokenized corpus in `runs/data/` is sacred — never delete it to
free space; archive or surface it instead.

## How work is tracked — issues, labels, priority

`docs/ROADMAP.md` owns the narrative — the why, the order, the proof gates, the
graveyard of killed ideas. **GitHub issues own per-item state** — one closeable unit each.
When you spot a hypothesis worth testing — a smoke test, an ablation, a small or full
run — file an issue so the queue reflects what we actually intend to do. Progress and
live state live in issues, *not* in the codebase. Knowledge that is coupled to the code
(a finding about what an architecture did, tied to a commit) belongs in the repo
(`docs/findings/`); forward-looking state (what's running, what's next) belongs outside,
in issues. Working plans stay local and gitignored (`docs/plans/`, `aux*`).

**Type labels (what kind of work it is) — in priority order:**

1. **`architecture`** — the *repository's* architecture and environment. Comes before
   everything: the first job is an environment Claude can trust and operate
   programmatically, without surprises. (Note the name clash: a change to the *model's*
   architecture — say GQA → multi-head latent attention — is an **`idea`**, not this.
   This label is about the harness/repo, not the network.)
2. **`tools`** — actual code that is *not* LLM research per se: the harness, instruments,
   runners, CI. Comes second — tools are what let ideas be tested cheaply.
3. **`ideas`** — things to try on the LLM itself (architecture/recipe changes,
   hypotheses). Pick these in **any order, your judgment**. An idea may jump ahead of a
   tool only when it genuinely makes sense — usually when it's small.
4. **`optimization`** — makes the *code* cheaper in memory or compute **without changing
   what the model is**. Same model, fewer resources. (If it changes the model, it's an
   `idea`. GQA → MLA is an idea; chunking the cross-entropy to free activation memory is
   an optimization.) Can land any time it's ready.
5. **`documentation`** — changes to `.md`, skills, findings. Can land **any time**, even
   mid training-run. Doc-only commits (markdown and/or comments) need no issue — make
   them in their own small PR, judiciously.

**Orthogonal labels (combine with a type):**
- **Lane** — `cpu` runs alongside a GPU job; `gpu` is the single RTX 2060, a serial
  queue, one run at a time; `blocked` has an unmet dependency, stated as "Blocked by #N".
- **`bug`** — a defect; attaches to whichever type it lives in. A bug that **blocks the
  active lane** (e.g. a crash stopping the running GPU job) jumps the queue — fix what's
  in the way first. A bug on a path nobody is running waits its turn.

**The ready-queue.** An issue is ready when it's open, not `blocked`, has no assignee,
and its lane is free. The principle behind the priority order: anything that *affects another item* leads
— whether it changes the implementation or changes how we *think* (a result that reframes
the question). Repo-architecture, tools, and tests ripple downstream, so they lead; a
full training run is last because nothing depends on its output.

**Claiming work.** An issue with an assignee is being worked on — never start it.
Starting any issue means: check its linked PRs for prior work, then assign it. The
claim releases when the PR merges; a PR closed unmerged still owes its record first
(rule 5 above, plus the closing vocabulary), then unassign so the issue re-enters the
ready-queue. GPU runs additionally drop a "▶ started" comment — the card is a serial
queue, and it must be visible what's holding it.

**Closing the loop.** Every PR that addresses an issue links it with "Closes #N" so the
merge closes it. Closing *without* a PR uses a controlled vocabulary in the closing
comment so history stays greppable: "superseded-by #N", "negative-result", or
"wont-fix: <reason>".

## Repo map — what's where

Two architectures coexist, selected at launch by `MODEL_ARCH` (see `config.py`); they
have different param trees, so a run of one cannot resume the other's checkpoint:
- **`refiner`** — Plan A, `CausalRefiner` in `plan_a_model.py`: causal within-window
  depth recurrence (a shared block looped K times under a causal mask). The **live bet**
  — the depth-recurrence mechanism the project is built on.
  (`docs/findings/2026-06-13-plan-a-depth-recurrence-works.md`.)
- **`reasoner`** — `UniversalReasoner` in `model.py`: the original cross-window "hunch"
  design. The hunch is **proven inert** (`docs/findings/2026-06-13-cross-window-hunch-inert.md`),
  so this is effectively a vanilla random-depth transformer — kept as the control
  baseline. It is the current `MODEL_ARCH` default; the base-run plan switches to
  `refiner` (tracked in issues).

The reusable scaffold (stable across whatever idea we try next) vs the swappable
experiment (the arch behind the flag):

| Concern | Files |
|---|---|
| **Config (single source of truth)** | `config.py` — every architecture/training constant, the dtype policy, the arch selector |
| **Model — live** | `plan_a_model.py` (CausalRefiner), `layers.py` |
| **Model — control/graveyard** | `model.py` (UniversalReasoner) |
| **Training loop** | `trainer.py` (loop + data pipeline; + `plan_a_trainer.py` adapter), `start_training.py` (entry), `grad_step.py`, `optimizers.py`, `schedules.py`, `validation.py` (held-out probe) |
| **Data** | `prefill.py` (tokenize corpus → `runs/data/`), `data_loaders.py`, `tools/data_curation/` |
| **Persistence & run state** | `checkpoint_utils.py`, `run_tracker.py`, `metrics_logger.py`, `monitor.py` |
| **Proof instrument** | `ablation_harness.py` — tiny toy-task depth ablations (parity / cumsum / state-tracking) at the *exact* arch we'd ship |
| **Diagnostics & smokes** | `tools/` — `eval_depth_curve.py`, `overfit_smoke.py`, `smoke_refiner_gpu.py`, `vram_headroom_smoke.py`, `dump_transcripts.py`, … |
| **Inference / plots** | `infer_local.py`, `plot_history.py` |
| **Tests** | `tests/` — invariants, determinism, golden-run, data hygiene. CPU by default (`FORCE_F32_COMPUTE`) so they run while the GPU trains; `RUN_TESTS_ON_GPU=1` for the real f16 path. CI runs them on every push/PR to `main`. |

Hardware reality: one **RTX 2060 (6GB, Turing)** — no bf16 tensor cores, so **f16
compute is the permanent policy** (`config.py`); the GPU lane is serial. Tokenizer is
**`r50k_base`** (50257 vocab, `VOCAB_SIZE=50304` padded); the embedding + tied LM head is
the single biggest VRAM line.

## How to dig into our past

- **What worked and what didn't** → `docs/findings/` (dated, one conclusion each, with
  the evidence and the relation to prior work; novel results only — non-novel outcomes
  live in their PRs, tombstoned in the ROADMAP graveyard). Start at `docs/findings/README.md`.
- **The why and the graveyard** → `docs/ROADMAP.md` (narrative + killed ideas with
  reasons so they stay dead).
- **Per-item state / what's live** → GitHub issues (`gh issue list`). The roadmap points
  at issues; issues never hardcode mutable plans the roadmap should own.
- **Design rationale for the live arch** → `docs/design/plan-a.md`.
- **Local-only scratch** (gitignored) → `docs/plans/`, `aux*` — working notes, not truth.

## Token Optimization Rules (RTK)

RTK (Rust Token Killer) is installed globally to save context during terminal output.
1. **Prepend `rtk`** to commands with massive output: `rtk git diff`, `rtk git status`,
   `rtk test` / `rtk cargo test` / `rtk npm test`, `rtk run <command>` for verbose logs.
2. RTK strips ANSI codes, truncates repetitive linter/test walls, and compresses output.
