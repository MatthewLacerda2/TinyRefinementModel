# Working agreement & research doctrine

How we work together, how we know a result is real, how the repo is laid out, and
where to dig into what we already learned. It should fit in your head in one sitting;
when it stops fitting, split a piece into a skill (below) instead of letting it sprawl.
It holds what is specific to this project. What the literature settles, Claude already
knows; this file doesn't restate it.

## How we work

Prefer simple language that explains, at a high level, what we or the code are doing.

We MUST obey Richard Sutton's "the bitter lesson": the model's intelligence comes from
scalable methods, not hand-crafted ones.

Structure a good architecture and readable code first, then tighten it. Compactness
serves readability; it is not the finish line. If the clearest version isn't the
densest, leave it clear — no clever one-liners nobody can debug.

A comment names the knob, says why this value, and links where the story lives: its
finding, its PR, or the test that guards it. The story is written once; retold in four
files, it drifts four ways.

Push back when it's earned:
- A feature that doesn't move the model's final performance: say so, and why.
- An idea the literature has **documented to fail**: flag it immediately. One it has
  **simply not tried yet**: stay curious — that is what research is.
- What the literature **settled in favour** and we don't have yet is work to do, not
  research to re-run: "adopt X, expect ~Y, source Z", behind a smoke check that it
  breaks nothing. Where our setting differs (f16, a 6GB card, batch 2), measure *only
  that part* (#26, #287, #382 are what forgetting this cost).

**No rule here is beyond question.** Rules make the work predictable, so every session
handles the same situation the same way. When one stops serving that, say so and change
it in a PR — never quietly work around it. A rule followed but not believed rots faster
than one argued with.

**Align before building.** If the idea isn't clear — to the owner or to you — stop: no
plan, no code, until it is defined for both of you.

Favor tiny smoke tests and ablations: knowing what holds and what breaks beats a
speculative improvement every time. **But just because something can run on the CPU
doesn't mean it should.** A CPU toy run shows a mechanism works at all; a verdict about
the language model needs the real model on the real data, on the card when it's free.

## How we know something worked

The owner's instinct is to experiment a lot, and it's right — but trying one idea ten
ways hands you a "winner" by chance. The cure is never fewer experiments; it's **cheap,
controlled, pre-committed** ones. Rules 1–5 are how a result earns belief; rule 6 is
how its apparatus gets cleaned up. The procedure is the `experiment` skill.

1. **Clear the noise floor.** A single-seed number is not a result. Measure
   seed-to-seed variance first; a delta counts only if it clears it, and you report the
   spread. Our own past deltas have sat inside the noise (0.001–0.005 nats).
   - **Compute the floor from the task**: two arms that both fail are "within 2σ" of
     each other, so a parity criterion needs an absolute floor arm beside it (#246).
   - **Gate on margin, not on failure**: a check that fires only after the failure is a
     post-mortem (#235: finite loss, 0.6% from the f16 ceiling).
2. **One variable, matched pairs.** Same seed, same data order, same config but the one
   knob. An unmatched comparison tells you nothing.
3. **Pre-register the kill criterion — and the prediction.** The threshold, against the
   noise floor, is written before the run so enthusiasm can't move it. The prediction
   tests the person: four about depth recurrence were all wrong in the same direction
   (#238, #242, #246). Write down what you expect, and that you were wrong last time.
4. **Earn the comparison with a strong baseline.** A win over an undertrained control
   is a mirage (see the base-model bar).
5. **Record every result, where it belongs.** Novelty decides the home:
   - **Not novel** (we confirmed what the literature settles): the PR is the record. If
     it kills or gates something, one tombstone line in the ROADMAP graveyard links it.
   - **Novel** (Claude doesn't know it and can't find it online — negatives included,
     they're the moat): a dated `docs/findings/` entry citing `Spec: <path>` or
     `Evidence: observational — <why>`. **Unsure → novel**: deleting a finding is cheap,
     rediscovering one isn't.
   - The PR says "novel because…" or "settled by…". `python -m instruments.audit`
     checks in CI whether a verdict can be trusted, never what it is. Exploration needs
     no ceremony; publication does.
6. **A tombstone takes its apparatus with it.** The PR that writes a graveyard entry
   deletes that line's harness and its `tests/apparatus/` guards in the same diff; its
   `specs/` stay, because findings cite them. This is the one rule that stops the suite
   from growing forever.

**Rules the build enforces** (each guarded by a test; this keeps only the why):
- **No hidden defaults on the hot path.** Every knob is a `Config` field in
  `trm/settings.py`, recorded in the run's metadata — an unnamed default is a recipe
  nobody chose.
- **Config is read once, by `Config.from_env`, and passed down.** No other environment
  read in `trm/` (paths go through `settings.location`), no knob as a module constant;
  only entry points read `CONFIG` (#475).
- **Outside data is parsed into a typed object at the boundary** — `Config`,
  `spec_file`, `ResumeState`, `RunMetadata` — so a misspelled key fails where it's read,
  never mid-run (#475, #477, #478, #578).
- **Dtype policy: matmuls in f16, accumulators in f32.** An accumulator that isn't f32
  has a line in `trm/config.py` saying why; that file, not this one, says which.
- **Every schedule declares its horizon** (steps or a fraction of the budget) and the
  launch banner prints both. A short pair inherits the shape of the run it informs;
  warmup is the exception.

**The base-model bar.** At ~138M params "useful" is unreachable; a base that is locally
fluent and globally lost is undertrained until proven otherwise, and a run that died
early licenses no conclusion. Two numbers on one yardstick (`instruments/yardstick/`,
exact match to lm-eval-harness), LAMBADA accuracy / ppl:
- **The gate — GPT-2-small, 0.3256 / 40.06**: what our size reaches on our budget. The
  base spec (`experiments/base/specs/001-plain-base.toml`) keeps at or above it; more
  than 0.05 below means a bug in data, LR, tokenizer or eval, fixed before any
  architecture work.
- **The neighbour — SmolLM2-135M, 0.4289 / 19.26**: our size with ~200× our data. The
  distance to it is the token axis, not a defect.

Until a vanilla model trained to completion clears the gate, no architecture ablation
is interpretable. A miss below the gate is a bug; a miss below the neighbour is budget.

## Weights, cards and storage

**Weights are a cache; the recipe is the master copy.** A reproducible run (commit,
config, seed, tokenizer, data manifest) costs compute to regenerate, not knowledge. So
the model card lives in git (`docs/registry/`), weights are gitignored with a `sha256`,
and only a champion and a couple of notable challengers get a card. Comparing two
differently-built models is a ballpark check, never a cause — only rule 2 claims one.

The SSD (`runs/`) is the live tier; the HDD (`/mnt/d_drive/TRM_cold/`) is cold blob
storage — copy files there, never train off it. **The tokenized corpus is sacred**:
never delete it to free space.

## How work is tracked

`docs/ROADMAP.md` owns the narrative: the why, the order, the gates, the graveyard.
**GitHub issues own per-item state**, one closeable unit each — file one for every
hypothesis worth testing. Knowledge coupled to code lives in the repo (rule 5); what's
running and what's next lives in issues; working plans stay in gitignored `docs/plans/`
and `aux*`.

**Type labels, in priority order** — the owner's phases: a lean, readable codebase and
the tools that measure it first, then what the literature settled, then our own bets.
1. **`codebase`** — code that reads clearly and stays lean, automation that makes a
   drift fail a test or a lint, rules that leave Claude no ambiguity.
2. **`tools`** — the harness, instruments, referees, runners, telemetry, CI. An
   instrument prints a plain, reproducible report: what it measured, the result, the
   speed, the cost in CPU, GPU, RAM and VRAM — numbers, never prose.
3. **`optimization`** — cheaper in memory, compute or tokens-to-quality **without
   changing what the model is** (chunked cross-entropy is; GQA → MLA isn't). Freed
   memory and speed decide what the model can be.
4. **`adopt`** — settled in the literature, not here yet: "Adopt X", source and
   expected effect in the body, gated by a smoke check. Before filing a model change,
   ask whether it is this.
5. **`architecture`** — the genuinely open changes (shape, recipe, hypotheses), judged
   by a pre-registered matched pair. Any order, your judgment.
6. **`documentation`** — `.md`, skills, findings. Lands any time, even mid-run; needs
   no issue.

Lanes (`cpu`, `gpu`, both, `blocked`) and the other labels (`local`, `base-gate`,
`bug`, `plan`, `agent`) are in the `work-tracking` skill.

**The ready-queue.** Ready = open, not `blocked`, unassigned, lane free. Whatever
affects another item leads — in code or in how we think. Within a tier: what others are
blocked by, then `gpu` items while the card is idle, then judgment.
`python -m instruments.queue` computes it (`--cloud` without this machine's card,
corpus and weights); when these rules change, it changes in the same PR.

**A bug found mid-task** gets fixed here or filed — never left unrecorded, because then
nobody looks again. Ask the owner only when there's a real choice: a fix that changes
what the model *is*, swells the PR, or has no clear right answer. Say plainly what a fix
does NOT cover ("guarded the symptom, root cause on #N").

**What Claude may do without asking.** Start any job under 48 hours of compute, as many
as a question needs (the limit is per job), and merge any PR it judges ready — CI
green, verdict recorded, what it doesn't cover said plainly. **A single job of 48 hours
or more — in practice a base run — waits for the owner**: surface it with its budget.
The permission waives asking, never the discipline: pre-registered, claimed, recorded.

Claim an issue before starting it (an assignee means hands off), close with
"Closes #N", **merge, never squash** (`gh pr merge <N> --merge`). Before a base run,
whatever only pays if the run has it lands first — the `base-gate` checklist is in the
`training-runs` skill.

## Repo map

**Four trees; the folder declares the kind.**

| if it… | it goes in | and it lives |
|---|---|---|
| is part of what the model *is* or how it trains | `trm/` | forever |
| belongs to one research line | `experiments/<line>/` | as long as that line |
| measures something, permanently | `instruments/` | forever |
| guards a property | `tests/{core,apparatus,expensive}/` | with what it guards |
| is per-issue scratch | `aux*`, `docs/plans/` | until you're done (gitignored) |

Nothing goes in the repo root (a test fails on a root `.py`). `trm/` never imports
`experiments/` or `instruments/`, and no research line imports another, so a tombstone
stays one folder's deletion. Worktrees go beside the checkout (`git worktree add
../trm-<name>`), never inside: a checkout reads only its own `.env`, so a sibling takes an absolute
`DATA_ROOT` from the shell (#541). Entry points
run as modules (`python -m trm.train.start`); the `Makefile` is the front door, and its
header lists every target — launch through it, never by hand-assembling supervisor flags.

There is one model: `PlainTransformer` in `trm/model/plain.py`. The refiner and the
cross-window reasoner before it live in `docs/findings/` and the graveyard (#292).

| Concern | Where |
|---|---|
| Config | `trm/settings.py` (`Config`, every knob); `trm/config.py` (dtype policy, non-knob constants) |
| Model | `trm/model/` — `plain.py`, `rope.py`; `build_model(config, rngs)` is the one factory |
| Training | `trm/train/` — `loop.py`, `trainer.py`, `start.py`; one concern per file |
| Data | `trm/data/` — `prefill.py` (corpus → tokens), `loaders.py` |
| Run state | `trm/runtime/` — `supervisor.py`, `launch.py`, `rewind.py`, `cold.py` |
| Inference | `trm/infer.py` |
| Verifiable worlds | `trm/rl/` — procedural Python tasks with tests, and the sandbox that runs a candidate |
| Specs | `experiments/<line>/specs/*.toml`, format in `docs/design/experiment-spec.md` |
| Research lines | `experiments/recipe/` (`tokens_to_ce.py`, the real pair trainer), `mix/`, `base/`; tombstoned lines keep only `specs/` |
| Instruments | `instruments/` — the `instruments` skill is the index |
| Tests | `tests/README.md` (tiers, and this machine's traps). CI runs core + apparatus, the audit, and `make lint` — run it before pushing and delete what it finds |

Hardware: one **RTX 2060 (6GB, Turing)** — no bf16 tensor cores, so f16 compute is
permanent; the GPU lane is serial. Tokenizer `r50k_base` (`VOCAB_SIZE=50304` padded);
the tied embedding is the biggest VRAM line.

## Skills — the how-to, loaded when the task needs it

| skill | for |
|---|---|
| `experiment` | pre-register → `make pair` → verdict → finding or tombstone |
| `instruments` | which instrument answers which question; the contract for a new one |
| `training-runs` | launching, watching, resuming and storing a GPU run; the base-gate checklist; OOMs |
| `work-tracking` | labels, the queue, claiming, parked drafts, CI, closing, merging |

## How to dig into our past

- **What worked and what didn't** → `docs/findings/README.md` (novel results only).
- **The why and the graveyard** → `docs/ROADMAP.md`.
- **What's live** → GitHub issues (`gh issue list`).
- **Design docs** → `docs/design/` (`plan-a.md` is the retired depth-recurrence design).
