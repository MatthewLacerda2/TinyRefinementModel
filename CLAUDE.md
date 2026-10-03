# Working agreement & research doctrine

This file is the whole context: how we work together, how we know a result is real,
how the repo is laid out, and how to dig into what we already learned. It is meant to
be small enough to read in one sitting — if it ever stops fitting in your head, that's
the signal to split a piece of it into a skill, not to let it sprawl.

## How we work

Prefer simple language that explains what we or the code are doing at a high level.

We MUST obey Richard Sutton's "the bitter lesson". The intelligence of our model must
come from scalable methods, not hand-crafted.

First structure a good architecture and write code that is readable and organized.
Then tighten it — denser, more compact — but compactness serves readability, it is not
the finish line. If the clearest version of something isn't the densest, leave it
clear. Don't end on clever one-liners nobody can debug later.

A comment names the knob and links where the story lives; the story is written once.
The incident behind a constant belongs in its finding, its PR, or the test that guards
it — the comment says what the thing is, why this value, and points there. A story
retold in four files drifts in four directions.

Push back when it's earned:
- If a feature or addition doesn't move the model's final performance, say so and say
  why it isn't pulling its weight.
- If an idea contradicts what the literature has settled, flag it immediately. But
  *calibrate*: push hard on **documented dead-ends**, stay curious about **genuinely
  untried** ground. Research means trying what the literature hasn't settled — don't
  suppress a novel idea just because it's unproven. The line is "documented to fail"
  versus "simply not yet tried."
- The other side of the same line: **what the literature settled *in favour* and we
  don't have yet is work to do**, not research to re-run. Say so up front — "adopt X,
  expect ~Y, source Z" — and land it behind a smoke check that it breaks nothing, not
  a matched pair asking whether it works. "Settled" means settled in the literature's
  setting; where ours differs (f16, a 6GB card, batch 2), measure *only that part* — a
  throughput bench for a shape the paper judged in tokens — never the whole question
  again. Muon's three pairs and an LR sweep to GPT-3's own table value are what
  forgetting this costs (#26, #287, #382).

**No rule here is beyond question.** Rules exist to guide the work and make it
predictable and programmatic, so any session handles the same situation the same way.
When one stops serving that, or costs more than it returns, say so and change it in a
PR; never quietly work around it. A rule that is followed but not believed rots faster
than one that is argued with.

Align before building. The user must have a clear, defined idea of what he's trying to
say. If the idea isn't yet clear — to him or to you — **stop**: don't plan, don't
implement. Get the idea defined for both of you first. Alignment of understanding comes
before everything downstream.

Favor smoke tests and ablations, preferably tiny so we test fast and know exactly what
works and what doesn't. Knowing — pinning down what holds and what breaks — beats a
speculative model improvement every time.

**Just because something can run on the CPU does not mean it should.** A CPU toy run
establishes that a mechanism works at all; a verdict about the language model comes only
from the real model on the real data — a mechanism can pass its toy gate and still be
suppressed on language. When the card is free, measure there —
`experiments/recipe/tokens_to_ce.py` is the real trainer for a few hours per arm-seed.
**Run-length tiers for real-model pairs** (the LR schedule completes in-run, ≥3 seeds,
the readout is tokens-to-target, never CE at a fixed step): recipe knobs (LR, optimizer,
weight decay, mix) ~512 opt steps ≈ 67M tokens ≈ 4h per arm-seed; architecture changes
~2,000 opt steps ≈ 260M tokens ≈ 15h per arm-seed (both well under the 48-hour line, so
Claude's to run — see "What Claude may do without asking"); anything a plain champion
can warm-start, a few hundred steps. Revisable defaults, not laws. **The `cpu` label
means buildable without the card, not preferably measured without it**: the lane stays
because cloud sessions build tooling while the card trains, but the measurement of an
idea goes to the card.

## How we know something worked

The user's instinct is to experiment a lot: try every idea several ways, pull pieces
out to see what happens. That instinct is right — and it is exactly the thing that
manufactures false wins if it isn't disciplined. Trying one idea ten ways hands you a
"winner" by chance alone. So the cure is never *fewer* experiments; it's **cheap,
controlled, pre-committed** experiments. Rules 1–5 are how a result earns belief; rule 6
is how the apparatus that produced it gets cleaned up afterwards:

1. **Clear the noise floor.** A single-seed number is not a result. We measure
   seed-to-seed variance first (what "no real difference" looks like), and a delta only
   counts if it clears that floor. Report the spread, not just the point estimate. Our
   own past deltas have sat *inside* the noise (0.001–0.005 nats) — that is what an
   unguarded comparison looks like.

   **Compute the floor from the task; never assume it from the vocabulary.** Two arms
   that both *fail* are trivially "within 2σ" of each other, so a parity criterion
   without an absolute floor beside it reads a shared failure as a pass (#246: both
   arms "at parity" had learned to guess the mode). Measure the floor by sampling the
   generator, and declare it as an arm.

   **A check that can only fire after the failure is a post-mortem, not a gate.** Gate
   on *margin*: a finiteness check stayed green while the model trained itself to 0.6%
   of the f16 ceiling (#235).

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

   **Register the prediction too, beside the criterion.** The threshold tests the idea;
   the prediction tests the person holding it, and only the second one exposes a
   *pattern*: four registered predictions about depth recurrence (#238/#242/#246) were
   all wrong **in the same direction**, crediting the mechanism too much. Write down
   what you expect, and write down that you were wrong last time.

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
   - **A findings entry names where it came from.** Either `Spec: <path>` (the
     pre-registered spec that judged it, and it must resolve) or
     `Evidence: observational — <why no control applies>`. Observational is a real
     category — an anomaly found by accident while running something else — not a
     loophole. Enforced by the `finding-cites-spec` rule of
     `python -m instruments.audit`, which CI runs over whatever a change
     can reach; entries before 2026-09-12 are grandfathered, since rewriting their
     provenance now would be inventing it. The same audit asks of every spec
     whether its verdict can be *trusted* — floor, seeds, sigma, criteria committed
     before results, verdict reproducible — and never what the verdict is (that
     stays with `instruments/verdict.py`). **Exploration needs no ceremony;
     publication does.** Start a spec with `python -m instruments.experiment <path> --new`.

6. **A tombstone takes its apparatus with it.** The knowledge is what we keep; the
   scaffolding that produced it is not. So the PR that writes a graveyard entry also
   deletes that line's harness and its `tests/apparatus/` guards, in the same diff —
   grep `tests/apparatus/` before opening it. This is the one rule that stops the
   suite from growing forever: tests aren't retired by judgment calls nobody makes,
   they're retired by the kill they belong to. The finding survives the harness.

**Rules the build enforces, so nobody has to remember them.** Each is guarded by a
test, or by an open issue that adds one; this file keeps only the *why*.
- **No hidden defaults on the hot path.** Every knob the optimizer or model reads
  is a field of `Config` in `trm/settings.py` and recorded in the run's metadata: an
  unnamed default is a recipe nobody chose, that a library upgrade can change silently.
- **Config is read once, by `Config.from_env`, and passed down.** No other read of
  the environment for a knob in `trm/` (paths go through `settings.location`), and no
  knob is a module constant: a library module takes the `Config` (or the value) its
  caller hands it, and only entry points read this process's `CONFIG`. Import-time
  reads scattered over eleven files froze values in whatever order modules loaded and
  forced the `start.py` / `run_budget` ordering dance (#475).
- **The dtype policy is about compute, not state.** Matmuls run f16; anything that
  *accumulates* — the residual stream, the gradient accumulator, an optimizer
  moment — belongs in f32: past ~4k, f16's 10 mantissa bits round a block's O(1)
  contribution to nothing. An accumulator that is not f32 has a line in `config.py`
  saying why; read that file, not this one, for which ones are f32 today.
- **Every schedule declares its horizon** — absolute steps or a fraction of the
  budget — and the launch banner prints both, so a short pair cannot silently train
  on a different mix than the run it licenses. Judgment call that stays
  prose: *a short pair inherits the shape of the run it informs* — warmup is the
  legitimate exception, since it stabilizes optimizer state, not the recipe.

**The base-model bar.** At ~138M params the right target is *not* "useful / gets the
prompt" (unreachable at this scale). A base that is locally fluent and globally lost
is undertrained until proven otherwise, and a run that died early — for whatever
reason — licenses no conclusion about the architecture. Two numbers on one
yardstick, with different jobs:

- **The gate** is what our size reaches on our token budget: **GPT-2-small** (124M, ~10B
  tokens), LAMBADA last-word accuracy 0.3256 / ppl 40.06, measured by our own instrument
  (`instruments/yardstick/`, exact match to lm-eval-harness). The base spec
  (`experiments/base/specs/001-plain-base.toml`) judges against it: at or above is KEEP,
  more than 0.05 below is the "fix the pipeline" branch — the bug is in the data / LR /
  tokenizer / eval, and it gets fixed before any clever-architecture work.
- **The neighbour** is what our size reaches with ~200× our data: **SmolLM2-135M**
  (~2T tokens, WSD), 0.4289 / 19.26 by the same instrument. It is the reference for
  "top for its size", and the distance to it is the token axis, not a defect. A run
  that lands between the gate and the neighbour did what its budget allows; it is a
  champion, and the gap above it is where the data-axis research starts.

Until a *vanilla* model trained to completion clears the gate, no architecture ablation
is interpretable: if the base is mush, you can't tell whether a change helped or just
stirred the mush. The full base run is therefore also the validity check on the whole
pipeline. Read a miss below the gate as a bug, never a miss below the neighbour.

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
runs, ablations, smoke tests. The 1TB HDD (`/mnt/d_drive`, under `TRM_cold/`) is the
*cold* tier — mirror artifacts there once they're old or done (champion weights, and the
tokenized corpus, which is regenerable but cost ~a day to build). Treat the HDD as dumb
blob storage (copy files); don't train off it or rely on its symlinks/permissions. This
also keeps the near-full SSD from filling. A run launched with `COLD_ROOT` set does its
own share (`trm/runtime/cold.py`): milestones mirrored, a full-state copy per stall
window, mirrored milestones pruned from the SSD to a margin. The tokenized corpus in
`runs/data/` is sacred — never delete it to free space; archive or surface it instead.

## How work is tracked — issues, labels, priority

`docs/ROADMAP.md` owns the narrative — the why, the order, the proof gates, the
graveyard of killed ideas. **GitHub issues own per-item state** — one closeable unit each.
When you spot a hypothesis worth testing — a smoke test, an ablation, a small or full
run — file an issue so the queue reflects what we actually intend to do. Progress and
live state live in issues, *not* in the codebase. Knowledge that is coupled to the code
(a finding about what an architecture did, tied to a commit) belongs in the repo —
`docs/findings/` when it's novel, a ROADMAP-graveyard tombstone plus its PR otherwise
(rule 5 above); forward-looking state (what's running, what's next) belongs outside,
in issues. Working plans stay local and gitignored (`docs/plans/`, `aux*`).

**Type labels (what kind of work it is) — in priority order:**

1. **`tools`** — the infrastructure: code that is *not* LLM research per se, built to
   develop, measure and investigate — the harness, instruments, runners, telemetry,
   CI. Comes first because every later judgment stands on it. Above all, an instrument
   prints a **plain, reproducible report of how something ran** — what it measured,
   the result, the speed, and what it cost in CPU, GPU, RAM and VRAM — numbers a
   person can compare across runs to say whether a change made things faster or
   better, never prose a model wrote about them.
2. **`architecture`** — how things are *defined*, for both the codebase and the model.
   For the repo: **code that reads clearly and stays lean**, with modules whose shape
   is obvious; **automation that keeps it on the rails**, so a regression or a drift
   fails a test or a gate; **rules that make Claude's work unambiguous**, written where
   a fresh session finds them. For the model: what it is and how it trains — its
   shape (GQA → multi-head latent attention), its recipe (LR, mix, weight decay) and
   its hypotheses. Within the tier, **any order, your judgment**. A model change whose
   *outcome is uncertain* — "maybe this works, I don't know" — is judged by a matched
   pair. A directed fix with a known method (the document separator masked as pad,
   #373) also carries **`bug`**: nobody is wondering whether to do it, so a smoke
   check that it breaks nothing gates it, not a pair. **Before filing a model change,
   ask whether the literature settled it.** If it did, it is a `bug` from birth: title
   it "Adopt X", and put the source and expected effect in the body.
3. **`optimization`** — makes the *code* cheaper in memory or compute **without changing
   what the model is**. Same model, fewer resources. (GQA → MLA changes the model, so
   it is `architecture`; chunking the cross-entropy to free activation memory is an
   optimization.) Speed and memory make every later run cheaper, and they decide what
   the model can be — freed memory decides whether a layer, a batch or a longer window
   fits, and speed decides how many tokens a base run buys in its days.
4. **`documentation`** — changes to `.md`, skills, findings. Can land **any time**, even
   mid training-run. Doc-only commits (markdown and/or comments) need no issue. Fold a
   small one into a PR already in flight; open its own small PR only when none is.

**Orthogonal labels (combine with a type):**
- **Lane** — `cpu` runs alongside a GPU job; `gpu` is the single RTX 2060, a serial
  queue, one run at a time; `blocked` has an unmet dependency, stated as "Blocked by #N".
  **Both `cpu` and `gpu` = partial-cpu**: the wiring, tests, and smoke logic are
  buildable on CPU now, but the confirming measurement needs the card. Do the CPU half
  whenever (cloud sessions parallelize freely, the card doesn't), park it as a **draft
  PR** stating exactly what waits on the GPU, and the GPU tail takes its turn in the
  serial queue when the card is back. The claim (assignee + draft PR) keeps the parked
  issue out of the ready-queue meanwhile. The draft's "what waits" section is a
  **resume protocol a context-free session can execute**: exact commands (including
  the env/config knobs that make them hit the changed code), what counts as pass
  against what baseline, and the pre-named fallback if it fails — finishing must need
  only the card, never this conversation's memory. (Template case: PR #98 / #84.)
  Undraft with `make ready PR=N`, and wait on a push with `make ci-wait PR=N`: CI
  skips drafts and re-runs on undraft, so the checks you see right after are the
  draft's skipped set — never push an empty commit to "retrigger" (#539).
- **`local`** — needs this machine's trained weights, tokenized corpus or HDD. A
  session without them (a cloud session) cannot finish it. Orthogonal to lane: a
  `cpu` item can be `local` (a scan of the corpus), and a `gpu` item need not be.
- **`base-gate`** — must land before the next base run (the test under "Before a base run
  launches", below). Orthogonal to type: an optimization, a telemetry tool and a model
  bug can all carry it.
- **`bug`** — a defect; attaches to whichever type it lives in. A bug that **blocks the
  active lane** (e.g. a crash stopping the running GPU job) jumps the queue — fix what's
  in the way first. A bug on a path nobody is running waits its turn.

**Spotting a bug mid-task.** Claude decides what to do with it — fix it in the current
task, record it on a related issue, open its own issue, or surface it for the owner to
call. **If there is nothing to decide, don't ask:** fix it in the next related PR, or
file the issue and work it. Bring it to the owner only when there is a real choice to
make — a fix that changes what the model *is*, one that would swell the PR past what a
reviewer can hold, or a defect whose right answer is genuinely unclear.

The failure mode this exists to prevent is a bug going *unrecorded* because it was
inconvenient to the task in hand. A bug found and left unwritten is worse than one never
found, because now nobody will look again. Two habits follow:

- **A bug found while building something else still gets written down**, even if it is
  not fixed. If it is not in this PR, it is an issue, and the PR says so.
- **Say plainly what a fix does NOT cover.** "Guarded the symptom, root cause on #N" is a
  complete answer; "fixed" when only the symptom was addressed is not.

**The ready-queue.** An issue is ready when it's open, not `blocked`, has no assignee,
and its lane is free. The principle behind the priority order: anything that *affects
another item* leads — whether it changes the implementation or changes how we *think*
(a result that reframes the question). Tools ripple downstream into every measurement,
so they lead; within `architecture`, the repo's structure and a result that reframes the
model's question lead a pair or sweep nothing depends on. A base run is the opposite: ablations cannot be read without it and warm-starts
need it, so the `base-gate` list is a launch checklist to close or waive, not a queue to
drain.

**`python -m instruments.queue` computes it.** It ranks only what these rules decide —
tier order, then issues other open issues are blocked by, and **when the card is idle,
`gpu`-lane items first within their tier** (an idle card is the scarce resource going
to waste; see "measure there" above) — and says so where they stop deciding (within a
tier is judgment). It also surfaces labels it can check and that fail:
a `blocked` whose blockers are all closed (an open PR named as a blocker counts as
open), an issue with no type label. And it surfaces every issue untouched for more
than `STALE_DAYS` (21) — neither it nor a PR claiming it — unless it waits on an open
blocker or on the card as a parked draft: each is kept with a comment saying why,
which resets the clock, or closed with the closing vocabulary below. When the rules
here change, the tool changes in the same PR; prose and command must not drift.

**Cloud sessions are the exception.** The project is built for this machine: its
card, its corpus, its weights. A session without them (no `runs/data/`) takes work
from `python -m instruments.queue --cloud`, which drops the `gpu`-only lane and
everything `local`. It builds and tests; it does not measure. A GPU tail becomes a
draft PR with its resume protocol, and the measurement runs here.

**What Claude may do without asking (the owner's standing permission).**
- **Start any job that takes under 48 hours to compute**, and as many of them as a
  question needs: smoke tests, ablations, matched pairs, sweeps, small models, and
  Claude's own hypotheses about what works or doesn't. The limit is per job, not per
  question — a sweep of fifteen 4-hour runs is sixty hours in all and still needs
  nobody's permission, because no one job of it is long.
- **Merge any PR Claude judges ready**, its own included (with a merge commit, below).
  Ready means what this file asks of it: CI green, the verdict recorded, what the PR
  does *not* cover said plainly.

The one thing that waits for the owner: **starting any single job of 48 hours or more**
— in practice a base run. Surface it with its budget and wait. The permission waives
asking, never the discipline: every experiment is still pre-registered through the
referee, claimed on its issue, and recorded per rule 5.

**Before a base run launches: whatever only pays if the run has it.** The test is two
questions, not a label. *Can it be judged before the run* (does it work, is it worth
it)? And *does its benefit need the run to have been trained or logged with it*? Both
yes → it lands before the run starts, because afterwards is too late for these weights.
That covers anything that makes the run lighter or faster (its days buy more tokens),
anything that changes what the weights learn (recipe, shape, a model bug),
and every piece of telemetry (a metrics column, a per-block reading, a margin the
supervisor watches): a log the run did not write cannot be recovered from its
checkpoints. What the finished weights can use at any time does not gate the run: a KV
cache for generation, an eval, a plot of logs that already exist. An issue that
passes the test but cannot land in time is waived by the owner by name, not skipped
silently. **The `base-gate` label marks the issues that pass the test**, applied when
an issue is filed or when the test is re-asked of it; the base run's launch checklist is
the open `base-gate` list, empty or waived.

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

**Merge, never squash.** PRs land with a merge commit (`gh pr merge <N> --merge`), not
squash or rebase, so every commit on the branch stays in main's history.

## Repo map — what's where

**Four trees, and the folder is the declaration of kind.** A new file has exactly
one home, and the rule is short enough to hold in your head:

| if it… | it goes in | and it lives |
|---|---|---|
| is part of what the model *is* or how it trains | `trm/` | forever |
| belongs to one research line | `experiments/<line>/` | as long as that line |
| measures something, permanently | `instruments/` | forever |
| guards a property | `tests/{core,apparatus,expensive}/` | with what it guards |
| is per-issue scratch | `aux*`, `docs/plans/` | until you're done (gitignored) |

Nothing goes in the repo root — there are no `.py` files there, and
`tests/core/test_package_layout.py` fails the build if one appears. Entry points are run
as modules: `python -m trm.train.start`, `python -m trm.data.prefill`, `python -m trm.infer`.
The front door is the `Makefile` (`make test`, `make test-affected`, `make gate`,
`make launch BUDGET=…`, `make report RUN=…`): launch a base run through it, never by
assembling supervisor flags by hand.

**The direction is enforced, not just intended:** `trm/` never imports from
`experiments/` or `instruments/`, and one research line never imports another — so
tombstoning a line stays a single folder deletion (rule 6). The same test holds that line.

Three architectures coexist, selected at launch by `MODEL_ARCH` (see `trm/settings.py`).
They have different param trees, so a run of one cannot resume another's checkpoint,
and resuming a run that is not `plain` requires naming its architecture:
- **`plain`** — `PlainTransformer` in `trm/model/plain.py`: N distinct causal blocks,
  no loop, no gate, no time signal, no depth dial. **The default and the live
  architecture.**
- **`refiner`** — `CausalRefiner` in `trm/model/refiner.py`: a shared block looped K
  times under a causal mask (depth recurrence). Not the live bet — why is in the
  ROADMAP and its findings. Kept selectable because the 4B champion is one, and
  loading that checkpoint requires `MODEL_ARCH=refiner`.
- **`reasoner`** — `UniversalReasoner` in `trm/model/reasoner.py`: effectively a
  vanilla random-depth transformer, kept as a control baseline.

| Concern | Files |
|---|---|
| **Config (single source of truth)** | `trm/settings.py` — `Config`, every knob a launch can set (the arch selector included), read once from the environment and recorded whole; `trm/config.py` — the dtype policy and the constants that are not knobs. Entry points hand this process's `CONFIG` down; library code takes a `Config` |
| **Model contract** | `trm/model/contract.py` — what the loop requires of a model (tokens + depth → predictions + auxiliary terms). Every arch implements this; the loop knows nothing else about any of them |
| **Model — live** | `trm/model/plain.py` (PlainTransformer), sharing `Block` with `refiner.py`, plus `rope.py`; `trm/model/__init__.py` `build_model(config, rngs)` is the one factory every entry point builds through |
| **Model — retired/control** | `trm/model/refiner.py` + `refiner_lm.py` (CausalRefiner — retired as the bet, kept to load the champion), `trm/model/reasoner.py` + `layers.py` (UniversalReasoner and its block) |
| **Training loop** | `trm/train/` — `trainer.py` (loop + data pipeline), `start.py` (entry); the rest is one concern per file |
| **Data** | `trm/data/` — `prefill.py` (tokenize corpus → `runs/data/`), `loaders.py` |
| **Persistence & run state** | `trm/runtime/` — `layout.py` (jax-free: checkpoint names, retention), `supervisor.py` (unattended runs: budget stop, divergence/stall kills, crash relaunch — `python -m trm.runtime.supervisor`), `rewind.py` (resume from an earlier checkpoint — `python -m trm.runtime.rewind`); the rest is one concern per file |
| **Inference** | `trm/infer.py` |
| **Verifiable worlds** | `trm/rl/` — `tasks.py` (procedural Python problems with their own tests, split train/held-out by a hash of the instance), `sandbox.py` + `_sandbox_child.py` (run a candidate under kernel limits and say what happened). The world the model is meant to learn in by trying; it trains nothing on its own |
| **Experiment specs** | `experiments/<line>/specs/*.toml` — the pre-registration as a file a machine can apply (hypothesis, arms, criteria, kill/keep bars), refereed by `instruments/verdict.py` and run by `python -m instruments.experiment <spec>`. Format: `docs/design/experiment-spec.md` |
| **Research lines** | `experiments/depth/` — `ablation_harness.py` (tiny toy-task depth ablations at the *exact* arch we'd ship), `eval_refiner_transfer.py` (the depth-transfer probe); `experiments/scratchpad/harness.py` |
| **Instruments** | `instruments/` — `verdict.py` (the referee: pre-registered spec + recorded numbers → KEEP/KILL/INCONCLUSIVE; pins σ_pooled so findings stop recomputing it by hand), `experiment.py` (the runner: gate → sweep → record → judge → findings draft) and `results.py` (the `RESULT {...}` line harnesses print for it), `queue.py` (the ready-queue: what to work on next, and why), `yardstick/` (LAMBADA: GPT-2-small is the gate and the calibration, SmolLM2-135M the neighbour), the smokes (`overfit_smoke`, `smoke_refiner_gpu`, `vram_headroom_smoke`, …), `bench_train_step`, `mem_profile`, `timemachine`, `milestone_report`, `dump_transcripts`, `plots`, `python_world` (pass@k on `trm/rl`'s tasks, per difficulty level — the "can it do something" readout beside LAMBADA's "can it talk") |
| **Tests** | `tests/` — three tier folders, `core/` · `apparatus/` · `expensive/`, and the folder is the declaration (`tests/README.md`; a test file dropped straight into `tests/` fails collection). CPU by default (`FORCE_F32_COMPUTE`) so they run while the GPU trains; `RUN_TESTS_ON_GPU=1` for the real f16 path. CI runs core + apparatus on every push/PR to `main`, plus a lint status: `ruff check .` (errors and bugs only) and `vulture` (dead code — functions, classes, constants nothing references), both configured in `pyproject.toml`. `make lint` runs both; run it before pushing, and delete what it finds. |

Hardware reality: one **RTX 2060 (6GB, Turing)** — no bf16 tensor cores, so **f16
compute is the permanent policy** (`trm/config.py`); the GPU lane is serial. Tokenizer is
**`r50k_base`** (50257 vocab, `VOCAB_SIZE=50304` padded); the embedding + tied LM head is
the single biggest VRAM line.

## How to dig into our past

- **What worked and what didn't** → `docs/findings/` (dated, one conclusion each, with
  the evidence and the relation to prior work; novel results only — non-novel outcomes
  live in their PRs, tombstoned in the ROADMAP graveyard). Start at
  `docs/findings/README.md`.
- **The why and the graveyard** → `docs/ROADMAP.md` (narrative + killed ideas with
  reasons so they stay dead).
- **Per-item state / what's live** → GitHub issues (`gh issue list`). The roadmap points
  at issues; issues never hardcode mutable plans the roadmap should own.
- **Design docs** → `docs/design/` (the experiment-spec format; `plan-a.md` is the
  depth-recurrence design, not the live arch).
- **Local-only scratch** (gitignored) → `docs/plans/`, `aux*` — working notes, not truth.
