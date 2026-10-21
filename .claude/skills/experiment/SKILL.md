---
name: experiment
description: Pre-register, run and judge a matched pair or sweep on this repo, then record the result where it belongs. Use when testing whether a model or recipe change works, writing an experiment spec, launching `make pair`, reading a verdict, or writing a finding or graveyard tombstone.
---

# Running an experiment

The principles are CLAUDE.md's rules 1–6. This is the procedure that applies them.
The spec format is `docs/design/experiment-spec.md`; read it when writing one.

**First, does it need a pair at all?**
- Settled in the literature → it is `adopt`: one smoke check that it breaks nothing,
  plus a measurement of only what our setting changes (f16, the 6GB card, batch 2). No spec.
- Exploring (poking at a run, a toy probe nobody will cite) → no ceremony.
- A claim someone will rely on → a spec. Publication needs the ceremony; exploration doesn't.

## 1. Pre-register

```
python -m instruments.experiment experiments/<line>/specs/<issue>-<slug>.toml --new
```

writes a skeleton with the runner's two traps already avoided. Fill in:
- **arms**: the control is identical but for one knob — same seeds, same data order,
  same backend (never resume a half-CPU sweep on the GPU).
- **a floor arm** whenever a criterion is parity: two failing arms are "within 2σ" too (rule 1).
- **criteria** against the noise floor, plus the kill/keep bars.
- **the prediction**, and the direction you were wrong last time (rule 3).
- **length**, from these defaults for real-model pairs: the LR schedule completes
  in-run, ≥3 seeds, and the readout is tokens-to-target (`instruments/curves.py`),
  never CE at a fixed step.

  | change | opt steps | tokens | per arm-seed |
  |---|---|---|---|
  | recipe knob (LR, optimizer, weight decay, mix) | ~512 | ≈67M | ≈4h |
  | architecture | ~2,000 | ≈260M | ≈15h |
  | anything a plain champion can warm-start | a few hundred | | |

  A short pair inherits the schedule *shape* of the run it informs; warmup is the exception.

**Commit the spec before any result exists.** The audit checks that the criteria commit
came first.

## 2. Run

Claim the issue and drop a "▶ started" comment (the card is a serial queue). Then:

```
make pair SPEC=experiments/<line>/specs/<id>.toml   # detached unit trm-pair, log runs/<id>.log
make pair-status                                    # unit, arms recorded, current step, disk
make pair-pause / make pair-resume                  # resume continues each arm from its checkpoint
```

The gate runs `tests/core` first; `python -m instruments.pair start <spec> --no-gate`
skips it when it already passed on this exact code.
- **Don't pull the root checkout while a pair runs.** Every arm imports `trm/` from root;
  work in a sibling worktree.
- A partial arm that died restarts from zero: resume only *estimates* the data position,
  which breaks same-data-order matching.
- **Never start a GPU job as a Claude background task.** Claude Code kills those under low
  RAM, and the trainer dies with them. `make pair` detaches properly.

## 3. Judge

The runner appends `[results.*]` to the spec and prints the referee's verdict
(KEEP / KILL / INCONCLUSIVE), then `pair_report` (the CE curve against the control, the
seeds, activation margins, pace). To re-read later:
`python -m instruments.pair_report <spec> --runs <dir>`. Then `make audit` checks
whether the verdict can be trusted (CI runs it too).

Read the σ with its seed count in mind: `verdict.py` compares raw multiples of σ_pooled,
and with 3 seeds a 2σ gap is weaker evidence than the number suggests (#599).

## 4. Record (rule 5)

- **Not novel** → the PR is the record (what ran, numbers, verdict). If it kills or gates
  something, add one tombstone line to the `docs/ROADMAP.md` graveyard linking the PR.
- **Novel, or unsure** → `docs/findings/<date>-<claim>.md`, starting from the draft the
  runner wrote into the run folder. It must cite `Spec: <path>` or
  `Evidence: observational — <why no control applies>`, and gets one line in
  `docs/findings/README.md`.
- **A kill takes its apparatus with it (rule 6):** the same PR deletes the line's harness
  and its `tests/apparatus/` guards. Grep `tests/apparatus/` first. The `specs/` stay.

The PR says "novel because…" or "settled by…".
