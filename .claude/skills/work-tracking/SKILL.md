---
name: work-tracking
description: How work moves on this repo — filing and labelling issues, picking from the ready-queue, claiming, parking a partial-cpu draft PR with its resume protocol, CI on undraft, closing, and merging. Use when filing an issue, choosing what to work on, opening, undrafting or merging a PR, or closing an issue without one.
---

# Tracking work

CLAUDE.md holds the rules that decide *what* to work on (the type tiers and the
ready-queue); `python -m instruments.queue` computes them. This is the mechanics.

## Filing

Every issue gets exactly one **type** (`codebase`, `tools`, `optimization`, `adopt`,
`architecture`, `documentation`), a **lane**, and whichever orthogonal labels apply.
An `adopt` title reads "Adopt X", and its body names the source and the expected effect.

| label | means |
|---|---|
| `cpu` | buildable without the card, so it can run beside a GPU job. Not "measure on the CPU": when the card is free, measure there |
| `gpu` | needs the RTX 2060, a serial queue, one run at a time |
| `cpu` + `gpu` | partial-cpu: build and test now, the confirming measurement waits for the card (below) |
| `blocked` | an unmet dependency, stated in the body as "Blocked by #N" |
| `local` | needs this machine's weights, corpus or HDD; a cloud session can't finish it |
| `base-gate` | must land before the next base run (the test is in the `training-runs` skill) |
| `bug` | a defect, in whatever tier it lives; one that blocks the active lane jumps the queue |
| `plan` | an epic or a design not ready to start; the queue shows it as "not ready to start" |
| `agent` | Claude filed it on its own initiative, or opened a PR with no issue. Ranks like any other |

## Picking and claiming

`python -m instruments.queue` (or `--cloud` in a session without `runs/data/`: it drops
the `gpu`-only lane and everything `local`). Within a tier, past what the tool ranks,
it's judgment.

An assignee means someone is working on it, so never start it. To start an issue, check
its linked PRs for prior work, then assign yourself. A GPU run also posts a
"▶ started" comment, so it's visible what holds the card. An issue untouched for 21
days surfaces as stale: keep it with a comment saying why (which resets the clock), or
close it.

## Partial-cpu: the parked draft

Do the CPU half, open it as a **draft PR**, and leave the GPU tail for when the card is
free. The assignee plus the draft keep it out of the queue meanwhile. The draft's
**"What waits"** section is a resume protocol a session with no memory of this
conversation can execute:
- the exact commands, including the env/config knobs that make them hit the changed code;
- what counts as a pass, against what baseline;
- the fallback, named in advance, if it fails;
- for anything touching `trm/train/grad_step.py`:
  `RUN_TESTS_ON_GPU=1 pytest tests/expensive/test_gpu_identity.py`. The CPU ignores
  buffer donation, so only the card shows that class of bug (#540).

`instruments.queue` reads that section: a draft whose "What waits" names the card is
waiting on the GPU, not stale.

## PRs and CI

- Run `make lint` before pushing.
- A doc-only change (markdown and/or comments) needs no issue. Fold a small one into a
  PR already in flight.
- Undraft with `make ready PR=N`; wait on a push with `make ci-wait PR=N`. CI skips
  drafts and re-runs on undraft, so the checks visible right after undrafting are the
  draft's skipped set. Never push an empty commit to "retrigger" (#539).
- The PR body says what changed, the verdict if there was an experiment ("novel
  because…" / "settled by…"), and plainly what it does **not** cover.
- A bug found along the way is fixed here or filed as an issue, and the PR says which.
- Before a self-merge, run the `code-reviewer` agent; a "block" stops the merge.

## Closing

- A PR that addresses an issue says "Closes #N".
- Merge with a merge commit, never squash or rebase: `gh pr merge <N> --merge`.
- Closing without a PR uses one of: `superseded-by #N`, `negative-result`,
  `wont-fix: <reason>`.
- A PR closed unmerged still owes its record (rule 5) first; then unassign the issue so
  it re-enters the queue.
