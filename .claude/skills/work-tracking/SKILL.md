---
name: work-tracking
description: How work moves on this repo — filing and labelling issues, picking from the ready-queue, claiming, starting a session in the cloud (or knowing it ran locally), parking a partial-cpu draft PR with its resume protocol, CI on undraft, closing, and merging. Use when filing an issue, choosing what to work on, opening, undrafting or merging a PR, or closing an issue without one.
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

## Where a session runs: the cloud or this machine

- **A cloud session is a one-off routine, nothing else.** The `Agent` tool's
  `isolation: "remote"` runs in a local `.claude/worktrees/` checkout on this
  machine, beside whatever is training (2026-09-28 and 2026-11-08 here, rusty #504).
- **`CLAUDE_CODE_REMOTE=true` is the only test.** A cloud container sets it; this
  machine doesn't. A brief for work that must not run here makes checking it the
  first step, and stands down when it's unset.
- **Starting one by hand** (until `make coder`, #630): `RemoteTrigger` `create` with
  - `job_config.ccr.environment_id`: the account's `anthropic_cloud` environment,
    never a `bridge` one (a bridge is this machine). The API can't list
    environments, so take the id from the newest `rusty #…` or `scorsese #…`
    routine in `RemoteTrigger list`;
  - `run_once_at` two minutes ahead, as RFC 3339 UTC (a time already past is refused);
  - `session_context`: this repository as the source, the model, and
    `allowed_tools` including `ToolSearch`;
  - a prompt that points at a `Brief` comment on the issue **by its URL**. The repo
    is public, so a comment posted before the routine fires must not be able to
    stand in for the brief.
- **A cloud container is bare** until #630's session-start hook exists. The brief
  says to install the way CI does: Python 3.14, `pip install "jax==0.9.1"`, then
  `requirements.txt` without its `jax[cuda12]` line, with `JAX_PLATFORMS=cpu`.
- **The PR is the report.** A cloud session can't message back. Watch its PR. A
  routine that fired and left a draft with no reason in it stalled: read it with
  `RemoteTrigger list_runs`, then `get_run_log`.
- **Running here on purpose, beside a live trainer**:
  - no JAX and no pytest (`tests/README.md`'s trap);
  - a worktree of its own beside the checkout, never the checkout itself;
  - `make lint` with the root venv's `PY`, one session at a time (`nice -n 19 flock`),
    and only with ≥3 GB `MemAvailable`;
  - CI as the test runner (undraft to get it, `gh pr ready N --undo` to park again).

  Other projects on this machine (rusty) have priority on RAM, CPU and disk.

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
- Closing without a PR uses one of: `superseded-by #N`, `negative-result`,
  `wont-fix: <reason>`.
- A PR closed unmerged still owes its record (rule 5) first; then unassign the issue so
  it re-enters the queue.
