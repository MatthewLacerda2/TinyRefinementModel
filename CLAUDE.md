Prefer a plain language version that explains what we, or the code, is doing, not a highly technical language. The user is trying to architect intelligence, not decorate the implementation.

First structure a good architecture and write code that is readable and organized. Then tighten it — make it denser and more compact — but compactness is in service of readability, not a finish line. If the clearest version of something isn't the densest, leave it clear. Don't end on clever one-liners nobody can debug later.

If the user is trying to implement features or additions that don't add to the LLM's final performance, you absolutely should push back and warn why what he is doing isn't adding to the table.

If the user's ideas go against what is well known and documented in the literature, let him know right away and push back if he's making a clear mistake unknowingly. But calibrate the pushback: push hard on documented dead-ends, and stay curious about genuinely-untried ground. Research means trying things the literature hasn't settled, so don't suppress a novel idea just because it's unproven — the line is "documented to fail" versus "simply not yet tried."

The user MUST show he has a clear and defined idea of what he is trying to communicate. Even before trying to plan, let alone implement, if the user does not yet understand what he is saying, you must postpone planning and implementing, in favor of getting the idea clear and defined for both you and the user. There must be an alignment of understanding, following the rules previously mentioned here.

You must favor smoke-tests and ablation, preferably tiny so we can test fast and know damn well every possibility of what works and doesn't. Trying things out so we always know what works and doesn't is preferable over model improvements.

## Issues & PRs (how work is tracked)
The roadmap is dual-tracked: `docs/ROADMAP.md` holds the narrative — the why, the order, the proof gates — and GitHub issues hold the per-item state, one closeable unit each. When you spot a new hypothesis worth testing — a smoke test, an ablation, a small or full training run — file it as an issue so the queue always reflects what we actually intend to do.

Every issue should end in a PR if it succeeds. An issue may be closed without a PR when it was made stale by another issue or PR, when it was tried and didn't work, or for another sensible reason — always state that reason in a closing comment so the history explains itself.

Order of priority — do the cheap, certain things first and save the worry-budget for last:
1. code readability and maintainability
2. architecture and organization
3. smoke tests
4. ablations
5. full training run

The principle behind the order: anything that affects another item takes priority — whether it changes the implementation or changes how we *think* about the problem (a test result, an ablation that reframes the question). Architecture, tools, and tests lead because they ripple into everything downstream; a full training run is last because nothing depends on its output. By the same rule, a bug that blocks the **active** lane (e.g. a crash that stops the running GPU job) jumps ahead — fix what's in the way before pulling the next item. "Blocks the active lane" is narrower than "is a bug": a bug on a path nobody is currently running waits its turn in the cpu lane.

Documentation-only changes — commits that touch only markdown and/or code comments — don't need an issue. Make them in their own issueless PR, judiciously, at any point.

Labels: `smoke`, `ablation`, `blocked`, `cpu`, `gpu`, `bug`, `documentation`, `roadmap`. The lane labels say what resource an item needs: `gpu` is the one RTX 2060 — a serial queue, one run at a time; `cpu` work (code, tests, tooling, docs) runs in parallel to a GPU run. `blocked` means an unmet dependency, stated as "Blocked by #N" in the body.

Picking the next item (the ready-queue): an issue is ready when it is open, not labelled `blocked`, and its lane is free (the `gpu` lane runs one job at a time; the `cpu` lane runs alongside). Every blocked issue must name its blocker as "Blocked by #N" so the pick is mechanical, not guesswork. When you start an issue's run, assign the issue and drop a "▶ started" comment so it is visible what is holding the single GPU.

Every PR that addresses an issue must have that issue assigned to it — link it with "Closes #N" in the PR body so the merge closes the issue. Closing *without* a PR uses a controlled vocabulary in the closing comment so the history stays greppable: "superseded-by #N", "negative-result", or "wont-fix: <reason>".

## Token Optimization Rules (RTK)
This system has Rust Token Killer (RTK) installed globally. To save context window and avoid token bloat during terminal tool executions, adhere to these rules:

1. **Prepend Token-Heavy Commands:** Always prepend `rtk` to commands that generate massive terminal outputs.
   - Use `rtk git diff` instead of `git diff`
   - Use `rtk status` or `rtk git status` for large repository states
   - Use `rtk test` or `rtk cargo test` / `rtk npm test` for running test suites
   - Use `rtk run <command>` for verbose compiler outputs or logs

2. **Expected Behavior:** RTK will automatically strip ANSI escape codes, truncate repetitive linter/test walls of text, and compress the output.
