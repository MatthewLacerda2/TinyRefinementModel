# The three test tiers

Every test file lives in exactly one tier folder. The folder is the declaration —
there is no marker to remember and no registry to update, and `tests/conftest.py`
refuses to collect a file dropped straight into `tests/`.

The tiers answer one question: **what breaks if I delete this?**

| tier | what it guards | when it runs | when it gets deleted |
|---|---|---|---|
| `core/` | a property the model or the training path must always have | every push and PR | only alongside the code it guards, in the same PR |
| `apparatus/` | the wiring of a research harness or a diagnostic instrument | every push and PR | **with its research line** — when a line is tombstoned in `docs/ROADMAP.md`, its harness and these tests go in the same PR |
| `expensive/` | exact numeric trajectories; anything measured in minutes | its own CI job, opt-in locally | same rule as `core/` |

## Why `apparatus/` exists

Tests accumulate here faster than anywhere else, and that is not a defect —
`test_scratchpad_harness.py` is 477 lines because the scratchpad ablations are
only interpretable if the wiring is provably what the write-up claims. The
problem was never that these get written; it is that nothing said when they
*stop* being load-bearing. A harness outlives its usefulness the moment its
research line is killed, and until now that moment had no consequence.

So the rule is deletion, not silence: apparatus tests keep running in CI (a
harness that rots quietly is worse than one that fails loudly), and the
tombstone is what removes them. If you are writing a ROADMAP graveyard entry,
grep `tests/apparatus/` before you open the PR.

## Where a new test goes

- Pins a bug that actually happened, or a property stated in a design doc → `core/`.
- Proves an ablation harness measures what its write-up says it measures, or
  that a `tools/` instrument reports honestly → `apparatus/`.
- Takes minutes, or compares against recorded numbers → `expensive/`.

If a test seems to belong in two tiers, it is usually two tests.

## Running them

```bash
pytest tests/core          # the fast loop while you work
pytest tests/              # everything; expensive/ self-skips without its env flag
RUN_GOLDEN=1 pytest tests/expensive # what CI's golden-run job forces on
RUN_TESTS_ON_GPU=1 pytest tests/    # the real f16 path, GPU must be free
```

Tests default to CPU (`FORCE_F32_COMPUTE`) so the suite stays runnable while a
training process owns the card.
