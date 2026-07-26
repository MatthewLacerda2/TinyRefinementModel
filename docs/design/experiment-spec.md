# The experiment spec — a pre-registration a machine can apply

**Status:** build 1 of #40 landed; the runner that consumes these (build 3) is next.

## Why a file and not prose

Rule 3 says write the kill criterion down before the run, so enthusiasm cannot
move the goalpost after the number arrives. That works — every finding in
`docs/findings/` has an honest pre-registration in it. But a threshold written in
prose still needs *a reader* to apply it, and the reader is the same person who
wants the idea to work.

A threshold written in a spec file is applied by `instruments/verdict.py`, which
has no opinion about the outcome. The spec is the pre-registration; the module is
the referee; the human is still the judge on what to do about the verdict.

## Shape

One TOML file per experiment, under `experiments/<line>/specs/`, so it dies with
its research line like everything else there (#143).

| section | holds |
|---|---|
| `[experiment]` | id, title, hypothesis, status, links to issue + finding + commit |
| `[protocol]` | task, seeds, dims, steps, the exact command, the metric, stated caveats |
| `[arms.<name>]` | `role = "control"｜"treatment"` and the config-delta that defines the arm |
| `[criteria.<name>]` | one comparison: `rule`, `treatment`, `control`, `sigmas`, `points`, `require` |
| `[verdict]` | `keep_if` / `kill_if` (lists of criterion names), and `recorded` once it has run |
| `[readouts]` | observations that explicitly carry **no** keep/kill weight |
| `[results.<point>]` | per-arm numbers, written **after** the run |

Three comparison rules, which is all the recorded experiments have ever needed:

- `within` — `|Δ| ≤ n·σ_pooled`, i.e. parity / no harm
- `beats` — `Δ ≥ n·σ_pooled`, a win
- `loses` — `Δ ≤ −n·σ_pooled`, a kill trigger

`points` names which sweep points a criterion covers; `require = "all" | "any"`
says whether every point must satisfy it or merely one.

## The conventions it pins

**σ is the sample sigma** (`statistics.stdev`, ddof=1) and
**σ_pooled = √((σ_t² + σ_c²) / 2)**. Not a new choice — the one every finding
already used. `tests/apparatus/test_verdict.py` proves it by recomputing their
published σ figures from their published per-seed numbers.

**Verdicts are three-valued.** `KEEP` and `KILL` are the pre-registered branches;
`INCONCLUSIVE` is what you get when neither fired, and it is a real answer. #77
landed there — both of its keep branches missed and it pre-registered no explicit
kill — and the record says so rather than rounding it to a kill.

**Kill beats keep.** A spec satisfying both branches is badly drafted, and the
safe reading of "the kill condition fired" is that the idea is dead.

## Results may be per-seed *or* summarized

An arm is either `[0.958, 0.981, 0.977]` or `{ mean = 0.8144, sigma = 0.0281, n = 3 }`.

The summary form exists because that is how some results genuinely are: #86's leg A
published per-depth means and a combined σ_pooled without the underlying seeds.
The alternative was reconstructing per-seed numbers that happen to reproduce the
published mean and sigma — which is inventing measurements nobody took. The format
carries the coarser truth instead of a prettier fiction.

## Pre-registration and the file's honesty

Criteria and results live in one file, which is only honest because **git records
the order**: a spec is committed with its criteria *before* its run, and the
results land in a later commit. That ordering is the evidence, not the file.

The three specs checked in so far are marked `status = "retrofit"` — they were
back-cast from experiments already completed and published, so both halves
arrived in one commit. They cannot themselves prove pre-registration; their
findings do. They exist to falsify the format (#40's instruction: *if the format
can't express what those experiments actually did, the format is wrong*), and
they earned two changes already — the summary form above, and `require`/`readouts`,
which #77 needed for its three-way branch and its no-weight observations.

## What this does not do yet

It judges; it does not run. Build 3 is the runner that consumes a spec, executes
the sweep, writes the result rows, and calls this module for the verdict. Until
then these files are a pre-registration format plus a referee that has been shown
to agree with every judgement the project has already made.
