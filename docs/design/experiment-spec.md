# The experiment spec — a pre-registration a machine can apply

**Status:** builds 1 and 3 of #40 landed — the spec is a file, `instruments/verdict.py`
judges it, and `instruments/experiment.py` runs it. Build 4 (the safety layer) is next.

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
| `[execution]` | how to actually run it (optional — see below) |
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

## Running one — `[execution]` and the runner

A spec with an `[execution]` section can be run, not just judged:

```bash
python -m instruments.experiment experiments/depth/specs/NNN-thing.toml
```

The runner gates on `tests/core`, sweeps, journals every measurement as it
arrives, appends the `[results.*]` tables to the spec, hands the whole thing to
`verdict.py`, and writes a findings draft into the (gitignored) run folder. It
never writes to `docs/findings/`: producing evidence is a machine's job,
publishing a finding is not.

```toml
[protocol]
metric_key = "acc"          # which number in the harness's RESULT line is judged

[execution]
command = ["python", "-m", "experiments.depth.ablation_harness", "--task", "statetrack"]
seed_flag = "--seed"
seeds = [0, 1, 2]
env = { JAX_PLATFORMS = "cpu" }

[arms.islands]
flags = ["--per-pass-loss", "--islands"]
```

One subprocess per (arm, seed): `command` + the arm's `flags` + the seed. The
runner understands no harness's knobs — arms name their own flags — which is
what keeps `instruments/` from acquiring a dependency on a research line that
rule 6 will one day delete.

**Legs**, because real experiments here have more than one. #86 trained at depths
1–8 and then ran a *second* command (depth 8, longer eval sequence, extra eval
depths) to measure extrapolation past the cap; #77 ran three seeds and then
extended one arm to six under a fresh pre-registration. A leg carries its own
flags, and may override the seeds and restrict the arms:

```toml
[execution.legs.extrapolation]
flags = ["--depths", "8", "--test-seq", "48", "--eval-depths", "12,16"]
arms = ["sinusoidal", "table"]
```

Named legs namespace their points (`extrapolation/d8`), so two legs measuring
depth 8 under different conditions cannot silently overwrite each other. The
retrofits falsified the runner exactly as they falsified the format: a
single-command runner could have run neither #77 nor #86.

**How numbers come back.** Harnesses print a human table *and* one machine line
per measurement (`instruments/results.py`):

    RESULT {"point": "d8", "acc": 0.7123, "ce": 0.8310}

The harness reports what it measured and which sweep point it was; the runner
supplies the arm and the seed, because the runner is what launched the process. A
harness that finished silently is an error, not an empty result.

**Resume.** Every measurement is journalled to `runs/experiments/<id>/results.jsonl`
as it lands, keyed by a fingerprint of the exact command and environment. A sweep
that dies at seed 4 of 6 continues; a sweep whose flags changed re-runs.

The three retrofit specs deliberately carry **no** `[execution]` block. Their
point names (`d12`, `statetrack_6seed`) come from the prose of the findings they
were transcribed from, not from what a harness emits today, and inventing an
`[execution]` whose rerun would produce differently-named points would be a spec
that only looks reproducible. They are records; new specs are jobs.
