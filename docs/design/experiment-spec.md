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
| `[stages]` | two looks at a parity pair: seeds at each look, the spending function, futility (optional — see "Two looks" below, #604) |
| `[stall]` | a base run's early-stop rule: which held-out CE it reads, window, bar, decay (read by `python -m instruments.stall`, #468) |
| `[results.<point>]` | per-arm numbers, written **after** the run |

Every reader goes through `instruments/spec_file.py`, which types each table (#478).
A key the machine acts on is strict: an unknown key in `[experiment]`, `[arms.*]`,
`[criteria.*]`, `[execution]`, `[verdict]` or `[readouts]` fails the load, naming
the table and the key. `[protocol]` types the keys tools read and keeps the rest
as written, since retrofits record their own setup there.

Four comparison rules:

- `beats` — `Δ ≥ n·σ_pooled`, a win
- `loses` — `Δ ≤ −n·σ_pooled`, a kill trigger
- `equivalent` — parity / no harm *shown*: the gap's 90% confidence interval lies
  inside `±n·σ_pooled` (a TOST at one-sided p ≤ 0.05), or inside `±margin` when the
  criterion registers an absolute `margin` in the metric's own units (with
  `sigmas = 0`), as equivalence testing does and as a staged spec must. It never holds at σ_pooled 0:
  identical seeds (a quantized tokens-to-target reading) show nothing, so state a
  parity claim on such a metric with the interpolated crossing (#547)
- `within` — `|Δ| ≤ n·σ_pooled`: no difference *seen*. How the recorded parity
  claims were registered, kept so they read as registered; a new parity claim uses
  `equivalent`

Every point also reports the one-sided p of its rule's claim, from Student's two-sample t
with each arm's seed count (#599). σ_pooled is the seeds' spread, not
the error of the means, so the same 2σ is p 0.09 at two seeds a side, 0.035 at three,
0.0007 at eight. The rule decides; `python -m instruments.audit` flags a carrying
criterion whose p is above 0.05 (`holds-at-seed-count`). Parity is what costs seeds:
at a 2σ margin, three a side show it only for a gap under ~0.26σ, eight for one
under ~1.1σ.

`points` names which sweep points a criterion covers; `require = "all" | "any"`
says whether every point must satisfy it or merely one.
`min_delta` (optional, `beats`/`loses` only) adds an absolute margin in the metric's
own units on top of the sigma bar — for a claim like "15% fewer tokens", where tight
seeds would let a 3% difference clear 2σ (#26).

## The conventions it pins

**What a seed varies, on the real model.** A seed sets `MODEL_SEED` and `DATA_SEED`
together; the field comments in `trm/settings.py` give the detail.

| knob | varies with the seed | does not vary |
|---|---|---|
| `MODEL_SEED` | the initial weights | anything on a resume or a branch, where the weights come from the checkpoint |
| `DATA_SEED` | where each chunk file starts being read (an offset under 1,025 tokens, drawn per file); which bucket each row of a micro-step comes from | the order of documents, since every bucket's files are read in string-sorted order (chunk_10 before chunk_2), each front to back; the held-out probe rows (in a bucket larger than the probe's tail) |

So two seeds read nearly the same documents in the same order, and σ on a fresh
`tokens_to_ce` pair is **init variance**, not data variance (#378): a lower bound on
the real noise. A delta that clears 2σ here is not thereby shown to clear it across
data orders. A gradient event at the same opt step in every seed of every arm (#364) is
what shared data looks like. The one exception is a branch (`DATA_BRANCH`, #489,
`experiments/mix/branch_decay.py`): each seed skips its own
`DATA_SEED × DATA_BRANCH_SEED_STRIDE` rows, so the seeds read different documents from
shared weights, and there σ is data variance. A knob that varies data order on a fresh
run is open on #378.

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

**A squash merge erases the order.** Every runner-era spec on `main` (235, 238,
242, 246, 026-stage1) was pre-registered in a PR branch, run, and squash-merged —
so on `main` its criteria and its results share one commit, and git holds no
evidence of which came first. The audit reports those as grandfathered (with the
commit date) rather than red, because nothing can recover the order now. From
2026-09-15 on, pre-register the spec in **its own PR** and let the results land in
a later one, as 026-stage2 did: that survives any merge strategy.

## The audit — can the verdict be trusted?

`python -m instruments.audit` (`make audit`) runs a fixed checklist over the specs
a change can reach — the same diff `tests/affected.py` uses, so scope is never a
person's choice — and in CI on every push and PR. It never decides KEEP/KILL; it
judges whether the recorded verdict is *evidence*. One line per rule, a reason on
every red, and a spec that predates a rule is grandfathered with the date, never
rewritten:

| rule | red when |
|---|---|
| `floor-both-arms` | a criterion the verdict rests on passed with both arms at a declared floor arm or `[protocol] cap` |
| `control-reached-target` | a tokens-to-target metric (`metric_key` contains `tokens_to`) has no `[protocol] cap`, or a control seed sits at the cap and is not voided |
| `target-reached-late` | a tokens-to-target metric's control reaches the target in the first half of the cap: the bar then judges the start of the run, not the run (#361, #547). Specs whose criteria predate 2026-10-08 are grandfathered |
| `seeds-complete` | an arm a criterion compares has fewer seeds than `[execution]` (or `[protocol] seeds`) registered, and the rest are not voided. A `[stages]` spec that settled at look 1 counts against look 1's seeds |
| `sigma-plausible` | σ_pooled is 0 at a criterion point (one seed, or identical seeds); a σ over 10× the gap is *noted* on green, since an honest null lands there one time in ten |
| `prediction-registered` | the hypothesis registers no prediction (rule 3) |
| `criteria-predate-results` | the commit that introduced `[criteria]` does not precede the one that introduced measured `[results]` |
| `verdict-current` | the referee, re-run on the recorded numbers, does not reproduce `verdict.recorded` |
| `finding-cites-spec` | a KEEP/KILL spec is cited by no finding's `Spec:` line, no graveyard tombstone (path or `#issue`), and links no `finding =` of its own; in the other direction, a finding dated 2026-09-12 or later names no spec and is not `Evidence: observational — <why>` |
| `run-ended-by-budget` | a log named in `[runs.<arm>] logs = [...]` / `dirs = [...]` ends in a divergence kill or non-finite skips instead of a budget stop; n/a when the spec names no log |

Two small keys exist for the audit's sake. A seed that is missing for a stated
reason is declared rather than silently absent:

```toml
[void.run.adamw]
seeds = [1]
reason = "never reached CE 5.85 inside the cap; re-registered as 027"
```

A tokens-to-target spec sets its target from the control's own curve, near its
endpoint (~85-90% of the run), never from an estimate; read it off a control seed
before any treatment result and commit it (#494's amendment). Prefer the harness's
`tokens_to_target_interp_M`, which interpolates between the probes that bracket the
target: `tokens_to_target_M` is quantized to the probe interval, so seeds inside one
interval tie exactly and their sigma reads 0 (#547).

and a tokens-to-target metric declares its cap once, in the metric's own units,
so a control that never reached the target is visible to a machine:

```toml
[protocol]
metric_key = "tokens_to_target_M"
cap = 67.1
```

A spec with no measured results (only constant arms, or none) is reported as
*pending* and audited on nothing — a spec mid-run is not a spec that failed.

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

## Two looks — `[stages]` (#604)

A parity pair costs seeds: ~8 a side where a `beats` pair stops at 3. A spec may
register two looks instead of one fixed count, and stop at the first when it
settles:

```toml
[stages]
seeds = [4, 8]                # seeds per arm at look 1 and look 2, cumulative
spending = "obrien-fleming"   # or "pocock": the Lan–DeMets spending function
futility = true               # stop at look 1 when parity is already ruled out

[criteria.no_harm]
rule = "equivalent"
treatment = "treatment"
control = "control"
sigmas = 0.0
margin = 0.01                 # absolute, in the metric's units, fixed before the run
```

**Why each look has its own bar.** Judging at both looks with the fixed bar
inflates the false-positive rate. Lan & DeMets (1983) spend the overall one-sided
α (0.05) across the looks as a function of the information fraction, look 1's
seeds over look 2's. O'Brien–Fleming-type (the default) spends almost nothing
early, so look 2's bar stays near α: at 4 then 8 seeds, one-sided p ≤ 0.0056 at
look 1 and ≤ 0.048 at look 2. Pocock-type spends evenly: 0.031 then 0.030. The
referee turns each look's z bar into the nominal one-sided p the TOST must reach;
`tests/apparatus/test_group_sequential.py` checks the bars against the published
two-look tables and that a synthetic null keeps the error at α.

**What a look decides.** The referee judges a look on each arm's first `seeds[k]`
values (a declared constant arm is used as is). By default it replays the design:
look 1 first, and look 2 only when look 1 did not settle, so a look-1 stop stands
even when look 2's seeds were run anyway. At look 1:
- KEEP or KILL at look 1's bar: settled, stop.
- futile: an `equivalent` keep criterion shows the gap *beyond* its margin at
  look 1's bar. Settled as INCONCLUSIVE: parity can no longer be shown, and a kill
  is the spec's own branch to fire. Futility only makes parity claims rarer, so the
  error stays at or under α. `futility = false` turns it off.
- otherwise not settled: run look 2's seeds and judge again at look 2's bar.

The verdict prints the look, its seeds and its bar. Measured on a synthetic null
(σ 1, margin 2σ, 4 then 8 seeds, 10,000 pairs a row, 2026-11-08):

| spending | P(parity) at gap = margin | power at gap 0 | stopped at look 1, gap 0 | mean seeds a side, gap 0 |
|---|---|---|---|---|
| O'Brien–Fleming | 0.050 | 0.968 | 10% | 7.6 |
| Pocock | 0.049 | 0.945 | 46% | 6.2 |

O'Brien–Fleming keeps the fixed design's power and saves little; Pocock saves
about a quarter of the seeds for some power. Choose before the run.

**Scope, as the owner decided on #604.** Only `equivalent` spends α: `beats` and
`loses` are σ bars, not α-calibrated, judged as written at every look, and a
`beats` pair already stops at 3 seeds. So a staged spec needs an `equivalent` keep
criterion, every `equivalent` criterion registers an absolute `margin`, and
`[execution] seeds` holds look 2's count. Exactly two looks. Without `[stages]` a
spec is judged once, at its full seed count, exactly as before #604
(`tests/apparatus/test_verdict_pinned.py`).

The runner does not stop at look 1 yet: it runs every registered seed, and the
referee then judges at look 2, which is correct but saves nothing (#629).

## Two ways a criterion can be quietly meaningless

Both were hit on real sweeps, both by criteria that parsed, ran, and returned a
verdict nobody could act on.

**Separation is an effect size, not a t-statistic.** `instruments/verdict.py`
separates arms by `(mean_t − mean_c) / RMS(σ_t, σ_c)`. That denominator is the
*spread of the arms*, so it does **not** shrink as you add seeds, and the control's
own variance is in it. A claim of the form "the control is erratic" is therefore
unmeetable when phrased as a per-depth mean comparison — the more erratic the
control, the harder it is to beat. #235 read +0.84σ at depth 8 while the control
held a seed scoring 0.7893 there, below its own depth 2.

The fix is to collapse the claim into **one number per run** and compare *those*
means. #238's `min_deep` (worst accuracy at any depth ≥ 2) does this: a run whose
recurrence diverged scores low however well its depth-1 arm did.

**σ-phrased criteria assume the arms are comparable in magnitude.** #242 compared a
checkpoint whose depth gain was −6.25 nats against one at −0.0001, with a sampling
sigma of ~0.0004, and reported "+124σ" and "+283σ". Those numbers are arithmetic,
not evidence. When arms can differ by orders of magnitude, say so in `[protocol]
notes` and read the raw table — a separation in the hundreds means the sigma frame
has stopped applying, not that the result is overwhelming.

## Constant arms

An arm marked `constant = true` is never run; its values are declared in
`[results]` as an inline summary. This exists because the referee can otherwise only
express *relative* comparisons, and a parity criterion needs an absolute floor
beside it — two arms that both fail a task are trivially "within 2σ" of each other.

Declare the floor by **measuring the task generator**, not by assuming uniformity
over the vocabulary. #246 build 1 assumed 1/7 = 0.143 where the majority-class rate
was 0.34, and both arms sat exactly at a majority-class strategy while the criterion
called it parity.

Constants are merged into the results before *both* judging and recording, so a
declared floor survives its own experiment and appears in the finding's table as
`declared, not run`.
