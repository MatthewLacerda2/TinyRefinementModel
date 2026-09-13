"""The runner: a spec goes in; runs happen; a verdict and a findings draft come out.

Build 1 of #40 made the pre-registration a file and `instruments/verdict.py` the
referee. This is the other half — the thing that actually executes the sweep the
spec describes, so the numbers the referee judges are the numbers the spec asked
for, and nobody hand-copies a table into a criterion.

    python -m instruments.experiment experiments/depth/specs/077-per-pass-supervision.toml

What it does, in order:

1. **Gate.** Run the fast test tier first. A sweep launched on a broken tree
   burns hours and produces numbers nobody should trust. `--no-gate` skips it.
2. **Sweep.** One subprocess per (arm, seed), built from the spec's
   `[execution] command` plus that arm's `flags` plus the seed. Every
   measurement the harness emits (see `instruments/results.py`) is appended to a
   journal as it arrives, so a sweep that dies at seed 4 of 6 resumes instead of
   restarting.
3. **Record.** Collect the journal into `[results.<point>]` tables and append
   them to the spec file — the same file the criteria live in, which stays
   honest because git records that the criteria commit came first.
4. **Judge.** Hand spec + results to `verdict.py` and print what it says.
5. **Draft.** Write a findings document with the numbers, the separations, and
   the verdict already filled in, into the run folder (gitignored). A human
   decides whether it belongs in `docs/findings/`; the runner never publishes.

The runner never imports a research line. It only knows how to launch a command
and read the RESULT protocol, which is what keeps `instruments/` from dying with
whichever `experiments/` folder gets tombstoned next.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass

from instruments import results as result_lines
from instruments.verdict import (
    Spec, evaluate, load_recorded_results, load_spec, mean_sigma,
)

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # runs harnesses and records their RESULT lines; the numbers belong to the harness and verdict.py

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "experiments"
GATE_COMMAND = (sys.executable, "-m", "pytest", "tests/core", "-q", "-x")


@dataclass(frozen=True)
class Leg:
    """One pass of the sweep: a set of flags, the arms it runs, the seeds it uses.

    Legs exist because real experiments here have more than one. #86 trained at
    depths 1–8 and then ran a *second* command — depth 8, a longer eval sequence,
    extra eval depths — to measure extrapolation past the cap. #77 ran three
    seeds, then pre-registered an extension to six for one arm only. A runner
    that could express neither would have been a runner for experiments we do not
    do, so the same retrofit that falsified the spec format falsified this too.

    An unnamed leg is the common case: one command, one seed list, every arm.
    """

    name: str
    flags: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]


@dataclass(frozen=True)
class Execution:
    """How to turn a leg, an arm and a seed into a command line."""

    command: tuple[str, ...]
    seed_flag: str
    env: dict[str, str]
    arm_flags: dict[str, tuple[str, ...]]
    legs: tuple[Leg, ...]
    metric: str

    def argv(self, leg: Leg, arm: str, seed: int) -> tuple[str, ...]:
        return (*self.command, *leg.flags, *self.arm_flags[arm], self.seed_flag, str(seed))

    def point(self, leg: Leg, emitted: str) -> str:
        """Namespace a harness's point name by leg, so two legs measuring "d8"
        under different conditions do not silently overwrite each other."""
        return f"{leg.name}/{emitted}" if leg.name else emitted

    def fingerprint(self, leg: Leg, arm: str, seed: int) -> str:
        """Identifies a run by everything that could change its number, so a
        resumed sweep reuses a row only if it came from the same command."""
        payload = json.dumps(
            {"argv": self.argv(leg, arm, seed), "env": self.env}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def runs(self):
        for leg in self.legs:
            for arm in leg.arms:
                for seed in leg.seeds:
                    yield leg, arm, seed


def load_execution(spec: Spec) -> Execution:
    """Read the `[execution]`, `[arms.*]` and `[protocol]` keys the runner needs.

    Validated up front and loudly: a spec whose command is missing must fail
    before the gate runs, not two hours into a sweep.
    """
    ex = spec.meta.get("execution")
    if not ex:
        raise ValueError(
            f"spec {spec.id} has no [execution] section — it can be judged but not run")
    if "command" not in ex:
        raise ValueError(f"spec {spec.id}: [execution] is missing 'command'")

    if isinstance(ex["command"], str):
        # A string is iterable, so this used to become one argument per CHARACTER
        # and the sweep launched `p`, `y`, `t`, ... — 103 arguments for a command
        # nobody could read in the failure. Reject it where the mistake is made.
        raise ValueError(
            f"spec {spec.id}: [execution] command must be a LIST of arguments, not a "
            f'string. Write ["python", "-m", "pkg.mod", "--flag", "value"], because a '
            f"string here would be iterated character by character.")
    command = tuple(str(part) for part in ex["command"])
    if not command:
        raise ValueError(f"spec {spec.id}: [execution] command is empty")
    if command[0] in ("python", "python3"):
        # Run the sweep under the SAME interpreter as the runner, which is what
        # GATE_COMMAND already does. A bare "python" resolves against PATH — the
        # system interpreter, with no jax — so a spec would pass its gate and then
        # every arm would die on ModuleNotFoundError. The alternative is baking a
        # venv path into a committed spec, which is worse.
        command = (sys.executable, *command[1:])

    all_arms = spec.meta.get("arms", {})
    # An arm marked `constant = true` is NOT run: its values are supplied in the
    # spec's [results] as a Summary. This exists because the referee can otherwise
    # only express RELATIVE comparisons, and some questions need an absolute floor --
    # two arms that both fail a task are trivially "within 2 sigma" of each other,
    # and without a floor that reads as parity rather than as a null result. Declaring
    # it explicitly (rather than inferring "no flags means constant") keeps the
    # typo check below honest: a misspelled arm name still fails loudly.
    arm_flags = {name: tuple(str(f) for f in body.get("flags", ()))
                 for name, body in all_arms.items() if not body.get("constant")}
    if not arm_flags:
        raise ValueError(f"spec {spec.id}: no runnable [arms.*] — every arm is constant")

    named = {c.treatment for c in spec.criteria.values()} | {c.control for c in spec.criteria.values()}
    missing = sorted(named - set(all_arms))
    if missing:
        raise ValueError(
            f"spec {spec.id}: criteria compare arms that no [arms.*] section defines: {missing}")

    metric = spec.meta.get("protocol", {}).get("metric_key")
    if not metric:
        raise ValueError(
            f"spec {spec.id}: [protocol] needs metric_key — which number in the "
            f"harness's RESULT line the criteria are about")

    default_seeds = tuple(int(s) for s in ex.get("seeds", ()))
    raw_legs = ex.get("legs") or {"": {}}
    legs = []
    for name, body in raw_legs.items():
        seeds = tuple(int(s) for s in body.get("seeds", default_seeds))
        if not seeds:
            raise ValueError(
                f"spec {spec.id}: leg {name or '(unnamed)'!r} has no seeds, and "
                f"[execution] declares no default 'seeds'")
        arms = tuple(body.get("arms", tuple(arm_flags)))
        unknown = sorted(set(arms) - set(arm_flags))
        if unknown:
            raise ValueError(f"spec {spec.id}: leg {name!r} names undefined arms: {unknown}")
        legs.append(Leg(name=str(name), flags=tuple(str(f) for f in body.get("flags", ())),
                        seeds=seeds, arms=arms))

    return Execution(
        command=command,
        seed_flag=str(ex.get("seed_flag", "--seed")),
        env={str(k): str(v) for k, v in ex.get("env", {}).items()},
        arm_flags=arm_flags,
        legs=tuple(legs),
        metric=str(metric),
    )


# --- the journal --------------------------------------------------------------

def journal_path(spec_id: str) -> pathlib.Path:
    return RUNS_DIR / spec_id / "results.jsonl"


def read_journal(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_journal(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


# --- running ------------------------------------------------------------------

def run_gate(cwd: pathlib.Path) -> None:
    """The fast tier, before anything expensive. Failing here aborts the sweep."""
    print(f"gate: {' '.join(GATE_COMMAND)}", flush=True)
    proc = subprocess.run(GATE_COMMAND, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(
            f"gate failed (exit {proc.returncode}) — fix the tree before spending a sweep on it")


def run_one(execution: Execution, leg: Leg, arm: str, seed: int,
            cwd: pathlib.Path) -> list[dict]:
    """Launch one (leg, arm, seed) and return its measurements as journal rows.

    Harness stdout is streamed through to the terminal as well as captured: a
    sweep is long, and a runner that swallows its output leaves you watching a
    blank screen for an hour wondering whether it hung.
    """
    argv = execution.argv(leg, arm, seed)
    env = {**os.environ, **execution.env, "PYTHONPATH": str(cwd)}
    print(f"\n$ {' '.join(argv)}", flush=True)

    started = time.time()
    captured: list[str] = []
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        captured.append(line)
        sys.stdout.write(line)
    proc.wait()
    elapsed = time.time() - started

    if proc.returncode != 0:
        raise RuntimeError(f"{arm} seed {seed} exited {proc.returncode}")

    measurements = result_lines.parse("".join(captured))
    if not measurements:
        raise RuntimeError(
            f"{arm} seed {seed} finished but emitted no RESULT line — the harness "
            f"is not reporting machine-readably (see instruments/results.py)")

    fingerprint = execution.fingerprint(leg, arm, seed)
    return [{**m, "arm": arm, "seed": seed, "fingerprint": fingerprint,
             "elapsed_s": round(elapsed, 1),
             "point": execution.point(leg, m["point"])}
            for m in measurements]


def sweep(spec: Spec, execution: Execution, *, cwd: pathlib.Path = REPO_ROOT,
          resume: bool = True) -> list[dict]:
    """Run everything the spec asks for, skipping what the journal already has."""
    path = journal_path(spec.id)
    done = read_journal(path) if resume else []
    already = {(r["arm"], r["seed"], r["fingerprint"]) for r in done}

    rows = list(done)
    for leg, arm, seed in execution.runs():
        key = (arm, seed, execution.fingerprint(leg, arm, seed))
        if key in already:
            print(f"skip: {leg.name or 'run'} {arm} seed {seed} (already in journal)", flush=True)
            continue
        fresh = run_one(execution, leg, arm, seed, cwd)
        append_journal(path, fresh)
        rows.extend(fresh)
    return rows


def collect(rows: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    """Journal rows -> {point: {arm: [value per seed]}}, the shape verdict wants.

    Sorted by seed so the recorded list order is reproducible, and so two runs of
    the same sweep write byte-identical result tables.
    """
    grouped: dict[str, dict[str, dict[int, float]]] = {}
    for row in rows:
        if metric not in row:
            raise KeyError(
                f"{row['arm']} seed {row['seed']} at point {row['point']}: no {metric!r} "
                f"in its RESULT line (has {sorted(k for k in row if k not in ('arm', 'seed', 'fingerprint', 'elapsed_s', 'point'))})")
        grouped.setdefault(row["point"], {}).setdefault(row["arm"], {})[row["seed"]] = float(row[metric])
    return {point: {arm: [v for _, v in sorted(by_seed.items())] for arm, by_seed in sorted(arms.items())}
            for point, arms in sorted(grouped.items())}


# --- recording ----------------------------------------------------------------

def _toml_key(name: str) -> str:
    """Leg-namespaced points contain '/', which is not a bare TOML key."""
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else json.dumps(name)


def render_results_toml(results, metric: str) -> str:
    """Render `[results.*]` tables.

    An arm's value is either a list of per-seed numbers or a Summary mapping
    ({mean, sigma, n}) — the latter being how a `constant = true` floor declares
    itself, and how #86-style retrofits record published summaries.
    """
    lines = ["", f"# --- Results (per seed, written by instruments/experiment.py; metric = {metric}) ---"]
    for point, arms in results.items():
        lines.append(f"[results.{_toml_key(point)}]")
        for arm, values in arms.items():
            summary = values if isinstance(values, dict) else (
                {"mean": values.mean, "sigma": values.sigma, "n": values.n}
                if hasattr(values, "mean") else None)
            if summary is not None:
                body = ", ".join(f"{k} = {v}" for k, v in summary.items())
                lines.append(f"{arm} = {{ {body} }}")
            else:
                lines.append(f"{arm} = [{', '.join(f'{v:.4f}' for v in values)}]")
        lines.append("")
    return "\n".join(lines)


def merge_constants(spec, results, spec_path):
    """Fold a spec's `constant = true` arms into a sweep's measured results.

    Constants are declared in `[results]` and never run, so they are absent from
    whatever the sweep produced. Both the referee AND the file need them: without
    this, `record_results` wrote a second `[results."<point>"]` table for the same
    point (TOML keeps the last, so the declared floor vanished) and `evaluate` was
    handed results with no floor arm at all, failing with "point has no arm
    'chance'" only after every run had completed.
    """
    constants = {name for name, body in (spec.meta.get("arms") or {}).items()
                 if body.get("constant")}
    if not constants:
        return results
    # Parsed, not raw: verdict.py needs a Summary object, and the raw TOML table
    # is a plain dict that mean_sigma would iterate as strings.
    declared = load_recorded_results(spec_path)
    merged = {point: dict(arms) for point, arms in results.items()}
    for point, arms in declared.items():
        for arm, value in arms.items():
            if arm in constants:
                merged.setdefault(point, {})[arm] = value
    return merged


def record_results(spec_path: pathlib.Path, results, metric: str, *, force: bool = False) -> bool:
    """Append the `[results.*]` tables to the spec file.

    Refuses to touch a spec that already carries MEASURED results unless forced:
    silently rewriting a recorded number is how a pre-registration stops meaning
    anything.

    A `constant = true` arm is the exception. Its values live in `[results]` by
    design (that is how a floor is declared without running it), so their presence
    must not look like a completed sweep — otherwise declaring a floor silently
    prevents the sweep from ever recording, which is exactly what happened on #246:
    18 runs completed and were then dropped on the floor.
    """
    text = spec_path.read_text()
    constants = {name for name, body in (load_spec(spec_path).meta.get("arms") or {}).items()
                 if body.get("constant")}
    measured = re.findall(r"^\s*([A-Za-z_][\w-]*)\s*=", text[text.index("[results."):], re.M) \
        if "[results." in text else []
    if measured and set(measured) - constants and not force:
        print(f"\nspec already carries measured results — not overwriting {spec_path} (use --force)")
        return False

    if constants and "[results." in text:
        results = merge_constants(load_spec(spec_path), results, spec_path)
        text = text[:text.index("[results.")].rstrip("\n")

    spec_path.write_text(text.rstrip("\n") + "\n" + render_results_toml(results, metric))
    print(f"\nrecorded results into {spec_path}")
    return True


def draft_finding(spec: Spec, execution: Execution, results, verdict, today: str) -> str:
    """A findings document with the numbers already in it, for a human to edit.

    Written to the run folder, never to `docs/findings/`. The runner produces
    evidence; publishing a finding is a judgement, and rule 5 gives that to the
    person, not the referee.
    """
    lines = [
        f"# DRAFT — {spec.title}",
        "",
        f"**Verdict: {verdict.outcome}** — {verdict.reason}",
        "",
        f"Spec `{spec.id}` · drafted {today}",
        "",
        "> Generated by `instruments/experiment.py` from the pre-registered spec. The",
        "> numbers and the verdict are the runner's; the prose, the interpretation, and",
        "> the decision to publish are not. Delete this block when you have read it.",
        "",
        "## Hypothesis (pre-registered)",
        "",
        spec.hypothesis.strip(),
        "",
        "## Protocol",
        "",
        f"- command: `{' '.join(execution.command)}`",
        f"- metric: {spec.meta.get('protocol', {}).get('metric', execution.metric)}",
        "",
        "| leg | flags | arms | seeds |",
        "|---|---|---|---|",
    ]
    lines += [f"| {leg.name or '(single)'} | `{' '.join(leg.flags) or '(none)'}` | "
              f"{', '.join(leg.arms)} | {list(leg.seeds)} |" for leg in execution.legs]
    lines += ["", "| arm | flags |", "|---|---|"]
    lines += [f"| {arm} | `{' '.join(flags) or '(none)'}` |"
              for arm, flags in execution.arm_flags.items()]

    lines += ["", "## Results", ""]
    for point, arms in results.items():
        lines += [f"### {point}", "", "| arm | mean | sigma | n | per seed |", "|---|---|---|---|---|"]
        for arm, values in arms.items():
            mean, sigma = mean_sigma(values)
            # A constant arm (a declared floor) or a retrofit summary has no per-seed
            # values to list — it is a Summary, not a sequence.
            if hasattr(values, "mean"):
                n, per_seed = values.n, "declared, not run"
            else:
                n, per_seed = len(values), ", ".join(f"{v:.4f}" for v in values)
            lines.append(f"| {arm} | {mean:.4f} | {sigma:.4f} | {n} | {per_seed} |")
        lines.append("")

    lines += ["## Criteria", "", "| criterion | rule | comparison | separation | met |", "|---|---|---|---|---|"]
    for c in verdict.criteria:
        crit = c.criterion
        detail = ", ".join(f"{p.point} {p.sigmas:+.2f}σ" for p in c.points)
        lines.append(f"| {crit.name} | {crit.rule} {crit.sigmas}σ ({crit.require}) | "
                     f"{crit.treatment} vs {crit.control} | {detail} | {'yes' if c.holds else 'NO'} |")

    if spec.readouts:
        lines += ["", "## Readouts (no keep/kill weight)", ""]
        lines += [f"- {r}" for r in spec.readouts]

    lines += ["", "## What it means", "", "_To be written by a human._", ""]
    return "\n".join(lines)


# --- entry point --------------------------------------------------------------

SCAFFOLD = """\
# PRE-REGISTERED. Commit this file BEFORE running it: git recording that the
# criteria preceded the numbers is the only thing that makes them criteria.

[experiment]
id = "{id}"
status = "open"
title = "TODO one line: the question, not the answer"
hypothesis = \"\"\"
TODO. What you think is true, why, and what would have to be seen for it to be
false. Register the PREDICTION here as well as the reasoning -- the criterion tests
the idea, the prediction tests the person holding it, and only the second exposes a
pattern of being wrong in one direction.
\"\"\"
issue = 0

[protocol]
metric_key = "acc"
metric = "TODO what the number means, including its units"
matched = "same seed => same init, same data order; only the one knob differs"
floor_note = \"\"\"
TODO, or delete this and declare a floor arm below. Two arms that both FAIL are
trivially "within 2 sigma" of each other, so a parity criterion needs an absolute
floor beside it -- MEASURED from the task generator, never assumed from the
vocabulary (#246 build 1 assumed 1/7 where the majority class was 0.34).
Delete this note if the metric is a difference whose null is genuinely zero.
\"\"\"

[execution]
# A LIST, not a string. A string is iterable, so it becomes one argument per
# CHARACTER and the sweep launches 103 single-character arguments.
command = ["python", "-m", "experiments.TODO.harness", "--steps", "2500"]
seeds = [0, 1, 2]
seed_flag = "--seed"

[arms.control]
role = "control"
flags = []

[arms.treatment]
role = "treatment"
flags = ["--TODO"]

# An absolute floor, never run; its value is declared here. Measure it by sampling
# the task generator. Delete this arm only if floor_note explains why.
# [arms.chance]
# role = "floor"
# constant = true
#
# [results.TODO_point]
# chance = {{ mean = 0.0, sigma = 0.0001, n = 3 }}

[criteria.treatment_wins]
rule = "beats"
treatment = "treatment"
control = "control"
sigmas = 2.0

[criteria.treatment_loses]
rule = "loses"
treatment = "treatment"
control = "control"
sigmas = 2.0

[verdict]
keep_if = ["treatment_wins"]
kill_if = ["treatment_loses"]
"""


def write_scaffold(path: pathlib.Path) -> int:
    """A valid skeleton with the traps pre-avoided.

    Both of the runner's blocking defects were things a first-time spec author would
    type: `command` as a string (iterated per character) and a bare `python` (the
    system interpreter, no jax). Neither is discoverable from the error. The scaffold
    is where that knowledge lives, rather than in whoever happens to remember.
    """
    if path.exists():
        raise SystemExit(f"{path} already exists — refusing to overwrite a spec")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SCAFFOLD.format(id=path.stem.split("-")[0]))
    print(f"wrote {path}\n\nFill every TODO, then COMMIT IT BEFORE RUNNING. "
          f"tests/apparatus/test_spec_floor_lint.py refuses a spec that still "
          f"carries one.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=pathlib.Path, help="path to an experiment spec TOML")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands the sweep would run, and stop")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the tests/core gate (use when you have just run it)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the journal and re-run every (arm, seed)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite results already recorded in the spec file")
    ap.add_argument("--new", action="store_true",
                    help="write a pre-registration skeleton at SPEC and stop, with the "
                         "runner's two blocking traps already avoided (command as a "
                         "list, not a string; the interpreter question settled)")
    args = ap.parse_args(argv)

    if args.new:
        return write_scaffold(args.spec)

    spec = load_spec(args.spec)
    execution = load_execution(spec)

    planned = list(execution.runs())
    print(f"== {spec.id}: {spec.title} ==")
    print(f"{len(planned)} runs across {len(execution.legs)} leg(s)")

    if args.dry_run:
        for leg, arm, seed in planned:
            print(f"$ {' '.join(execution.argv(leg, arm, seed))}")
        return 0

    if not args.no_gate:
        run_gate(REPO_ROOT)

    rows = sweep(spec, execution, resume=not args.no_resume)
    results = merge_constants(spec, collect(rows, execution.metric), args.spec)
    record_results(args.spec, results, execution.metric, force=args.force)

    verdict = evaluate(spec, results)
    print()
    print(verdict.describe())

    today = datetime.date.today().isoformat()
    draft = RUNS_DIR / spec.id / "finding-draft.md"
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_text(draft_finding(spec, execution, results, verdict, today))
    print(f"\nfindings draft: {draft}")
    if spec.recorded and spec.recorded != verdict.outcome:
        print(f"NOTE: the spec records {spec.recorded}; this run reached {verdict.outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
