"""The base run's referee: its spec, its yardstick journal, its verdict, its card (#294).

    python -m instruments.base_run verdict --spec experiments/base/specs/001-plain-base.toml --run runs/run_x
    python -m instruments.base_run card    --spec ... --run runs/run_x [--out docs/registry/run_x.md]

The supervisor calls `score_checkpoint` at every milestone (a LAMBADA subsample
on the CPU, beside training) and at completion (the full set on the card), and
`verdict_for` beside its BUDGET_COMPLETE line. `write_model_card` fills
docs/registry/MODEL_CARD_TEMPLATE.md from what the run recorded; a human edits
the one-line "why it is notable" and nothing else.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from instruments import verdict as referee

REPO = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = "yardstick.jsonl"

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "LAMBADA acc (completion)": ("measured", "the full 5,153-example set on the final checkpoint"),
    "LAMBADA acc (milestone)": ("sampled", "a fixed --limit subsample, scored on the CPU beside training"),
    "verdict": ("measured", "instruments.verdict over the spec's criteria and the completion yardstick"),
}


def load_base_spec(path) -> referee.Spec:
    """A base-run spec: exactly one runnable arm, one constant reference, a budget."""
    spec = referee.load_spec(path)
    arms = spec.meta.get("arms") or {}
    runnable = [n for n, b in arms.items() if not b.get("constant")]
    constant = [n for n, b in arms.items() if b.get("constant")]
    if len(runnable) != 1 or len(constant) != 1:
        raise ValueError(f"{path}: a base-run spec has exactly one runnable arm and one constant "
                         f"reference arm, found runnable={runnable} constant={constant}")
    if constant[0] not in (spec.meta.get("results") or {}).get("run", {}):
        raise ValueError(f"{path}: the reference arm {constant[0]!r} declares no [results.run] value")
    if "budget_tokens" not in (spec.meta.get("protocol") or {}):
        raise ValueError(f"{path}: [protocol] must declare budget_tokens")
    return spec


def recorded_arch(run_dir) -> str | None:
    """The MODEL_ARCH the run recorded in run_metadata.json, or None if it recorded none.

    A checkpoint has to be restored as the architecture it was trained as, and that is
    a fact about the run, not about the process scoring it: a resumed or archived run
    scores as what it was, whatever MODEL_ARCH this shell happens to have."""
    path = pathlib.Path(run_dir) / "run_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("parameters", {}).get("MODEL_ARCH")


def score_checkpoint(run_dir, checkpoint_path, *, step, limit=None, on_cpu=False, arch=None,
                     python=sys.executable) -> dict | None:
    """Run the yardstick on a checkpoint and append one line to the run's journal.

    `arch` defaults to the run's recorded MODEL_ARCH; with none recorded the yardstick
    falls back to MODEL_ARCH from config."""
    arch = arch or recorded_arch(run_dir)
    run_dir = pathlib.Path(run_dir)
    out = run_dir / f"yardstick_step{step}{'_limit' + str(limit) if limit else ''}.json"
    argv = [python, "-m", "instruments.yardstick.eval_yardstick", "--checkpoint-path", str(checkpoint_path),
            "--json-out", str(out), "--no-heldout"]
    if arch:
        argv += ["--arch", arch]
    if limit:
        argv += ["--limit", str(limit)]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if on_cpu:
        env.update({"JAX_PLATFORMS": "cpu", "FORCE_F32_COMPUTE": "1"})
    proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        (run_dir / JOURNAL).open("a").write(json.dumps({
            "step": step, "limit": limit, "error": (proc.stderr or proc.stdout)[-2000:],
            "when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}) + "\n")
        return None
    row = json.loads(out.read_text())
    entry = {"step": step, "limit": limit, "on_cpu": on_cpu, **row["lambada"],
             "when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with (run_dir / JOURNAL).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def journal(run_dir) -> list[dict]:
    path = pathlib.Path(run_dir) / JOURNAL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completion_entry(run_dir) -> dict | None:
    """The last full-set (no --limit) yardstick line, if any."""
    full = [e for e in journal(run_dir) if not e.get("limit") and "lambada_acc" in e]
    return full[-1] if full else None


def verdict_for(spec_path, run_dir) -> referee.Verdict:
    spec = load_base_spec(spec_path)
    entry = completion_entry(run_dir)
    if entry is None:
        raise ValueError(f"{run_dir}: no full-set yardstick in {JOURNAL} — score the final checkpoint first")
    arms = spec.meta["arms"]
    runnable = next(n for n, b in arms.items() if not b.get("constant"))
    metric = spec.meta["protocol"]["metric_key"]
    results = {"run": {runnable: [float(entry[metric])]}}
    from instruments.experiment import merge_constants
    results = merge_constants(spec, results, pathlib.Path(spec_path))
    return referee.evaluate(spec, results)


# --- the model card ------------------------------------------------------------

def _metrics_summary(run_dir):
    path = pathlib.Path(run_dir) / "metrics.csv"
    last_step, last_val, peak = 0, None, 0.0
    if path.exists():
        for row in csv.DictReader(path.open()):
            try:
                last_step = max(last_step, int(row["step"]))
            except (KeyError, ValueError):
                continue
            if row.get("val_ce"):
                last_val = float(row["val_ce"])
            if row.get("arena_peak_mib"):
                peak = max(peak, float(row["arena_peak_mib"]))
    return last_step, last_val, peak


def _weights_sha(checkpoints_dir) -> tuple[str, str]:
    """(step dir name, sha256 of its files in path order), or ('n/a', 'n/a')."""
    ck = pathlib.Path(checkpoints_dir)
    steps = sorted((p for p in ck.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)) if ck.is_dir() else []
    if not steps:
        return "n/a", "n/a"
    h = hashlib.sha256()
    for f in sorted(steps[-1].rglob("*")):
        if f.is_file():
            h.update(f.relative_to(steps[-1]).as_posix().encode())
            h.update(f.read_bytes())
    return steps[-1].name, h.hexdigest()


def card_fields(run_dir, spec_path=None) -> dict:
    """Everything the template asks for that the run recorded. Pure over files."""
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    params = meta.get("parameters", {})
    sections = meta.get("sections", [])
    hours = sum((s.get("duration_seconds") or 0) for s in sections) / 3600.0
    last_step, last_val, peak = _metrics_summary(run_dir)
    tokens_per_opt = (int(params.get("ACCUMULATION_STEPS", 0)) * int(params.get("BATCH_SIZE", 1))
                      * 2 * int(params.get("MAX_SEQ_LEN", 0)))
    ref = None
    if spec_path:
        spec = load_base_spec(spec_path)
        ref = spec.meta["results"]["run"]
    yard = completion_entry(run_dir)
    step_name, sha = _weights_sha(run_dir / "checkpoints")
    arch = params.get("MODEL_ARCH", "?")
    config_keys = ["LATENT_DIM", "NUM_HEADS", "PLAIN_LAYERS", "REFINER_ENCODER_LAYERS", "MAX_SEQ_LEN",
                   "MAX_STEPS_LIMIT", "BATCH_SIZE", "ACCUMULATION_STEPS", "DECAY_STEPS"]
    return {
        "run_id": meta.get("run_id", run_dir.name),
        "commit": meta.get("git_commit", "?"), "branch": meta.get("git_branch", "?"),
        "dirty": bool(meta.get("git_dirty", False)),
        "arch": arch,
        "config": ", ".join(f"{k}={params[k]}" for k in config_keys if k in params),
        "tokenizer_vocab": params.get("VOCAB_SIZE", "?"),
        "seeds": f"DATA_SEED={params.get('DATA_SEED', '?')}, MODEL_SEED={params.get('MODEL_SEED', '?')}",
        "budget": params.get("TRAIN_TOKEN_BUDGET"),
        "sections": len(sections), "hours": hours,
        "last_step": last_step, "tokens_seen": last_step * tokens_per_opt,
        "val_ce": last_val, "peak_vram_mib": peak,
        "lambada_acc": yard["lambada_acc"] if yard else None,
        "lambada_ppl": yard["lambada_ppl"] if yard else None,
        "reference": ref, "spec": str(spec_path) if spec_path else None,
        "weights_step": step_name, "weights_sha256": sha,
    }


def render_card(f: dict) -> str:
    import math
    bar = "—"
    if f["lambada_acc"] is not None and f["reference"]:
        ref = next(iter(f["reference"].values()))
        ref_acc = ref["mean"] if isinstance(ref, dict) else ref
        bar = "matches / above" if f["lambada_acc"] >= ref_acc else f"below (reference {ref_acc})"
    ppl = f"{math.exp(f['val_ce']):.1f}" if f["val_ce"] is not None else "—"
    return f"""# Model card — `{f['run_id']}`

**One line:** <fill in: why this model is worth keeping — champion / milestone / notable negative>

## Recipe (what regenerates it)

| Field | Value |
|---|---|
| Commit SHA | `{f['commit']}`{' (**git_dirty: true**)' if f['dirty'] else ''} |
| `MODEL_ARCH` | `{f['arch']}` |
| Config snapshot | `{f['config']}` |
| Tokenizer | `r50k_base` (VOCAB_SIZE `{f['tokenizer_vocab']}`) |
| Seed(s) | `{f['seeds']}` |
| Budget | `TRAIN_TOKEN_BUDGET={f['budget']}` |
| Spec | `{f['spec'] or 'none — launched before #294'}` |

## Result (how it did)

| Metric | Value | Noise floor (seed σ) |
|---|---|---|
| Held-out val CE (last probe) | **{f['val_ce'] if f['val_ce'] is not None else '—'}** — ppl {ppl} | one seed; see instruments/probe_sigma.py for the probe's own noise |
| LAMBADA last-word acc | **{f['lambada_acc'] if f['lambada_acc'] is not None else 'not scored'}** | — |
| LAMBADA ppl | **{f['lambada_ppl'] if f['lambada_ppl'] is not None else 'not scored'}** | — |
| GPT-2-small yardstick | **{bar}** | — |

## Cost

| | |
|---|---|
| Peak VRAM (arena) | {f['peak_vram_mib']:.0f} MiB of 6144 |
| Wall-clock | {f['hours']:.1f} h across {f['sections']} session(s) |
| Tokens seen | {f['tokens_seen']:,} (opt step {f['last_step']:,}) |

## Weights (the regenerable cache)

| | |
|---|---|
| Checkpoint path (live) | `runs/{f['run_id']}/checkpoints/{f['weights_step']}` |
| Archive path (HDD) | <fill in when mirrored> |
| `sha256` | `{f['weights_sha256']}` |

## Notes

Generated by `python -m instruments.base_run card` from `run_metadata.json`, `metrics.csv`,
`{JOURNAL}` and the spec. Only the one-line summary above is typed by a human.
"""


def write_model_card(run_dir, spec_path=None, out=None) -> pathlib.Path:
    fields = card_fields(run_dir, spec_path)
    out = pathlib.Path(out) if out else REPO / "docs" / "registry" / f"{fields['run_id']}.md"
    out.write_text(render_card(fields))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["verdict", "card", "score"])
    ap.add_argument("--spec", default=None)
    ap.add_argument("--run", required=True, help="the run directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    run_dir = pathlib.Path(args.run)
    if args.what == "score":
        steps = sorted(int(p.name) for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit())
        entry = score_checkpoint(run_dir, run_dir / "checkpoints", step=steps[-1], limit=args.limit, on_cpu=args.cpu)
        print(json.dumps(entry, indent=1) if entry else "yardstick failed — see the journal")
        return 0 if entry else 1
    if args.what == "verdict":
        entry = completion_entry(run_dir)
        v = verdict_for(args.spec, run_dir)
        print(f"LAMBADA acc {entry['lambada_acc']:.4f} / ppl {entry['lambada_ppl']:.1f} — verdict: {v.outcome}")
        print(v.describe())
        return 0
    print(write_model_card(run_dir, args.spec, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
