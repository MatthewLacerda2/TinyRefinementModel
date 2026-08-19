"""Fixed-prompt transcripts per checkpoint — the logbook, not the referee (#203).

Same prompts, same seed, same depth ladder, every time. That makes two checkpoints
of the same run comparable by eye, and eventually two finished models comparable to
each other. What it is *not* is evidence: generated text cannot clear a noise floor
and cannot attribute cause, so nothing here decides KEEP/KILL. That stays with
`instruments/verdict.py`, the yardstick, and `experiments/depth/`. This file exists
so a future session can *see* what a checkpoint sounded like, with enough recorded
context to know what it was looking at.

Three defects in the previous version, all of which made old transcripts unusable:

1. **Not reproducible.** `generate_text` seeded its RNG from the wall clock, so the
   same checkpoint on the same prompts gave different text every run. The docstring
   claimed comparability the tool did not have. Fixed with an explicit seed, held
   constant across depths so depth is the only variable that moves.

2. **It would have killed the training run.** The module set
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` at import — ~3GB of a 6GB card, while the
   #157 base run holds 5.2GB. Device is now explicit, defaults to CPU, and the GPU
   path refuses to start next to a busy card.

3. **It recorded almost nothing.** Step, and a timestamp. Not the token count, the
   CE at the time, the depth, the device, or either of the two commits that matter
   (the one that trained the weights and the one generating the transcript).

And a fourth, subtler one: it hardcoded `temperature=0.5`, which #192 identified as
*below* the useful range — cold enough to make an undertrained model loop on its own
("the ratio of the ratio of the ratio"). Transcripts taken at 0.5 were reading the
decoder's failure mode, not the weights'. The default now follows
`trm.infer.DEFAULT_TEMPERATURE` so the logbook and the CLI show the same model.

Usage — safe to run while training, as long as you leave --device alone:

    PYTHONPATH=. python -m instruments.dump_transcripts
    PYTHONPATH=. python -m instruments.dump_transcripts --depths 4,8
    PYTHONPATH=. python -m instruments.dump_transcripts --device gpu   # card must be free
"""

import argparse
import datetime
import os
import subprocess

# Everything that reaches JAX is imported inside main(), *after* the device choice
# has been written into the environment. Hoisting them would pin the backend before
# --device is read, which is the whole hazard this module used to carry.

PROMPT_SET_VERSION = 1

# Frozen. Changing or removing a prompt bumps PROMPT_SET_VERSION; appending does
# not, because every file records the prompts that actually ran. Prompts 1-4 probe
# prose, 5-8 cover the code/math/procedure/narrative ground the first four miss —
# code and math are ~65% of the current curriculum and would otherwise go unlogged.
PROMPTS = [
    "the capital of france is",                    # factual recall, one right answer
    "the american flag is red, white and",         # loop detector: enumeration invites repetition
    "know the truth and the truth will set you",   # memorised common text
    "The Roman Empire was one of the largest",     # multi-sentence explanation
    "def fibonacci(n):",                           # code
    "2 + 2 = 4. 3 + 5 =",                          # arithmetic
    "To make bread, you first need to",            # ordered procedure
    "Once upon a time, in a small village,",       # long-range coherence, no factual anchor
]

# Doubling ladder across the trained range: 1 is the degenerate no-refinement
# baseline, 8 is MAX_STEPS_LIMIT. Deliberately spans rather than picks, because
# depth is a runtime dial on this architecture — the same weights serve at any of
# these, and which one is *best* is an open question this logbook is built to watch.
DEFAULT_DEPTHS = (1, 2, 4, 8)

DEFAULT_SEED = 42
REPETITION_NGRAM = 4

# A card holding more than this is presumed busy with a training run.
GPU_BUSY_MIB = 1024


def repetition_score(token_ids, n=REPETITION_NGRAM):
    """Fraction of n-grams that already appeared earlier in the same completion.

    The failure mode this whole logbook watches is looping, and by eye that is a
    vibe. This turns it into one deterministic float per completion, per depth —
    which is what makes "depth 8 sometimes overthinks" checkable rather than
    remembered. Measured on token ids, not words, so it is tokenizer-exact.

    0.0 means nothing repeated; a hard loop approaches 1.0.
    """
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    seen, repeats = set(), 0
    for gram in grams:
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / len(grams)


def parse_depths(text):
    """'1,2,4,8' -> (1, 2, 4, 8). Ordered, de-duplicated, positive."""
    depths = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        depth = int(piece)
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if depth not in depths:
            depths.append(depth)
    if not depths:
        raise ValueError("no depths given")
    return tuple(depths)


def opt_step_from_checkpoint(ckpt_step, accumulation_steps):
    """Orbax names checkpoints by samples consumed; the run talks in opt steps.

    The trainer resumes at `ckpt_step + 1`, so checkpoint 1458175 is opt step
    11392 and not 11391 — an off-by-one here would mislabel every entry in the
    logbook by one checkpoint interval.
    """
    return (ckpt_step + 1) // accumulation_steps


def nearest_metric(steps, values, target):
    """The last recorded value at or before `target`, or None.

    val_ce is written every 64 opt steps, so a checkpoint rarely lands on a row
    that has one. Falling back to the most recent earlier reading is honest; the
    alternative is leaving the field blank on almost every entry.
    """
    best = None
    for step, value in zip(steps, values):
        if step <= target and (best is None or step > best[0]):
            best = (step, value)
    return None if best is None else best[1]


def render_frontmatter(fields):
    """A YAML block. Scalars and flat lists only — that is all this needs.

    Frontmatter rather than JSON because the metadata has to be machine-readable
    (comparing ten of these should not mean regexing English prose) while the
    completions have to stay literal. In JSON the generated text becomes one line
    of \\n-escaped soup, which defeats the point of a file a human ever opens.

    Keys with a None value are dropped: a foreign or finished model has no
    metrics.csv beside it, and a blank field should be absent, never zero.
    """
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def transcript_filename(opt_step, device):
    """Device is in the name because it is a comparability class, not a detail.

    CPU runs f32 and GPU runs f16, so the same weights and the same seed diverge a
    few tokens in. Two series, and they must never be silently mixed on a plot.
    """
    return f"step_{opt_step:06d}_{device}.md"


def git_head():
    """The commit generating this transcript — distinct from the one that trained
    the weights, and after a few PRs they are nowhere near each other."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_memory_used_mib():
    """Used VRAM, or None if the card cannot be queried."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def select_device(device, force=False):
    """Write the backend choice into the environment before JAX is imported.

    Refuses the GPU when the card is already busy. The previous version of this
    module grabbed half the card at import time with no check at all, which would
    have OOM'd the run it was meant to observe.
    """
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        # CPU XLA cannot lower f16 matmuls; the tests use the same escape hatch.
        os.environ.setdefault("FORCE_F32_COMPUTE", "1")
        return
    used = gpu_memory_used_mib()
    if used is not None and used > GPU_BUSY_MIB and not force:
        raise SystemExit(
            f"❌ The GPU is holding {used} MiB — a training run almost certainly owns it.\n"
            f"   Transcripts on a busy card will OOM the run. Use --device cpu (a separate,\n"
            f"   equally valid series), wait for the card, or --force if you are certain.")
    # Modest arena: this shares the card with nothing by the time it gets here, but
    # a serving-sized fraction leaves room if something else appears.
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="fixed-prompt transcript logbook")
    parser.add_argument("--checkpoint-path", default=None,
                        help="checkpoint dir (default: the latest checkpointed run)")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="cpu (default) is safe beside a training run; gpu is the "
                             "canonical series but needs a free card")
    parser.add_argument("--force", action="store_true",
                        help="run on the GPU even if it looks busy")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                        help=f"depth ladder (default {','.join(str(d) for d in DEFAULT_DEPTHS)}); "
                             "every prompt runs at every depth")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed, held constant across depths (default {DEFAULT_SEED})")
    parser.add_argument("--temperature", type=float, default=None,
                        help="default: trm.infer.DEFAULT_TEMPERATURE")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", action="append", default=[],
                        help="extra prompt; recorded but marked non-standard so it "
                             "cannot pollute a comparison")
    parser.add_argument("--out", default=None,
                        help="output directory (default: runs/<run>/transcripts)")
    return parser


def main():
    args = build_arg_parser().parse_args()
    depths = parse_depths(args.depths)
    select_device(args.device, args.force)

    import tiktoken
    from trm.config import ACCUMULATION_STEPS, MODEL_ARCH, TOKENIZER_NAME, TOKENS_PER_OPT_STEP
    from trm.infer import DEFAULT_TEMPERATURE, generate_text
    from trm.runtime.checkpoints import discover_latest_checkpoint_run
    from trm.runtime.restore import restore_model

    temperature = DEFAULT_TEMPERATURE if args.temperature is None else args.temperature
    prompts = list(PROMPTS) + list(args.prompt)
    standard = len(PROMPTS)

    run_dir = None
    if args.checkpoint_path is None:
        discovered, run_id = discover_latest_checkpoint_run()
        if discovered is not None:
            run_dir = os.path.join("runs", run_id)

    model, ckpt_step = restore_model(args.checkpoint_path)
    opt_step = opt_step_from_checkpoint(ckpt_step, ACCUMULATION_STEPS)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    train_ce = val_ce = model_commit = model_dirty = None
    if run_dir is not None:
        try:
            from instruments.runlog import load
            log = load(run_dir)
            train_ce = nearest_metric(*log.column("ce"), opt_step)
            val_ce = nearest_metric(*log.column("val_ce"), opt_step)
            model_commit = log.metadata.get("git_commit")
            model_dirty = log.metadata.get("git_dirty")
        except (FileNotFoundError, ImportError, KeyError):
            pass
    if model_commit:
        model_commit = model_commit[:7]

    results = []
    print(f"▶ {len(prompts)} prompts x {len(depths)} depths = "
          f"{len(prompts) * len(depths)} completions, seed {args.seed}, on {args.device}")
    for depth in depths:
        for index, prompt in enumerate(prompts):
            started = datetime.datetime.now()
            # Same seed at every depth: depth is the only variable that moves, so
            # a difference between two completions is the depth and not the dice.
            tokens = generate_text(model, enc, prompt, max_new_tokens=args.max_new_tokens,
                                   temperature=temperature, depth=depth, seed=args.seed,
                                   quiet=True)
            elapsed = (datetime.datetime.now() - started).total_seconds()
            new_tokens = tokens[len(enc.encode(prompt)):]
            results.append({
                "prompt": prompt,
                "depth": depth,
                "standard": index < standard,
                "text": enc.decode(new_tokens).strip(),
                "repetition": repetition_score(new_tokens),
                "tokens_per_second": (len(new_tokens) / elapsed) if elapsed > 0 else None,
                "generated_tokens": len(new_tokens),
            })
            print(f"  d{depth} [{index + 1}/{len(prompts)}] "
                  f"rep {results[-1]['repetition']:.3f}  {prompt[:40]}")

    # The first completion at each depth pays for a JIT compile of the whole
    # generation step, which is ~50x the steady-state cost. Averaging it in would
    # make the throughput field a lie that gets worse the shorter the run.
    steady = [r for r in results if r["tokens_per_second"] is not None][len(depths):]
    throughput = round(sum(r["tokens_per_second"] for r in steady) / len(steady), 2) if steady else None

    fields = {
        "prompt_set_version": PROMPT_SET_VERSION,
        "step": opt_step,
        "checkpoint_step": ckpt_step,
        "tokens": opt_step * TOKENS_PER_OPT_STEP,
        "train_ce": None if train_ce is None else round(train_ce, 4),
        "val_ce": None if val_ce is None else round(val_ce, 4),
        "val_ce_depth": _val_depth(),
        "depths": list(depths),
        "device": args.device,
        "seed": args.seed,
        "temperature": temperature,
        "max_new_tokens": args.max_new_tokens,
        "model_arch": MODEL_ARCH,
        "model_commit": model_commit,
        "model_commit_dirty": model_dirty,
        "tool_commit": git_head(),
        "tokens_per_second": throughput,
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    out_dir = args.out or os.path.join(run_dir or ".", "transcripts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, transcript_filename(opt_step, args.device))
    with open(out_path, "w") as handle:
        handle.write(render_document(fields, results, depths))
    print(f"\n✨ {out_path}")


def _val_depth():
    """VAL_FIXED_DEPTH, recorded separately because val_ce is measured there while
    the completions span the whole ladder. One `depth:` key would silently claim
    the CE and the text describe the same configuration. They do not."""
    try:
        from trm.train.validation import VAL_FIXED_DEPTH
        return VAL_FIXED_DEPTH
    except ImportError:
        return None


def render_document(fields, results, depths):
    parts = [render_frontmatter(fields), ""]
    parts += [f"# Transcripts — opt step {fields['step']:,} · {fields['tokens'] / 1e9:.3f}B tokens", ""]

    # Mean repetition per depth, up top: if a depth overthinks, it loops, and this
    # is the one line that shows it without reading eight completions.
    parts += ["## Repetition by depth", "",
              "Fraction of 4-grams already seen earlier in the same completion. "
              "Lower is better; a hard loop approaches 1.0.", "",
              "| depth | mean repetition |", "|---|---|"]
    for depth in depths:
        at_depth = [r["repetition"] for r in results if r["depth"] == depth and r["standard"]]
        mean = sum(at_depth) / len(at_depth) if at_depth else 0.0
        parts.append(f"| {depth} | {mean:.3f} |")
    parts.append("")

    for prompt in dict.fromkeys(r["prompt"] for r in results):
        rows = [r for r in results if r["prompt"] == prompt]
        tag = "" if rows[0]["standard"] else "  *(non-standard, passed on the command line)*"
        parts += [f"## `{prompt}`{tag}", ""]
        for row in sorted(rows, key=lambda r: r["depth"]):
            parts += [f"**depth {row['depth']}** · repetition {row['repetition']:.3f}", "",
                      "```", row["text"] or "(empty)", "```", ""]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
