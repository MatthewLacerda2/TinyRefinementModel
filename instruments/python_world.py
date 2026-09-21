"""How much Python a checkpoint can actually write, per rung of the ladder.

LAMBADA asks whether the model can talk. This asks whether it can *do* something,
and it answers with a program that ran: every point here is a process that executed
a candidate against its tests and exited. Nothing is scored by a model.

The readout is `pass@k` per difficulty level, on held-out tasks the training split
cannot contain (`trm.rl.tasks` splits by a hash of the instance). Per level, not
overall, because the number's job is to separate two checkpoints: a single figure
over a mixed set saturates at one end or floors at the other, and at 135M parameters
HumanEval and MBPP both floor. A ladder always has a rung where two models differ.

Read the shape, not the height. A ~138M base model trained on a few billion tokens
is not going to solve level 4, and a level-1 pass rate that climbs from nothing is
the whole signal at this scale.

    # the generator checking itself — no model, no card, a few seconds
    python -m instruments.python_world --reference

    # a checkpoint
    python -m instruments.python_world --checkpoint-path runs/<run>/checkpoints \
        --tasks 20 --samples 4 --k 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import tiktoken

from instruments._common import add_checkpoint_argument
from instruments.arch import add_arch_argument
from instruments.results import emit
from trm.config import TOKENIZER_NAME
from trm.rl import sandbox, tasks

# Where a completion stops being the function. A model that has finished the body
# keeps going — into a second definition, a print, a comment, whatever the training
# data usually had next — and everything from there on is not part of the answer.
# Cutting at a token that starts a new top-level statement is the standard trim; it
# is a property of how these models generate, not a favour to the model.
STOPS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\n>>>",
         "\nassert ")

DEFAULT_MAX_NEW_TOKENS = 192
# Sampling, not greedy: pass@k is only meaningful over a distribution, and a greedy
# decode makes every one of the k attempts the same attempt.
DEFAULT_TEMPERATURE = 0.8


def trim(completion: str) -> str:
    """Cut a raw completion back to the function body it started writing."""
    cut = len(completion)
    for stop in STOPS:
        found = completion.find(stop)
        if found != -1:
            cut = min(cut, found + 1)  # keep the newline, drop what follows
    return completion[:cut].rstrip() + "\n"


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased estimator (Chen et al. 2021): the chance that a random k of the
    n attempts contains at least one that passed.

    Sampling k and asking "did any pass" is the same quantity but a noisy one, and
    it wastes the other n-k attempts we already paid to generate.
    """
    if k > n:
        raise ValueError(f"cannot estimate pass@{k} from {n} samples")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def attempts_for(model, enc, task, samples: int, *, seed: int, temperature: float,
                 max_new_tokens: int, depth, limits: dict) -> list:
    """Generate `samples` completions for one task and verify each one.

    The seed is derived from the task and the attempt number, so the same checkpoint
    scores the same on the same tasks twice — a readout that moves on its own cannot
    be used to compare two checkpoints.
    """
    from trm.infer import generate_text

    prompt_length = len(enc.encode(task.prompt))
    outcomes = []
    for attempt in range(samples):
        # A stable digest, not `hash()`: Python salts string hashing per process, so
        # `hash()` here would give the same checkpoint a different score every run.
        digest = hashlib.sha256(f"{task.key}:{attempt}:{seed}".encode()).digest()
        rolled = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        tokens = generate_text(model, enc, task.prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, depth=depth, seed=rolled,
                               quiet=True)
        body = trim(enc.decode(tokens[prompt_length:]))
        outcomes.append(sandbox.verify_task(task.prompt + body, task, **limits))
    return outcomes


def score_level(level: int, attempts, k: int) -> dict:
    """Roll one level's per-task outcomes into the numbers that get reported."""
    solved = [sum(1 for outcome in per_task if outcome.solved) for per_task in attempts]
    samples = len(attempts[0]) if attempts else 0
    reasons: dict[str, int] = {}
    for per_task in attempts:
        for outcome in per_task:
            reasons[outcome.status] = reasons.get(outcome.status, 0) + 1
    return {
        "level": level,
        "tasks": len(attempts),
        "samples": samples,
        "pass_1": sum(pass_at_k(samples, c, 1) for c in solved) / max(1, len(solved)),
        f"pass_{k}": sum(pass_at_k(samples, c, k) for c in solved) / max(1, len(solved)),
        # Tests passed rather than tasks solved: a model that gets the empty list
        # wrong and everything else right is in a different place from one that
        # produces nothing, and pass@k alone calls them both zero.
        "test_fraction": (sum(o.passed for p in attempts for o in p)
                          / max(1, sum(o.total for p in attempts for o in p))),
        "reasons": reasons,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="pass@k on generated Python tasks, per difficulty level")
    add_checkpoint_argument(ap)
    add_arch_argument(ap)
    ap.add_argument("--step", type=int, default=None,
                    help="which step of that dir to score (default: its newest)")
    ap.add_argument("--reference", action="store_true",
                    help="score the reference solutions instead of a model: the generator "
                         "checking itself, which must come back 1.0 on every level")
    ap.add_argument("--levels", default=",".join(str(x) for x in tasks.LEVELS),
                    help=f"comma-separated difficulty levels (default: all of {tasks.LEVELS})")
    ap.add_argument("--tasks", type=int, default=20, help="held-out tasks per level")
    ap.add_argument("--samples", type=int, default=4,
                    help="completions generated per task; pass@k is estimated from these")
    ap.add_argument("--k", type=int, default=4, help="the k in pass@k (at most --samples)")
    ap.add_argument("--split", default="held_out", choices=tasks.SPLITS,
                    help="which half to draw from; train is for eyeballing, not for a number")
    ap.add_argument("--seed", type=int, default=0, help="which tasks, and which rolls")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--depth", type=int, default=None,
                    help="refinement/reasoning depth at eval (default: trm.infer's)")
    ap.add_argument("--timeout", type=float, default=sandbox.DEFAULT_TIMEOUT_S,
                    help="seconds one attempt may run before it counts as a timeout")
    ap.add_argument("--json-out", default=None, help="write the per-level rows here too")
    args = ap.parse_args(argv)

    if args.k > args.samples:
        ap.error(f"--k {args.k} needs at least that many --samples (got {args.samples})")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    limits = {"timeout_s": args.timeout}

    model = enc = depth = None
    if not args.reference:
        from trm.config import INFERENCE_DEPTH
        from trm.runtime.restore import restore_arch
        model, step = restore_arch(args.arch, args.checkpoint_path, step=args.step)
        enc = tiktoken.get_encoding(TOKENIZER_NAME)
        depth = INFERENCE_DEPTH if args.depth is None else args.depth
        print(f"📐 {args.arch} at step {step}, depth {depth}, temperature {args.temperature}")
    else:
        print("📐 reference solutions — the generator checking itself")

    rows = []
    for level in levels:
        drawn = tasks.generate(args.tasks, seed=args.seed, split=args.split, level=level)
        started = time.time()
        if args.reference:
            attempts = [[sandbox.verify_task(task.reference, task, **limits)] for task in drawn]
            k = 1
        else:
            attempts = [attempts_for(model, enc, task, args.samples, seed=args.seed,
                                     temperature=args.temperature,
                                     max_new_tokens=args.max_new_tokens, depth=depth,
                                     limits=limits)
                        for task in drawn]
            k = args.k
        row = score_level(level, attempts, k)
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)

        reasons = " ".join(f"{name}={count}" for name, count in sorted(row["reasons"].items()))
        print(f"level {level}: pass@1 {row['pass_1']:.3f}  pass@{k} {row[f'pass_{k}']:.3f}  "
              f"tests {row['test_fraction']:.3f}  ({row['tasks']} tasks, {row['seconds']}s)  {reasons}")
        emit(f"level{level}", **{"pass_1": row["pass_1"], f"pass_{k}": row[f"pass_{k}"],
                                 "test_fraction": row["test_fraction"], "tasks": row["tasks"]})

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
        print(f"📝 {args.json_out}")

    if args.reference:
        broken = [row for row in rows if row["pass_1"] < 1.0]
        if broken:
            raise SystemExit(f"❌ the generator disagrees with its own reference on "
                             f"level(s) {[row['level'] for row in broken]}")
        print("✅ every reference solution passes every test it was given")
    return rows


if __name__ == "__main__":
    main()
