"""The small helpers every instrument used to carry its own copy of (#319).

Each one had two to seven copies, and the copies disagreed about what to do when
the thing they asked for was not there: one returned None, one raised, one had no
timeout and hung. So each helper here states its failure policy once, and a caller
that wants a louder failure makes it louder itself.

Importing this module is cheap: no jax, no trm.config. A helper that needs one of
them imports it inside the function, so `--help` stays instant and a pure-Python
test can use the rest.

Run-artefact readers (metrics.csv, run_metadata.json, checkpoint step dirs) are not
here; they live in `instruments/runlog.py`, the one place that knows those shapes.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# f16's largest finite value: a property of the dtype, not a config knob. Every margin
# measured against it (the smoke's headroom bar, the invariants' act_max check) keeps
# its own rationale where it is used.
F16_MAX = 65504.0


def git_head(short=True, cwd=REPO_ROOT):
    """The commit checked out at `cwd` (the repo by default), or None.

    Failure policy: None when git is missing, `cwd` is not a repository, or git
    does not answer within 5s. A provenance field that says "unknown" is honest; a
    tool that crashes or hangs because of one is not."""
    argv = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def gpu_memory_used_mib():
    """nvidia-smi's memory.used for the first card, in MiB, or None.

    Failure policy: None when there is no nvidia-smi, no card, or no answer within
    10s. Under cuda_async this is the reserved pool, not what a model uses; read
    `memory_stats()` for that. This is for "is something already holding the card?"."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def param_count(model):
    """Trainable parameters (nnx.Param leaves) in a built model."""
    import jax
    from flax import nnx
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def load_env():
    """Load the repo's `.env` into the environment (DATA_ROOT and friends).

    Where it is called decides what `.env` may set. trm.config reads MODEL_ARCH,
    TIME_SIGNAL and the rest once, at import: called before that import, `.env` can
    set them; called after, only what is read at run time (DATA_ROOT) takes effect,
    which is what a tool that wants config knobs set in the shell does on purpose.
    Values already in the environment win either way."""
    from dotenv import load_dotenv
    load_dotenv()


def module_env(root=REPO_ROOT, **extra):
    """The environment for running a sibling `python -m` CLI against `root`'s code.

    `root` goes first on PYTHONPATH, ahead of anything inherited, so the child imports
    the tree the caller means (a revived worktree, or this checkout) and never a stale
    one further down the path. `extra` sets more variables. What to do with the
    child's output and exit code stays the caller's policy: they genuinely differ
    (stream it, journal it, raise on it)."""
    inherited = os.environ.get("PYTHONPATH", "")
    path = str(root) + (os.pathsep + inherited if inherited else "")
    return {**os.environ, "PYTHONPATH": path, **{k: str(v) for k, v in extra.items()}}


def add_checkpoint_argument(parser, *, required=False, aliases=()):
    """`--checkpoint-path`, the one spelling, landing in `args.checkpoint_path`.

    `aliases` keeps older spellings working (`--checkpoint`, `--ckpt`) so no command
    already written in a finding or a spec breaks."""
    parser.add_argument(
        "--checkpoint-path", *aliases, dest="checkpoint_path", default=None, required=required,
        help="an orbax checkpoint MANAGER dir, the one holding numeric step dirs, not a step dir"
             + ("" if required else " (default: the latest checkpointed run under runs/)"))
    return parser
