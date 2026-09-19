"""Shared helpers for offline diagnostic tools.

Import AFTER setting any XLA env vars (each tool sets its own memory fraction
before importing jax through this module).
"""

import os

from flax import nnx
import orbax.checkpoint as ocp

from trm.config import LATENT_DIM, MODEL_ARCH, resolve_root
from trm.model import build_model
from trm.runtime.checkpoints import discover_latest_checkpoint_run, restore_tolerating_legacy
from trm.runtime.layout import CHECKPOINT_ITEMS
from trm.train.validation import VAL_SKIP_SAMPLES, read_heldout_rows

# Eval builds and scores at batch 1, never at the training BATCH_SIZE (#24). The
# reasoner's hunch_cache is shaped [batch, slots, dim] and its forward asserts on
# the leading dim, so a reasoner skeleton built at the training batch would fail
# to restore every checkpoint we have — all written when BATCH_SIZE was 1 — for
# no benefit, since eval reads a handful of rows.
EVAL_BATCH_SIZE = 1


def _restore_into(model, checkpoint_path, step=None):
    """Model-only Orbax restore into an already-built model skeleton, at `step` or,
    by default, the newest step the manager dir holds. A dir of many steps (a run's
    milestones) needs the step named, or it silently scores the newest one (#328)."""
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    mngr = ocp.CheckpointManager(checkpoint_path, item_names=CHECKPOINT_ITEMS)
    latest = mngr.latest_step() if step is None else step
    if latest is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}")
    if latest not in mngr.all_steps():
        raise SystemExit(f"No checkpoint at step {latest} under {checkpoint_path}")
    print(f"📖 Restoring model weights from step {latest} ({checkpoint_path})")
    restored = restore_tolerating_legacy(
        lambda model_target: mngr.restore(
            latest,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(model_target)),
        ),
        model,
    )
    nnx.update(model, restored["model"])
    return model, latest


def restore_arch(arch, checkpoint_path=None, *, step=None, dim=None, **overrides):
    """Model-only restore of `arch` from a checkpoint dir (default: the latest
    run's), at `step` (default: its newest).

    The skeleton comes from `trm.model.build_model`, the factory the trainer uses, so
    a restore builds exactly the param tree a run of that arch wrote. The arch must be
    the one the checkpoint was trained as; for a run, its run_metadata.json records it.
    `dim` and `overrides` exist so a test can restore a tiny checkpoint. The reasoner
    is built at EVAL_BATCH_SIZE unless told otherwise; see above."""
    if arch == "reasoner":
        overrides = {"batch_size": EVAL_BATCH_SIZE, **overrides}
    try:
        model = build_model(arch, LATENT_DIM if dim is None else dim, nnx.Rngs(42), **overrides)
    except ValueError as err:
        if "unknown architecture" not in str(err):
            raise  # a constructor's own complaint, not a bad name
        raise SystemExit(f"{err}; use plain, refiner or reasoner") from None
    return _restore_into(model, checkpoint_path, step)


def restore_model(checkpoint_path=None):
    """Restore as MODEL_ARCH, whatever this process was launched with."""
    return restore_arch(MODEL_ARCH, checkpoint_path)


def load_eval_batches(source="pretrain/fineweb-edu", num_rows=16, skip=VAL_SKIP_SAMPLES):
    """Held-out rows: skip past the data the training run has consumed.

    The default skip sits far beyond plausible consumption (an 8k-opt-step run
    reads under 1M fineweb samples of its 4.3M) — the old 200k default was
    inside the range long runs train through, contaminating the eval slice.

    Counted in ROWS and scored one row at a time, independent of BATCH_SIZE
    (#24): the eval slice must not move when a training throughput knob does, or
    every recorded yardstick number stops being comparable. Batch-1 is also the
    shape every stored checkpoint of both arches was written at."""
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise SystemExit("DATA_ROOT is not set.")
    source_dir = f"{resolve_root(data_root)}/{source}"

    batches = read_heldout_rows(source_dir, num_rows, skip)
    if not batches:
        # "No eval data available" alone sends you looking for a missing corpus.
        # The actual cause is almost always that the default skip -- sized for the
        # 30-chunk corpora -- overruns a smaller one: finemath has 19 chunks and
        # runs out well before 3,000,000 samples. Say which, and say the number
        # that would work, because the alternative (silently clamping the skip)
        # would quietly evaluate on tokens the run trained through.
        import glob
        chunks = sorted(glob.glob(f"{source_dir}/*.npy"))
        detail = (f" It holds {len(chunks)} chunk(s)." if chunks
                  else " No .npy chunks found there at all — check the path.")
        raise SystemExit(
            f"No eval data in {source_dir} after skipping {skip:,} samples.{detail} "
            f"A smaller corpus needs a smaller --skip; the skip exists to stay clear "
            f"of trained-through data, so reduce it deliberately rather than to zero.")
    return batches
