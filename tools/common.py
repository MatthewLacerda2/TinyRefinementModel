"""Shared helpers for offline diagnostic tools.

Import AFTER setting any XLA env vars (each tool sets its own memory fraction
before importing jax through this module).
"""

import os

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from config import LATENT_DIM, MODEL_ARCH, resolve_root

# Eval builds and scores at batch 1, never at the training BATCH_SIZE (#24). The
# reasoner's vestigial hunch_cache is shaped [batch, slots, dim] and its forward
# asserts on the leading dim, so a reasoner skeleton built at the training batch
# would fail to restore every checkpoint we have — all written when BATCH_SIZE
# was 1 — for no benefit, since eval reads a handful of rows.
EVAL_BATCH_SIZE = 1
from model import UniversalReasoner
from data_loaders import TextDataGenerator
from checkpoint_utils import discover_latest_checkpoint_run


def _restore_into(model, checkpoint_path):
    """Model-only Orbax restore into an already-built model skeleton."""
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "optimizer", "monitor_state", "step"),
    )
    latest = mngr.latest_step()
    if latest is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}")
    print(f"📖 Restoring model weights from step {latest} ({checkpoint_path})")
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
    )
    nnx.update(model, restored["model"])
    return model, latest


def build_model():
    """Fresh skeleton matching MODEL_ARCH. The two arches have different param
    trees, so a restore must build the same arch the checkpoint was trained as
    (select it at launch, e.g. MODEL_ARCH=refiner)."""
    if MODEL_ARCH == "refiner":
        from plan_a_trainer import RefinerForTraining
        return RefinerForTraining(LATENT_DIM, nnx.Rngs(42))
    return UniversalReasoner(LATENT_DIM, nnx.Rngs(42), batch_size=EVAL_BATCH_SIZE)


def restore_model(checkpoint_path=None):
    """Model-only restore from a checkpoint dir, defaulting to the latest run's.
    Builds the skeleton per MODEL_ARCH; use restore_reasoner/restore_refiner to
    pin an arch explicitly (e.g. eval_yardstick's --arch override)."""
    return _restore_into(build_model(), checkpoint_path)


def restore_reasoner(checkpoint_path=None):
    """Reasoner restore regardless of MODEL_ARCH, for explicit --arch overrides."""
    return _restore_into(
        UniversalReasoner(LATENT_DIM, nnx.Rngs(42), batch_size=EVAL_BATCH_SIZE), checkpoint_path)


def restore_refiner(checkpoint_path=None):
    """Refiner restore into the production wrapper (RefinerForTraining), so the
    checkpoint's saved 'model' state loads with matching structure. Imported
    lazily: reasoner-only tools shouldn't pay for the Plan A import."""
    from plan_a_trainer import RefinerForTraining

    return _restore_into(RefinerForTraining(LATENT_DIM, nnx.Rngs(42)), checkpoint_path)


def load_eval_batches(source="pretrain/fineweb-edu", num_rows=16, skip=3_000_000):
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

    gen = TextDataGenerator(source_dir)
    gen.skip_count = skip
    batches = []
    while len(batches) < num_rows:
        row, _ = gen.get_batch(1)
        if row is None:
            break
        batches.append(row)
    if not batches:
        raise SystemExit(f"No eval data available in {source_dir} after skipping {skip} samples.")
    return batches
