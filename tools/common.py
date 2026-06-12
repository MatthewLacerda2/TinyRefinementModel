"""Shared helpers for offline diagnostic tools.

Import AFTER setting any XLA env vars (each tool sets its own memory fraction
before importing jax through this module).
"""

import os

from flax import nnx
import orbax.checkpoint as ocp

from config import LATENT_DIM, BATCH_SIZE, resolve_root
from model import UniversalReasoner
from data_loaders import TextDataGenerator
from checkpoint_utils import discover_latest_checkpoint_run


def restore_model(checkpoint_path=None):
    """Model-only restore from a checkpoint dir, defaulting to the latest run's."""
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
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


def load_eval_batches(source="pretrain/fineweb-edu", num_batches=16, skip=200_000):
    """Held-out batches: skip past the data the training run has consumed."""
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise SystemExit("DATA_ROOT is not set.")
    source_dir = f"{resolve_root(data_root)}/{source}"

    gen = TextDataGenerator(source_dir)
    gen.skip_count = skip
    batches = []
    while len(batches) < num_batches:
        batch, _ = gen.get_batch(BATCH_SIZE)
        if batch is None:
            break
        batches.append(batch)
    if not batches:
        raise SystemExit(f"No eval data available in {source_dir} after skipping {skip} samples.")
    return batches
