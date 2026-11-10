"""Shared helpers for offline diagnostic tools.

Import AFTER setting any XLA env vars (each tool sets its own memory fraction
before importing jax through this module).
"""

import os

import orbax.checkpoint as ocp
from flax import nnx

from trm.config import resolve_root
from trm.model import build_model
from trm.runtime.checkpoints import discover_latest_checkpoint_run
from trm.runtime.layout import CHECKPOINT_ITEMS
from trm.settings import location
from trm.train.validation import read_heldout_rows, tail_skip


def restore_model(config, checkpoint_path=None, *, step=None, dim=None, **overrides):
    """Model-only restore, shaped by `config`, from a checkpoint dir (default: the
    latest run's), at `step` (default: its newest step — a dir of many steps, like a
    run's milestones, needs the step named or it silently scores the newest, #328).

    The skeleton comes from `trm.model.build_model`, the factory the trainer uses, so
    a restore builds exactly the param tree a run wrote. `dim` and `overrides` exist
    so a test can restore a tiny checkpoint."""
    model = build_model(config, nnx.Rngs(42), dim=dim, **overrides)
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    mngr = ocp.CheckpointManager(checkpoint_path, item_names=CHECKPOINT_ITEMS)
    step = mngr.latest_step() if step is None else step
    if step is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}")
    if step not in mngr.all_steps():
        raise SystemExit(f"No checkpoint at step {step} under {checkpoint_path}")
    print(f"📖 Restoring model weights from step {step} ({checkpoint_path})")
    restored = mngr.restore(step, args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))))
    nnx.update(model, restored["model"])
    return model, step


def load_eval_batches(config, source="pretrain/fineweb-edu", num_rows=16, skip=None):
    """Held-out rows, by default from the corpus's tail (#596), which a run reaches only
    if it exhausts the corpus. Rows are `config`'s shape, read the way its trainer reads them.

    Counted in ROWS and scored one row at a time, independent of BATCH_SIZE
    (#24): the eval slice must not move when a training throughput knob does, or
    every recorded yardstick number stops being comparable."""
    data_root = location("DATA_ROOT", "")
    if not data_root:
        raise SystemExit("DATA_ROOT is not set.")
    source_dir = f"{resolve_root(data_root)}/{source}"
    skip = tail_skip(source_dir, config.MAX_SEQ_LEN) if skip is None else skip

    batches = read_heldout_rows(source_dir, num_rows, skip,
                                max_seq_len=config.MAX_SEQ_LEN, data_seed=config.DATA_SEED)
    if not batches:
        import glob
        chunks = sorted(glob.glob(f"{source_dir}/*.npy"))
        detail = (f" It holds {len(chunks)} chunk(s)." if chunks
                  else " No .npy chunks found there at all — check the path.")
        raise SystemExit(f"No eval data in {source_dir} after skipping {skip:,} samples.{detail}")
    return batches
