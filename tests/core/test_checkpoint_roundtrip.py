"""Checkpoint save → restore → identical forward.

Exercises the same Orbax Composite pattern the trainer uses: weights from one
model are saved, restored into a differently-initialized model, and both must
produce identical logits on a fixed batch. Catches state-tree drift between
save and restore schemas — the failure mode that silently loads garbage."""

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from config import LATENT_DIM
from model import UniversalReasoner


def test_save_restore_roundtrip_preserves_forward(tmp_path, tiny_model, token_batch):
    tokens = jnp.asarray(token_batch)
    reference = np.asarray(tiny_model(tokens, max_steps=2, training=False, should_refresh=True).logits)

    mngr = ocp.CheckpointManager(
        tmp_path / "checkpoints",
        item_names=("model",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    mngr.save(0, args=ocp.args.Composite(model=ocp.args.StandardSave(nnx.state(tiny_model))))
    mngr.wait_until_finished()

    other = UniversalReasoner(LATENT_DIM, nnx.Rngs(1), batch_size=1)
    restored = mngr.restore(
        0, args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(other)))
    )
    nnx.update(other, restored["model"])

    roundtripped = np.asarray(other(tokens, max_steps=2, training=False, should_refresh=True).logits)
    np.testing.assert_array_equal(reference, roundtripped)
