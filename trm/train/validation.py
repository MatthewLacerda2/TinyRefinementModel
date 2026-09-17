"""Held-out validation: the same fixed batches, scored the same deterministic
way, on demand. Train CE cannot see overfitting or data drift; this curve is
the one decisions should read. The trainer drives it on its own cadence
(VAL_EVERY_OPT_STEPS, trm/runtime/layout.py); everything about *what* a probe measures lives here.
"""

import jax.numpy as jnp
from flax import nnx

from trm.config import EVAL_ROWS, MAX_SEQ_LEN, PAD_TOKEN_ID
from trm.data.loaders import TextDataGenerator
from trm.train.losses import chunked_cross_entropy_rows

VAL_ROWS = EVAL_ROWS
VAL_FIXED_DEPTH = 4
# Far past any plausible training consumption (an 8k-opt-step run consumes
# under 1M fineweb samples; fineweb holds 4.3M) so the slice stays held out.
VAL_SKIP_SAMPLES = 3_000_000


@nnx.jit
def _val_ce_sums(model, batch):
    """Masked CE sums over both windows, mirroring the training segment structure
    (window 1 opens the document, window 2 continues it) at a fixed depth.

    Scored exactly the way the grad step scores (#208): the model hands back
    pre-head states and the tied LM head is projected chunk by chunk, so the two
    [1, 512, 50304] f32 logit tensors (~206 MB) are never built. This probe runs
    inside the trainer's allocator on a fixed cadence, and a large, periodic,
    short-lived allocation is the shape that fragments an arena — the documented
    killer of every base run. `training=True` only selects the output form (plus
    remat, which is math-identical); no model uses it for dropout or noise.
    """
    seq1_in, seq1_out = batch[:, :MAX_SEQ_LEN], batch[:, 1:MAX_SEQ_LEN + 1]
    seq2_in, seq2_out = batch[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN], batch[:, MAX_SEQ_LEN + 1:2 * MAX_SEQ_LEN + 1]
    out1 = model(seq1_in, depth=VAL_FIXED_DEPTH, training=True, new_document=True)
    out2 = model(seq2_in, depth=VAL_FIXED_DEPTH, training=True, new_document=False)
    loss_sums, counts, _ = chunked_cross_entropy_rows(
        jnp.concatenate([out1.hidden, out2.hidden], axis=0),
        model.embed.embedding[...],
        jnp.concatenate([seq1_out, seq2_out], axis=0),
        PAD_TOKEN_ID)
    return loss_sums.sum(), counts.sum()


def read_heldout_rows(source_dir, rows, skip):
    """Up to `rows` held-out rows from `source_dir`, after skipping `skip` samples.

    The one held-out row reader: the trainer's probe and every offline tool read
    through it, so their slices cannot drift apart. It returns fewer rows (possibly
    none) when the corpus runs out, and each caller says what that means for it.

    One row per batch, independent of BATCH_SIZE (#24): this reproduces the pre-#24
    read pattern exactly — including where a file boundary lands mid-slice — so a
    measured val CE stays comparable to every number already recorded. Batching a
    handful of rows would buy nothing anyway."""
    gen = TextDataGenerator(source_dir)
    gen.skip_count = skip
    batches = []
    while len(batches) < rows:
        row, _ = gen.get_batch(1)
        if row is None:
            break
        batches.append(row)
    return batches


class ValidationProbe:
    """Loads VAL_ROWS fixed held-out rows once, then scores them on demand.
    Runs inside the model's `isolated_state`, so whatever the training stream
    carries is restored afterwards and validating never perturbs training."""

    def __init__(self, data_root, rows=VAL_ROWS, skip=VAL_SKIP_SAMPLES):
        self.data_root = data_root
        self.rows, self.skip = rows, skip
        self._batches = None

    def _load(self):
        batches = read_heldout_rows(f"{self.data_root}/pretrain/fineweb-edu", self.rows, self.skip)
        if not batches:
            print("⚠️ Validation disabled: no held-out data available past the skip range.")
        return batches

    def run(self, model):
        if self._batches is None:
            self._batches = self._load()
        if not self._batches:
            return None
        total, count = 0.0, 0
        with model.isolated_state():
            for batch in self._batches:
                # Every probe row is scored from a clean slate, so the measurement
                # is a property of the weights and not of the row order.
                model.reset_state()
                ce_sum, ce_count = _val_ce_sums(model, batch)
                total += float(ce_sum)
                count += int(ce_count)
        return total / max(count, 1)
