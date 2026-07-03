"""Held-out validation: the same fixed batches, scored the same deterministic
way, on demand. Train CE cannot see overfitting or data drift; this curve is
the one decisions should read. The trainer drives it on its own cadence
(VAL_EVERY_OPT_STEPS there); everything about *what* a probe measures lives here.
"""

import jax.numpy as jnp
import optax
from flax import nnx

from config import BATCH_SIZE, MAX_SEQ_LEN, PAD_TOKEN_ID
from data_loaders import TextDataGenerator

VAL_BATCHES = 4
VAL_FIXED_DEPTH = 4
# Far past any plausible training consumption (an 8k-opt-step run consumes
# under 1M fineweb samples; fineweb holds 4.3M) so the slice stays held out.
VAL_SKIP_SAMPLES = 3_000_000


@nnx.jit
def _val_ce_sums(model, batch):
    """Masked CE sums over both windows, mirroring the training segment structure
    (window 1 fresh, window 2 on the carried hunch) at a fixed depth."""
    seq1_in, seq1_out = batch[:, :MAX_SEQ_LEN], batch[:, 1:MAX_SEQ_LEN + 1]
    seq2_in, seq2_out = batch[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN], batch[:, MAX_SEQ_LEN + 1:2 * MAX_SEQ_LEN + 1]
    out1 = model(seq1_in, max_steps=VAL_FIXED_DEPTH, training=False, should_refresh=True)
    out2 = model(seq2_in, max_steps=VAL_FIXED_DEPTH, training=False, should_refresh=False)
    total = jnp.array(0.0)
    count = jnp.array(0)
    for logits, targets in ((out1.logits, seq1_out), (out2.logits, seq2_out)):
        mask = targets != PAD_TOKEN_ID
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        total += jnp.sum(ce * mask)
        count += jnp.sum(mask)
    return total, count


class ValidationProbe:
    """Loads VAL_BATCHES fixed held-out batches once, then scores them on demand.
    Restores the training stream's carried hunch afterwards, so validating never
    perturbs training."""

    def __init__(self, data_root):
        self.data_root = data_root
        self._batches = None

    def _load(self):
        gen = TextDataGenerator(f"{self.data_root}/pretrain/fineweb-edu")
        gen.skip_count = VAL_SKIP_SAMPLES
        batches = []
        while len(batches) < VAL_BATCHES:
            batch, _ = gen.get_batch(BATCH_SIZE)
            if batch is None:
                break
            batches.append(batch)
        if not batches:
            print("⚠️ Validation disabled: no held-out data available past the skip range.")
        return batches

    def run(self, model):
        if self._batches is None:
            self._batches = self._load()
        if not self._batches:
            return None
        saved_hunch = model.hunch_cache[...]
        total, count = 0.0, 0
        for batch in self._batches:
            model.hunch_cache[...] = jnp.zeros_like(saved_hunch)
            ce_sum, ce_count = _val_ce_sums(model, batch)
            total += float(ce_sum)
            count += int(ce_count)
        model.hunch_cache[...] = saved_hunch
        return total / max(count, 1)
