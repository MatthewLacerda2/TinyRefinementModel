"""Held-out validation: the same fixed batches, scored the same deterministic
way, on demand. Train CE cannot see overfitting or data drift; this curve is
the one decisions should read. The trainer drives it on its own cadence
(VAL_EVERY_OPT_STEPS); everything about *what* a probe measures lives here. What it
reads — how many rows, from where — is the run's Config (`ValidationProbe.of`, #475).
"""

import glob
import os

import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.config import EOT_TOKEN_ID
from trm.data.loaders import TextDataGenerator
from trm.train.losses import chunked_cross_entropy_rows

# The fineweb slice starts at VAL_SKIP_SAMPLES, past any plausible training
# consumption, and reads EVAL_ROWS rows (both trm/settings.py).
# The other corpora the trainer probes (#363), each read from its own tail: the last
# VAL_TAIL_ROWS samples, which a run reaches only if it exhausts that corpus. The
# shipped end-mix is 40% code and 25% math, and a probe that reads prose alone cannot
# see what a mixture change costs them. fineweb keeps its fixed skip above, so every
# val_ce on record stays comparable.
VAL_BY_SOURCE = ("codeparrot", "finemath")
VAL_TAIL_ROWS = 20_000


def heldout_targets(targets, pad_token_id):
    """The targets the held-out CE scores: the document separator is never one of
    them, whatever the pad is (#373). Masked as pad, it never was, so every val CE
    on record excludes it, and a pair that changes the pad must still compare the
    same positions."""
    return jnp.where(targets == EOT_TOKEN_ID, pad_token_id, targets)


@nnx.jit
def _val_ce_sums(model, batch):
    """Masked CE sums over both windows, mirroring the training segment structure
    (window 1 opens the document, window 2 continues it).

    Scored exactly the way the grad step scores (#208): the model hands back
    pre-head states and the tied LM head is projected chunk by chunk, so the two
    [1, 512, 50304] f32 logit tensors (~206 MB) are never built. This probe runs
    inside the trainer's allocator on a fixed cadence, and a large, periodic,
    short-lived allocation is the shape that fragments an arena — the documented
    killer of every base run. `training=True` only selects the output form (plus
    remat, which is math-identical); no model uses it for dropout or noise.

    A row is two windows and the target after them, so the window length is the
    row's own (`(width - 1) // 2`); the pad is the one the model masks.
    """
    window = (batch.shape[1] - 1) // 2
    seq1_in, seq1_out = batch[:, :window], batch[:, 1:window + 1]
    seq2_in, seq2_out = batch[:, window:2 * window], batch[:, window + 1:2 * window + 1]
    out1 = model(seq1_in, training=True)
    out2 = model(seq2_in, training=True)
    targets = heldout_targets(jnp.concatenate([seq1_out, seq2_out], axis=0), model.pad_token_id)
    loss_sums, counts, _ = chunked_cross_entropy_rows(  # pyright: ignore[reportGeneralTypeIssues] -- jax leaves custom_vjp's return untyped
        jnp.concatenate([out1.hidden, out2.hidden], axis=0),
        model.embed.embedding[...],
        targets,
        model.pad_token_id)
    return loss_sums.sum(), counts.sum()


def read_heldout_rows(source_dir, rows, skip, *, max_seq_len, data_seed):
    """Up to `rows` held-out rows from `source_dir`, after skipping `skip` samples.
    `max_seq_len` and `data_seed` are the run's (MAX_SEQ_LEN, DATA_SEED): the reader
    the trainer would build.

    The one held-out row reader: the trainer's probe and every offline tool read
    through it, so their slices cannot drift apart. It returns fewer rows (possibly
    none) when the corpus runs out, and each caller says what that means for it.

    One row per batch, independent of BATCH_SIZE (#24): this reproduces the pre-#24
    read pattern exactly — including where a file boundary lands mid-slice — so a
    measured val CE stays comparable to every number already recorded. Batching a
    handful of rows would buy nothing anyway."""
    gen = TextDataGenerator(source_dir, max_seq_len=max_seq_len, rng=np.random.default_rng(data_seed))
    gen.skip_count = skip
    batches = []
    while len(batches) < rows:
        row = gen.get_batch(1)
        if row is None:
            break
        batches.append(row)
    return batches


def corpus_samples(source_dir, max_seq_len):
    """How many samples TextDataGenerator can read from `source_dir`, counted the way
    its skip counts them (whole strides per file)."""
    stride = 2 * max_seq_len + 1
    return sum(np.load(f, mmap_mode="r").shape[0] // stride
               for f in sorted(glob.glob(os.path.join(source_dir, "*.npy"))))


class ValidationProbe:
    """Loads `rows` fixed held-out rows once, then scores them on demand.

    `skip=None` reads the corpus's tail (the last VAL_TAIL_ROWS samples) instead of
    a fixed offset: the held-out slice for the probes of #363."""

    def __init__(self, data_root, *, rows, skip, max_seq_len, data_seed, source="fineweb-edu"):
        self.source_dir = f"{data_root}/pretrain/{source}"
        self.rows, self.skip, self.source = rows, skip, source
        self.max_seq_len, self.data_seed = max_seq_len, data_seed
        self._batches = None

    @classmethod
    def of(cls, config, data_root, source="fineweb-edu", **overrides):
        """The probe the trainer runs for `source`: EVAL_ROWS rows, fineweb from its
        fixed VAL_SKIP_SAMPLES, every other corpus from its tail (#363)."""
        return cls(data_root, **{"rows": config.EVAL_ROWS,
                                 "skip": config.VAL_SKIP_SAMPLES if source == "fineweb-edu" else None,
                                 "max_seq_len": config.MAX_SEQ_LEN, "data_seed": config.DATA_SEED,
                                 "source": source, **overrides})

    def load_rows(self):
        skip = self.skip
        if skip is None:
            skip = max(corpus_samples(self.source_dir, self.max_seq_len) - VAL_TAIL_ROWS, 0)
        batches = read_heldout_rows(self.source_dir, self.rows, skip,
                                    max_seq_len=self.max_seq_len, data_seed=self.data_seed)
        if not batches:
            print(f"⚠️ Validation of {self.source} disabled: no held-out data past the skip range.")
        return batches

    def run(self, model):
        if self._batches is None:
            self._batches = self.load_rows()
        if not self._batches:
            return None
        total, count = 0.0, 0
        for batch in self._batches:
            ce_sum, ce_count = _val_ce_sums(model, batch)
            total += float(ce_sum)
            count += int(ce_count)
        return total / max(count, 1)
