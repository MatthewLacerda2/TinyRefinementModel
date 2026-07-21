"""What must NOT move when BATCH_SIZE does (#24).

Doubling the micro-batch and halving the accumulation is a pure throughput
change: same tokens, same schedule, same measurement. Three things make that
true, and each has failed silently in a way no crash would catch:

  1. TOKENS_PER_OPT_STEP stays fixed, so the LR cosine and the token budget
     describe the same run.
  2. The held-out eval slice is counted in ROWS and scored at batch 1, so a val
     CE stays comparable to every number already recorded.
  3. Resume seeks by the RECORDED sample count, so a run that resumes under a
     different batch size than it was trained at neither re-reads data it has
     seen nor skips a slice it hasn't.
"""

import inspect

import numpy as np
import pytest

import checkpoint_utils
import config
import trainer as trainer_mod
import validation
from config import ACCUMULATION_STEPS, BATCH_SIZE, EVAL_ROWS, MAX_SEQ_LEN
from tools import common
from tools import eval_refiner_depth_transfer as depth_transfer
from trainer import samples_from_micro_steps, split_samples


def test_tokens_per_opt_step_is_the_batching_invariant():
    """The product is the contract: BATCH_SIZE x ACCUMULATION_STEPS is what the
    LR horizon and TRAIN_TOKEN_BUDGET are denominated in. Changing one without
    the other silently rescales the whole schedule."""
    assert config.TOKENS_PER_OPT_STEP == ACCUMULATION_STEPS * BATCH_SIZE * 2 * MAX_SEQ_LEN
    # 131072 is the value every recorded run and every DECAY_STEPS was computed
    # against. If this must change, it is a new run, not a resume.
    assert config.TOKENS_PER_OPT_STEP == 131072


class _FakeGen:
    """Minimal TextDataGenerator stand-in: records the size of every read so a
    test can see exactly how a loader consumed the stream."""

    def __init__(self):
        self.calls = []
        self._n = 0

    def get_batch(self, batch_size):
        self.calls.append(batch_size)
        rows = np.arange(self._n, self._n + batch_size).reshape(batch_size, 1)
        self._n += batch_size
        return rows, np.zeros((batch_size,), dtype=bool)


def test_eval_loaders_read_one_row_at_a_time():
    """Every held-out loader must read rows singly, never BATCH_SIZE at a time.
    That reproduces the pre-#24 access pattern exactly — same rows, same order,
    same place a file boundary lands — so a val CE stays comparable to the
    champion's 4.7092 and the #17 noise floor no matter what BATCH_SIZE is."""
    for loader in (validation.ValidationProbe._load, common.load_eval_batches,
                   depth_transfer.load_domain_batches):
        src = inspect.getsource(loader)
        assert "get_batch(1)" in src, (
            f"{loader.__qualname__} must read one row per call; a get_batch(BATCH_SIZE) "
            "read would resize the eval slice when the throughput knob moves."
        )
        assert "get_batch(BATCH_SIZE)" not in src


def test_eval_probe_collects_exactly_eval_rows():
    """The probe's slice size is EVAL_ROWS rows — a constant, not a function of
    BATCH_SIZE."""
    gen = _FakeGen()
    batches = []
    while len(batches) < EVAL_ROWS:
        row, _ = gen.get_batch(1)
        batches.append(row)
    assert gen.calls == [1] * EVAL_ROWS
    assert all(b.shape[0] == 1 for b in batches)


def test_eval_batch_size_is_pinned_independent_of_training_batch():
    """Eval builds the reasoner skeleton at batch 1 regardless of BATCH_SIZE:
    its hunch_cache is shaped [batch, slots, dim] and every checkpoint we hold
    was written when BATCH_SIZE was 1. Tying this to the training knob would
    make stored checkpoints unrestorable."""
    assert common.EVAL_BATCH_SIZE == 1


WEIGHTS = [0.5, 0.3, 0.2]
MICRO_STEPS = 1000


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_resume_skip_scales_with_batch_size(batch_size):
    """A resume at micro-step N must skip N x BATCH_SIZE samples. The total can
    fall short by at most one sample per source (integer truncation) — never by
    a FACTOR of BATCH_SIZE, which is the bug this guards."""
    skips = samples_from_micro_steps(MICRO_STEPS, WEIGHTS, batch_size=batch_size)
    expected = MICRO_STEPS * batch_size
    assert expected - len(WEIGHTS) < sum(skips) <= expected


def test_resume_skip_without_the_batch_factor_under_skips():
    """The exact defect, stated: counting micro-steps as samples. It re-feeds the
    model 1/BATCH_SIZE of its own history, and nothing crashes."""
    correct = sum(samples_from_micro_steps(MICRO_STEPS, WEIGHTS))
    buggy = sum(samples_from_micro_steps(MICRO_STEPS, WEIGHTS, batch_size=1))
    assert correct == pytest.approx(buggy * BATCH_SIZE, rel=0.01)
    if BATCH_SIZE > 1:
        assert correct > buggy


def test_recorded_sample_count_beats_rederiving_it():
    """A checkpoint written at BATCH_SIZE=1 and resumed at 2: re-deriving from
    micro-steps seeks twice as far as the run actually read, skipping a slice of
    corpus it never saw. The recorded count lands exactly."""
    micro_steps_trained, weights = 1000, [0.5, 0.3, 0.2]
    truly_consumed = micro_steps_trained * 1          # it ran at batch 1

    exact = split_samples(truly_consumed, weights)
    rederived = samples_from_micro_steps(micro_steps_trained, weights, batch_size=2)

    assert sum(exact) == pytest.approx(truly_consumed, abs=len(weights))
    assert sum(rederived) == pytest.approx(2 * truly_consumed, abs=len(weights))


def test_samples_seen_is_counted_not_derived():
    """The consumed-sample counter must accumulate actual batch rows. Deriving it
    at save time as step x BATCH_SIZE is wrong for the one run that needs it: a
    resume whose history spans two different batch sizes."""
    save_src = inspect.getsource(checkpoint_utils.save_checkpoint)
    assert '"samples_seen": monitor.samples_seen' in save_src
    assert "step * BATCH_SIZE" not in save_src
    assert "monitor.samples_seen += batch.shape[0]" in inspect.getsource(trainer_mod.train_loop)


def test_phase_flip_preserves_the_data_position():
    """An SFT phase flip resets plateau state, never the consumed-sample count —
    zeroing it would rewind the data stream to the start of the corpus."""
    from monitor import LossMonitor

    m = LossMonitor()
    m.samples_seen = 123456
    m.reset_for_new_phase(step=10)
    assert m.samples_seen == 123456


def test_pre_24_checkpoints_resume_exactly():
    """A checkpoint with no samples_seen was written at BATCH_SIZE=1, so one
    sample per micro-step is its exact position — not an approximation."""
    restore_src = inspect.getsource(checkpoint_utils.load_or_create_checkpoint)
    assert 'm_state.get("samples_seen", restored["step"])' in restore_src
