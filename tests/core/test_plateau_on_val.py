"""The plateau detector reads held-out CE, on the probe's cadence (#184)."""

from trm.runtime.monitor import LossMonitor


def test_flat_val_ce_plateaus_after_patience_and_an_improvement_resets_it():
    m = LossMonitor(patience=200, window=4, min_delta=0.01)
    for i, v in enumerate([4.0, 3.8, 3.6, 3.5]):
        m.push_val(v, step=64 * (i + 1))
    assert not m.plateaued
    for i in range(8):                       # flat within min_delta for 512 steps
        m.push_val(3.5 + 0.002 * (i % 2), step=64 * (5 + i))
    assert m.plateaued
    m.push_val(3.2, step=64 * 14)            # a real improvement
    assert not m.plateaued


def test_train_ce_no_longer_drives_the_plateau():
    """The curriculum moves train CE; a rising train CE must not read as a stall."""
    m = LossMonitor(patience=100, window=4, min_delta=0.005)
    for step in range(0, 2000, 5):
        m.push(step, 3.2 + step / 10_000, 3.2)   # train CE creeping UP for 2000 steps
    assert not m.plateaued
    assert m.best_ce == 3.2


def test_push_val_without_a_step_only_tracks_the_best():
    m = LossMonitor()
    assert m.push_val(3.0) and not m.push_val(3.1)
    assert m.ce_history == [] and not m.plateaued


def test_a_new_phase_clears_the_plateau():
    m = LossMonitor(patience=1, window=2)
    m.push_val(3.0, step=1)
    m.push_val(3.0, step=10)
    assert m.plateaued
    m.reset_for_new_phase(10)
    assert not m.plateaued


def test_the_trainer_reads_the_plateau_from_the_probe():
    import inspect
    from trm.train import trainer
    src = inspect.getsource(trainer.train_loop)
    assert "monitor.push_val(val_ce, opt_step)" in src and "plateaued = monitor.plateaued" in src
    assert "plateaued = monitor.push(" not in src


def test_probe_slices_are_disjoint_and_start_at_the_trainers_own():
    from instruments.probe_sigma import slice_offsets
    from trm.train.validation import VAL_SKIP_SAMPLES
    offs = slice_offsets(64, 3)
    assert offs[0] == VAL_SKIP_SAMPLES and offs[1] - offs[0] == 64 and offs[2] - offs[1] == 64
