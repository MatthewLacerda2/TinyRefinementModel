"""The mixture pair's harness (#489): the arm's endpoint is found by name, and the
per-source readout is the last one the run logged."""

from experiments.mix.branch_decay import last_by_source, step_name


def test_the_endpoint_checkpoint_is_named_by_micro_step():
    """The base run's pre-decay copy is opt step 30,528 and its dir is 1953791."""
    assert step_name(30528, 64) == "1953791"
    assert step_name(31552, 64) == "2019327"


def test_the_last_per_source_reading_wins(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,val_by_source\n64,codeparrot=0.90;finemath=2.30\n"
                       "128,\n192,codeparrot=0.85;finemath=2.25\n256,\n")
    assert last_by_source(metrics) == {"codeparrot": 0.85, "finemath": 2.25}
