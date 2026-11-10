"""The mixture pair's harness (#489): the arm's endpoint is found by name, and the
per-source readout is the last one the run logged."""

from experiments.mix.branch_decay import last_by_source, resume_env, step_name


def test_the_endpoint_checkpoint_is_named_by_micro_step():
    """The base run's pre-decay copy is opt step 30,528 and its dir is 1953791."""
    assert step_name(30528, 64) == "1953791"
    assert step_name(31552, 64) == "2019327"


def test_the_last_per_source_reading_wins(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,val_by_source\n64,codeparrot=0.90;finemath=2.30\n"
                       "128,\n192,codeparrot=0.85;finemath=2.25\n256,\n")
    assert last_by_source(metrics) == {"codeparrot": 0.85, "finemath": 2.25}


def test_only_the_branch_copy_is_branched_a_relaunch_resumes_its_own_stream(tmp_path):
    """Re-branching on a relaunch would replay the arm's rows and unmatch the pair (#489)."""
    copy = tmp_path / "1953791"
    for d in (copy, tmp_path / "best_val_ce", tmp_path / "milestones"):
        d.mkdir()
    assert resume_env(tmp_path, copy, {"A": "1"}) == {"A": "1", "DATA_BRANCH": "1"}
    (tmp_path / "1978367").mkdir()
    assert resume_env(tmp_path, copy, {"A": "1"}) == {"A": "1"}
