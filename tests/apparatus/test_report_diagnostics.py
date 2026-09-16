"""The report's diagnostics section speaks for the run's own architecture (#317).

metrics.csv keeps the reasoner-only columns for every arch, because old runs and
every reader depend on its schema. So "not measured by this architecture" must not
list them for a run recorded as plain: that is no news, and it buries the columns
whose absence is.
"""

from instruments import report
from instruments.runlog import RunLog


def _log(arch):
    metadata = {"parameters": {"MODEL_ARCH": arch}} if arch else {}
    return RunLog("run_x", [{"step": 64, "logz_mean": 2.0, "max_abs_logit": 9.0}], metadata)


def _absent_line(capsys, arch):
    report._print_diagnostics(_log(arch))
    return next(line for line in capsys.readouterr().out.splitlines() if "not measured" in line)


def test_a_plain_run_does_not_list_the_reasoner_only_columns(capsys):
    line = _absent_line(capsys, "plain")
    assert not any(col in line for col in report._REASONER_ONLY)
    assert "out_entropy" in line, "a column plain does report is still named when missing"


def test_a_reasoner_run_and_an_unrecorded_one_still_list_them(capsys):
    for arch in ("reasoner", None):
        assert "tau" in _absent_line(capsys, arch)
