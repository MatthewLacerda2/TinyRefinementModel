"""The report's diagnostics section names what a run left blank instead of drawing
it as zero (#105)."""

from instruments import report
from instruments.runlog import NOT_LOGGED, RunLog


def test_a_blank_column_is_named_as_not_logged(capsys):
    log = RunLog("run_x", [{"step": 64, "logz_mean": 2.0, "max_abs_logit": 9.0}], None)
    report._print_diagnostics(log)
    line = next(line for line in capsys.readouterr().out.splitlines() if NOT_LOGGED in line)
    assert "out_entropy" in line
    assert "logz_mean" not in line, "a column the run did log is not listed as missing"
