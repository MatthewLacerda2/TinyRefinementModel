"""The stall rule is read from the spec, and it can read what the run trains on (#468).

001's rule read fineweb-edu alone while 65% of the tokens were code and math, so it
could end a run whose other sources were still learning. The case below is that run's
shape: web flat after the ramp, code and math still falling.
"""

import datetime
import pathlib

import pytest

from instruments import runlog, stall

REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC_001 = REPO / "experiments/base/specs/001-plain-base.toml"
T0 = datetime.datetime(2026, 9, 20, tzinfo=datetime.timezone.utc)
END_MIX = "fineweb-edu=0.350 codeparrot=0.400 finemath=0.250"
HEADER = "step,val_ce,val_step,val_by_source,wall_clock,mix"


def write_run(tmp_path, web, code, math, ramp_hours=10, hours=72, decay_hour=None):
    """One held-out reading an hour. The ramp moves the mix and drops every CE fast;
    after it each source follows its own slope (nats per hour), and from `decay_hour`
    on every CE falls a further 0.01 an hour, as a WSD decay does."""
    rows = [HEADER]
    for h in range(hours + 1):
        ramp = h < ramp_hours
        mix = f"fineweb-edu={0.85 - 0.05 * h:.3f} codeparrot=0.100 finemath=0.050" if ramp else END_MIX
        drop = 0.1 * min(h, ramp_hours)   # large ramp gains the rule must not read
        drop += 0.01 * max(h - decay_hour, 0) if decay_hour is not None else 0.0
        ce = {s: base - drop - slope * max(h - ramp_hours, 0)
              for s, base, slope in (("web", 4.0, web), ("code", 2.0, code), ("math", 3.5, math))}
        when = (T0 + datetime.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(f"{100 * h},{ce['web']:.4f},{100 * h},"
                    f"codeparrot={ce['code']:.4f};finemath={ce['math']:.4f},{when},{mix}")
    run = tmp_path / "run_x"
    run.mkdir(parents=True)
    (run / "metrics.csv").write_text("\n".join(rows) + "\n")
    return runlog.load(str(run))


def rule(reads, min_gain=0.01, sources=("fineweb-edu", "codeparrot", "finemath")):
    return stall.Rule(reads=reads, window_hours=12, windows=2, min_gain=min_gain,
                      decay_opt_steps=1000, sources=("fineweb-edu",) if reads == "web" else sources)


def write_spec(tmp_path, body):
    path = tmp_path / "spec.toml"
    path.write_text(f"[stall]\nwindow_hours = 12\nwindows = 2\ndecay_opt_steps = 1000\n{body}\n")
    return path


def test_001s_table_is_the_rule_its_notes_registered():
    """Transcribed after the run, so the table must say exactly what the prose did."""
    text = " ".join(SPEC_001.read_text().split())
    assert "TWO CONSECUTIVE 12-hour windows each gain less than 0.01" in text
    assert "1,000-opt-step decay" in text
    assert stall.load_rule(SPEC_001) == stall.Rule(reads="web", window_hours=12, windows=2,
                                                  min_gain=0.01, decay_opt_steps=1000)


def test_web_only_ends_a_run_whose_code_and_math_still_learn(tmp_path):
    log = write_run(tmp_path, web=0.0, code=0.002, math=0.002)   # 0.024 per 12h
    assert stall.check(rule("web"), log)[0] == stall.STALLED
    every = rule("every-source", {"fineweb-edu": 0.01, "codeparrot": 0.01, "finemath": 0.01})
    assert stall.check(every, log)[0] == stall.LEARNING
    assert stall.check(rule("mixture"), log)[0] == stall.LEARNING    # 0.65 x 0.024 = 0.016


def test_every_source_stalls_once_all_of_them_do(tmp_path):
    log = write_run(tmp_path, web=0.0, code=0.0002, math=0.0)
    bars = {"fineweb-edu": 0.01, "codeparrot": 0.01, "finemath": 0.01}
    outcome, gains = stall.check(rule("every-source", bars), log)
    assert outcome == stall.STALLED
    assert gains["codeparrot"] == pytest.approx([0.0024, 0.0024], abs=2e-4)


def test_nothing_is_read_before_two_windows_exist_after_the_ramp(tmp_path):
    """23 post-ramp hours cannot fill two 12-hour windows, and the ramp's large
    drops must not fill them instead."""
    log = write_run(tmp_path, web=0.0, code=0.0, math=0.0, hours=33)
    assert stall.ramp_end_step(log) == 1000
    assert stall.check(rule("web"), log)[0] == stall.NOT_YET
    log = write_run(tmp_path / "later", web=0.0, code=0.0, math=0.0, hours=34)
    assert stall.check(rule("web"), log)[0] == stall.STALLED


def test_the_mixture_weights_are_the_runs_end_mix_over_the_sources_read(tmp_path):
    log = write_run(tmp_path, web=0.0, code=0.0, math=0.0)
    (_, ce), *_ = stall.mixture_series(log, ["fineweb-edu", "finemath"])
    # at the ramp's end: web 4.0 - 1.0, math 3.5 - 1.0, weighted 0.35 : 0.25
    assert ce == pytest.approx((0.35 * 3.0 + 0.25 * 2.5) / 0.60)
    with pytest.raises(ValueError, match="no weight"):
        stall.mixture_series(log, ["fineweb-edu", "fineweb"])


@pytest.mark.parametrize("body, message", [
    ('reads = "val"\nmin_gain = 0.01', "not one of"),
    ('reads = "every-source"\nsources = ["fineweb-edu", "finemath"]\nmin_gain = 0.01', "one bar per source"),
    ('reads = "mixture"\nsources = ["fineweb-edu"]\nmin_gain = {fineweb-edu = 0.01}', "one number"),
])
def test_a_rule_without_its_own_bars_does_not_load(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        stall.load_rule(write_spec(tmp_path, body))


def test_reading_the_contaminated_code_probe_says_so(tmp_path):
    log = write_run(tmp_path, web=0.0, code=0.0, math=0.0)
    assert not any("#485" in line for line in stall.report_check(rule("web"), log))
    lines = stall.report_check(rule("mixture"), log)
    assert any("codeparrot probe is contaminated (#485)" in line for line in lines)
    assert lines[-1] == "STALLED: start the 1,000-opt-step decay"


def test_the_floor_reports_every_quantity_a_rule_could_read(tmp_path):
    log = write_run(tmp_path, web=0.0, code=0.002, math=0.001)
    lines = stall.report_floor(rule("web"), log)
    for name in ("fineweb-edu", "codeparrot", "finemath",
                 "mixture(codeparrot+finemath+fineweb-edu)", "mixture(finemath+fineweb-edu)"):
        assert any(line.startswith(name + " ") for line in lines), name
    first, last = stall.floor_row(stall.readings(log, ["codeparrot"])["codeparrot"], 12).values()
    assert first[1] is None or first[1] == pytest.approx(0.024, abs=2e-4)
    assert last[1] == pytest.approx(0.024, abs=2e-4)


def test_the_cli_reads_the_committed_spec(tmp_path, capsys):
    run = write_run(tmp_path, web=0.0, code=0.0, math=0.0).run_dir
    assert stall.main(["--spec", str(SPEC_001), "--run", run]) == 0
    assert "STALLED" in capsys.readouterr().out


def test_a_floor_can_stop_where_the_decay_starts(tmp_path, capsys):
    """A WSD run's last fifth is its decay: read through it, the last third's gains
    are the decay's (0.12 per 12h here), not the plateau's (0)."""
    run = write_run(tmp_path, web=0.0, code=0.0, math=0.0, hours=96, decay_hour=80).run_dir

    def last_third_median():
        out = capsys.readouterr().out
        return next(line.split()[4] for line in out.splitlines() if line.startswith("fineweb-edu ")  # name, "last", "third", noise, median
                    and "last third" in line)
    stall.main(["--spec", str(SPEC_001), "--run", run, "--floor"])
    assert last_third_median() != "+0.0000"
    stall.main(["--spec", str(SPEC_001), "--run", run, "--floor", "--until-step", "8000"])
    assert last_third_median() == "+0.0000"
