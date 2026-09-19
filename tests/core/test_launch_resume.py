"""A base run survives a power cut (#384): the launcher records the run, and
`--resume` brings its supervisor back only when that is the right thing to do."""

from trm.runtime import launch

STATE = {"run_id": "run_20260920_010203", "stop_step": 30528}

HEARTBEAT = """2026-09-20 01:05:00 ▶ supervising pid 4242: python -m trm.train.start
2026-09-20 02:05:00 RUNNING: step 120/30528 (ce=6.1)
2026-09-20 02:10:00 ⚠ MARGIN: act_max 20,000 > 16,384 (a quarter of f16's 65,504)
2026-09-20 03:05:00 RUNNING: step 260/30528 (ce=5.4)
"""


def test_the_last_outcome_skips_alarms_and_notes():
    assert launch.last_outcome(HEARTBEAT) == "RUNNING"
    assert launch.last_outcome(HEARTBEAT + "2026-09-21 09:00:00 BUDGET_COMPLETE: reached opt step\n") == "BUDGET_COMPLETE"
    assert launch.last_outcome("") is None


def test_a_run_cut_mid_flight_is_resumed():
    assert launch.resume_refusal(STATE, "RUNNING", 260, None) == (None, False)
    assert launch.resume_refusal(STATE, None, 0, None) == (None, False), "cut before its first heartbeat"


def test_a_run_that_ended_on_purpose_is_left_alone_for_good():
    for outcome in ("BUDGET_COMPLETE", "KILLED_DIVERGENCE", "KILLED_OOM", "GAVE_UP"):
        why, over = launch.resume_refusal(STATE, outcome, 260, None)
        assert why and over, outcome
    why, over = launch.resume_refusal(STATE, "RUNNING", 30528, None)
    assert "stop step" in why and over


def test_a_held_card_is_a_reason_to_wait_not_an_end():
    why, over = launch.resume_refusal(STATE, "RUNNING", 260, (999, "someone else"))
    assert "held by pid 999" in why and not over


def test_resume_with_nothing_recorded_does_nothing(tmp_path, capsys):
    assert launch.resume(tmp_path / "none.json") == 0
    assert "no active base run" in capsys.readouterr().out
