"""The CE plateau must not be able to end a pretraining run.

On 2026-08-15 it did. The #157 base run reached opt step 5,055 of 30,518, the
plateau detector fired, and the trainer flipped to the SFT chat phase — new data
mixture, LR cut to 10%, CE from 3.31 to 8.57 — then OOM'd recreating the
optimizer. Forty-two hours of card, ended by a monitor doing exactly what it was
written to do.

The plateau was real: held-out val CE was flat at ~4.06 as well, so this is not a
story about a miscalibrated threshold. It is about what a flat loss curve
*means*. At 16% of budget with the LR still at 9.9e-5 — the cosine has barely
begun — a model that has stopped improving has not finished learning; it cannot
descend further until the anneal lets it. The detector cannot tell those apart,
and it was wired to a switch that reshapes the run.

These tests pin the two halves of the fix: the flip is off unless asked for, and
the log line the supervisor kills on is never printed unless a flip really
happened.
"""

import re

import pytest

from trm import config


PLATEAU_KILL_PHRASE = "CE Plateau Detected"


def _trainer_source():
    from pathlib import Path
    return Path(config.__file__).parent.joinpath("train", "trainer.py").read_text()


def test_the_sft_autoflip_is_off_by_default():
    """A run whose product is a base model must never wander into chat tuning.
    Unset means off; only an explicit opt-in turns it on."""
    assert config.SFT_ON_PLATEAU is False


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("", False)])
def test_the_autoflip_is_opt_in_by_env(monkeypatch, value, expected):
    monkeypatch.setenv("SFT_ON_PLATEAU", value)
    import importlib
    reloaded = importlib.reload(config)
    try:
        assert reloaded.SFT_ON_PLATEAU is expected
    finally:
        monkeypatch.delenv("SFT_ON_PLATEAU", raising=False)
        importlib.reload(config)


def test_a_plateau_is_reported_without_the_phrase_the_supervisor_kills_on():
    """The supervisor greps the log for "CE Plateau Detected" and kills the run on
    sight, to stop a pretrain silently becoming a chat model. So the *informational*
    notice — the one printed when the flip is disabled and training continues — must
    not contain that phrase, or reporting a plateau would kill the run it was meant
    to let survive.

    The phrase stays reserved for a genuine flip, which keeps the supervisor's guard
    working as a backstop for anyone who sets SFT_ON_PLATEAU=1 on a pretrain.
    """
    source = _trainer_source()

    notice = re.search(r'print\(f?"📉 \[Plateau\][^)]*\)', source, re.S)
    assert notice, "the plateau notice should still exist — a plateau is worth reporting"
    assert PLATEAU_KILL_PHRASE not in notice.group(0), (
        "the informational notice must not contain the phrase the supervisor kills on"
    )

    # And the phrase must still exist somewhere, guarding a real flip.
    assert PLATEAU_KILL_PHRASE in source, (
        "an actual SFT flip must still announce itself so the supervisor can catch it"
    )


def test_the_flip_is_gated_on_the_flag_not_only_on_the_plateau():
    """Guards the shape of the fix, not just its presence: the early-out has to
    consult SFT_ON_PLATEAU, otherwise a future edit could restore the old behaviour
    while leaving the flag defined and apparently honoured."""
    source = _trainer_source()
    assert "not SFT_ON_PLATEAU" in source, (
        "the plateau branch must check the flag before flipping"
    )


def test_the_supervisor_still_kills_on_a_real_flip():
    """The backstop must stay wired: if someone runs a pretrain with the flip on,
    the supervisor still ends it rather than letting the artifact be contaminated."""
    from trm.runtime.supervisor import KILL, KILLED_PLATEAU, Limits, Observation, State, decide

    obs = Observation(step=5055, ce=3.31, plateau_detected=True, alive=True)
    decision = decide(obs, Limits(stop_step=30518), State(entered_band=True))
    assert (decision.action, decision.outcome) == (KILL, KILLED_PLATEAU)


def test_the_notice_is_rate_limited():
    """`monitor.push` keeps returning True on every step until CE improves, so an
    unthrottled notice would print thousands of times and bury the log."""
    from trm.train.trainer import PLATEAU_NOTICE_EVERY

    assert PLATEAU_NOTICE_EVERY >= 50, "notice should be occasional, not per-step"
    source = _trainer_source()
    assert "last_plateau_notice" in source, "the notice must track when it last fired"
