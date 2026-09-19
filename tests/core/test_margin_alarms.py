"""The supervisor watches the f16 margins during a run (#368).

Every failure these warn of was visible in real time on the champion and nobody was
watching: activations at 65,120 of f16's 65,504 (#235), a loss scale pinned at 1
while the backward overflowed anyway (#199)."""

from trm.runtime.supervisor import margin_alarms, margin_changes

HEALTHY = {"act_max": "45.2", "loss_scale": "131072", "applied_zero_frac_dense_max": "0.000030",
           "arena_peak_mib": "4058", "arena_limit_mib": "4883"}


def kinds(row):
    return [kind for kind, _ in margin_alarms(row)]


def test_a_healthy_row_raises_nothing():
    assert margin_alarms(HEALTHY) == ()


def test_each_margin_fires_on_its_own_line():
    assert kinds({**HEALTHY, "act_max": "20000"}) == ["act_max"]
    assert kinds({**HEALTHY, "loss_scale": "2"}) == ["loss_scale"]
    assert kinds({**HEALTHY, "applied_zero_frac_dense_max": "0.6"}) == ["zero_grad"]
    assert kinds({**HEALTHY, "arena_peak_mib": "4800"}) == ["vram"]


def test_a_column_the_run_does_not_log_is_not_checked():
    """Older runs and CPU runs leave these blank; blank is not a crossing."""
    assert margin_alarms({"act_max": "", "loss_scale": "", "arena_peak_mib": "4800"}) == ()


def test_an_alarm_is_announced_once_when_it_appears_and_once_when_it_clears():
    first = margin_alarms({**HEALTHY, "act_max": "20000"})
    lines = margin_changes({}, first)
    assert len(lines) == 1 and lines[0].startswith("⚠ MARGIN: act_max 20,000")

    raised = dict(first)
    climbing = margin_alarms({**HEALTHY, "act_max": "30000"})
    assert margin_changes(raised, climbing) == [], "held for days, it must not announce every poll"

    cleared = margin_changes(raised, margin_alarms(HEALTHY))
    assert len(cleared) == 1 and cleared[0].startswith("margin cleared: act_max")
