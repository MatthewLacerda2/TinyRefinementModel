"""`make ready` must wait for the run the undraft started, never the skipped draft run (#539).

The fixture is PR #538's real `gh run list` output for one head SHA: the draft push's
runs (concluded `skipped`, 08:43:48), the runs the undraft triggered four seconds later,
and a pair from a later event on the same SHA.
"""

import pytest

from instruments.ci_wait import expected_head, select_runs, verdict

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the seconds-long
# lint job instead of the jax-heavy pytest job (#325).
pytestmark = pytest.mark.jaxfree

SHA = "62022ae637fd5f8be87d2da3c8cf07acb5d7578f"
DRAFT = [
    {"conclusion": "skipped", "createdAt": "2026-10-03T08:43:48Z", "databaseId": 37110671716,
     "event": "pull_request", "headSha": SHA, "status": "completed", "workflowName": "Review Gate"},
    {"conclusion": "skipped", "createdAt": "2026-10-03T08:43:48Z", "databaseId": 37110671736,
     "event": "pull_request", "headSha": SHA, "status": "completed", "workflowName": "CI"},
]
UNDRAFT = [
    {"conclusion": "", "createdAt": "2026-10-03T08:43:52Z", "databaseId": 37110674499,
     "event": "pull_request", "headSha": SHA, "status": "in_progress", "workflowName": "CI"},
    {"conclusion": "", "createdAt": "2026-10-03T08:43:52Z", "databaseId": 37110674508,
     "event": "pull_request", "headSha": SHA, "status": "queued", "workflowName": "Review Gate"},
]
OTHER = [  # another SHA and a push to main: never this PR's
    {"conclusion": "success", "createdAt": "2026-10-03T11:25:39Z", "databaseId": 37119617383,
     "event": "pull_request", "headSha": "2827a765d97c", "status": "completed", "workflowName": "CI"},
    {"conclusion": "failure", "createdAt": "2026-10-03T11:24:12Z", "databaseId": 37119536350,
     "event": "push", "headSha": SHA, "status": "completed", "workflowName": "CI"},
]


def ids(runs):
    return {r["databaseId"] for r in runs}


def test_before_the_undraft_run_exists_nothing_is_selected():
    # The moment after `gh pr ready`: only the skipped draft runs exist. Reading them as
    # the result is the exact mistake behind the empty "retrigger" commits.
    assert select_runs(DRAFT + OTHER, SHA) == []


def test_the_post_undraft_run_is_selected_over_the_skipped_one():
    assert ids(select_runs(DRAFT + UNDRAFT + OTHER, SHA)) == {37110674499, 37110674508}


def test_runs_from_before_the_event_are_excluded_even_if_not_skipped():
    later = [dict(r, databaseId=r["databaseId"] + 10**6, createdAt="2026-10-03T11:25:37Z") for r in UNDRAFT]
    before = frozenset(ids(DRAFT + UNDRAFT))
    assert ids(select_runs(DRAFT + UNDRAFT + later, SHA, before)) == ids(later)
    assert select_runs(DRAFT + UNDRAFT, SHA, before) == []


def test_newest_run_per_workflow_wins():
    later = [dict(r, databaseId=r["databaseId"] + 10**6, createdAt="2026-10-03T11:25:37Z") for r in UNDRAFT]
    assert ids(select_runs(UNDRAFT + later, SHA)) == ids(later)


def test_verdict_fails_on_any_failed_job_and_passes_skipped_ones():
    ok = {"CI": [{"name": "lint", "conclusion": "success"}, {"name": "golden", "conclusion": "skipped"}]}
    assert verdict(ok)[0]
    bad = dict(ok, **{"Review Gate": [{"name": "review", "conclusion": "failure"}]})
    passed, lines = verdict(bad)
    assert not passed and any("FAILED" in line for line in lines)


def test_after_a_push_the_local_commit_is_the_head_to_wait_on():
    # The first real use of `ci-wait` right after a push read the previous head's finished
    # runs: GitHub had not yet moved the PR's head to the pushed commit.
    assert expected_head("tools/x", "tools/x", "new") == "new"
    assert expected_head("tools/x", "main", "other") is None  # not this PR's branch: trust GitHub
    assert expected_head("tools/x", None, None) is None  # not in a checkout
