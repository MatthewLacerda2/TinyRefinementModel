"""The ready-queue ranks what CLAUDE.md's rules decide, and nothing more.

Built from fixtures, not from `gh`: what is under test is the reading of the
rules, and a test that needed the network would test GitHub's uptime instead.
"""

from instruments.queue import Card, blockers_named, build_queue, claimed_by_pr

FREE, BUSY = Card(True, "free"), Card(False, "held by a run")


def issue(n, *labels, body="", assignee=None, title=None):
    return {"number": n, "title": title or f"issue {n}", "body": body,
            "labels": [{"name": lb} for lb in labels],
            "assignees": [{"login": assignee}] if assignee else []}


def ranked(q):
    return {t: [e.number for e in es] for t, es in q.tiers.items() if es}


def excluded(q):
    return {n for n, _ in q.not_ready}


def flagged(q):
    return {n: why for n, why in q.needs_human}


def test_types_lead_in_claude_md_order():
    q = build_queue([issue(1, "documentation", "cpu"), issue(2, "ideas", "cpu"),
                     issue(3, "optimization", "cpu"), issue(4, "tools", "cpu"),
                     issue(5, "architecture", "cpu")], [], FREE)
    assert list(ranked(q)) == ["architecture", "tools", "ideas", "optimization", "documentation"]


def test_an_issue_with_two_types_sits_in_the_higher_one():
    assert ranked(build_queue([issue(1, "ideas", "tools", "cpu")], [], FREE)) == {"tools": [1]}


def test_claimed_work_is_not_ready_whether_by_assignee_or_by_an_open_pr():
    prs = [{"number": 90, "title": "Draft of the thing", "body": "Closes #2"},
           {"number": 91, "title": "Some fix (#3)", "body": ""}]
    q = build_queue([issue(1, "tools", "cpu", assignee="someone"),
                     issue(2, "tools", "cpu"), issue(3, "tools", "cpu")], prs, FREE)
    assert excluded(q) == {1, 2, 3} and not ranked(q)


def test_a_plan_is_not_ready_to_start():
    assert excluded(build_queue([issue(1, "ideas", "cpu", "plan")], [], FREE)) == {1}


def test_an_open_blocker_holds_the_issue_back_and_its_blocker_leads():
    q = build_queue([issue(1, "tools", "cpu"),
                     issue(2, "tools", "cpu"),
                     issue(3, "tools", "cpu", "blocked", body="Blocked by #2.")], [], FREE)
    assert excluded(q) == {3}
    assert ranked(q) == {"tools": [2, 1]}, "an issue others wait on affects another item"
    assert q.tiers["tools"][0].unblocks == [3]


def test_a_block_whose_blockers_all_closed_is_surfaced_not_obeyed_or_trusted():
    q = build_queue([issue(5, "tools", "cpu", "blocked", body="Blocked by #4")], [], FREE)
    assert "stale block" in flagged(q)[5]
    assert not ranked(q), "a stale label is a question for a human, not a green light"


def test_a_block_on_a_condition_holds_but_an_unexplained_one_is_flagged():
    q = build_queue([issue(1, "ideas", "gpu", "blocked", body="**Blocked by:** a base worth aligning"),
                     issue(2, "ideas", "gpu", "blocked", body="no reason given")], [], FREE)
    assert 1 in excluded(q) and 1 not in flagged(q)
    assert 2 in flagged(q)


def test_an_open_blocker_without_the_label_is_still_a_block_and_is_flagged():
    q = build_queue([issue(1, "tools", "cpu"), issue(2, "tools", "cpu", body="Blocked by #1")], [], FREE)
    assert 2 in excluded(q) and "no blocked label" in flagged(q)[2]


def test_a_busy_card_holds_the_gpu_lane_but_not_the_cpu_half():
    q = build_queue([issue(1, "tools", "gpu"), issue(2, "tools", "gpu", "cpu")], [], BUSY)
    assert excluded(q) == {1} and ranked(q) == {"tools": [2]}
    assert ranked(build_queue([issue(1, "tools", "gpu")], [], FREE)) == {"tools": [1]}


def test_an_untyped_issue_is_surfaced_not_guessed_into_a_tier():
    q = build_queue([issue(1, "bug", "gpu")], [], FREE)
    assert "no type label" in flagged(q)[1] and not ranked(q)


def test_next_step_names_an_issue_only_when_the_rules_decide_it():
    assert build_queue([issue(1, "tools", "cpu"), issue(2, "ideas", "cpu")], [], FREE
                       ).next_step().startswith("#1")
    tie = build_queue([issue(1, "ideas", "cpu"), issue(2, "ideas", "cpu")], [], FREE).next_step()
    assert "judgment" in tie and "#" not in tie
    assert build_queue([], [], FREE).next_step() == "nothing is ready"


def test_blocker_references_parse_lists():
    assert blockers_named("Blocked by #12, #3 and #40. Also see #99.") == [3, 12, 40]
    assert blockers_named("blocked by a condition") == []


def test_a_pr_claims_the_issue_it_closes():
    assert claimed_by_pr([{"number": 7, "title": "x", "body": "Fixes #3\nresolves #4"}]) == {3: 7, 4: 7}


# --- pushed work nobody can find (#74's failure) -------------------------------

def test_a_branch_with_old_commits_and_no_pr_is_stray_and_a_fresh_or_pr_backed_one_is_not():
    import datetime
    from instruments.queue import stray_branches
    now = datetime.datetime(2026, 9, 14, tzinfo=datetime.timezone.utc)
    branches = [
        {"name": "main", "committed_at": "2026-05-01T00:00:00Z"},
        {"name": "claude/nice-hypatia", "committed_at": "2026-07-05T12:00:00Z"},   # #74, 71 days
        {"name": "wip-today", "committed_at": "2026-09-12T12:00:00Z"},            # in progress
        {"name": "has-a-pr", "committed_at": "2026-06-01T00:00:00Z"},
    ]
    assert stray_branches(branches, {"has-a-pr"}, now) == [("claude/nice-hypatia", 70)]
    assert stray_branches(branches, {"has-a-pr", "claude/nice-hypatia"}, now) == []


# --- an idle card puts gpu-lane items first (#295) --------------------------------

def test_an_idle_card_leads_with_gpu_items_and_a_busy_one_does_not():
    issues = [issue(1, "tools", "cpu"), issue(2, "tools", "gpu"), issue(3, "tools", "gpu", "cpu")]
    idle = build_queue(issues, [], FREE)
    assert ranked(idle) == {"tools": [2, 3, 1]}
    assert "card idle" in idle.next_step() and idle.next_step().startswith("#2")
    busy = build_queue(issues, [], BUSY)
    assert ranked(busy) == {"tools": [3, 1]}, "gpu-only waits; the cpu half of a partial-cpu item is ready"
    assert "card idle" not in busy.next_step()


def test_unblockers_still_outrank_the_idle_card_rule():
    issues = [issue(1, "tools", "cpu"), issue(2, "tools", "gpu"),
              issue(3, "tools", "cpu", "blocked", body="Blocked by #1")]
    assert ranked(build_queue(issues, [], FREE))["tools"][0] == 1
