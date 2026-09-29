"""The ready-queue ranks what CLAUDE.md's rules decide, and nothing more.

Built from fixtures, not from `gh`: what is under test is the reading of the
rules, and a test that needed the network would test GitHub's uptime instead.
"""

from instruments.queue import CLOUD_CARD, Card, blockers_named, build_queue, claimed_by_pr

FREE, BUSY = Card(True, "free"), Card(False, "held by a run")


def issue(n, *labels, body="", assignee=None, title=None, updated="2026-09-27T00:00:00Z"):
    return {"number": n, "title": title or f"issue {n}", "body": body,
            "labels": [{"name": lb} for lb in labels],
            "assignees": [{"login": assignee}] if assignee else [], "updatedAt": updated}


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
    assert list(ranked(q)) == ["architecture", "optimization", "tools", "ideas", "documentation"]


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


def test_a_block_on_an_open_pr_holds_and_is_not_called_stale():
    """#476 named draft PR #418 as its blocker and was reported a stale block: an open
    PR was read as closed because only open issues counted (#480)."""
    q = build_queue([issue(476, "tools", "cpu", "blocked", body="Blocked by #418")],
                    [{"number": 418, "title": "#411 stage 1", "body": "", "isDraft": True}], FREE)
    assert 476 not in flagged(q) and 476 in excluded(q)
    assert dict(q.not_ready)[476] == "blocked by open PR #418"
    assert "stale block" in flagged(build_queue([issue(476, "tools", "cpu", "blocked",
                                                       body="Blocked by #418")], [], FREE))[476]


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


def test_blockers_with_a_note_beside_each_are_all_read():
    """#29's own body. Reading only the first number called it unblocked the moment
    #440 closed, with #441 still open and still in its way."""
    body = ("**Blocked by #440** (the world), **#441** (a base whose initial pass rate "
            "is non-zero), **#302** (the KV cache: rollouts dominate RL compute).\n"
            "## What gets built (once unblocked)\nSee #12 for the history.")
    assert blockers_named(body) == [302, 440, 441]
    issues = [issue(29, "ideas", "gpu", "blocked", body=body), issue(441, "ideas", "gpu", "blocked")]
    queue = build_queue(issues, [], FREE)
    assert not any(n == 29 for n, _ in queue.needs_human), "#441 is open: the block is live"


def test_a_condition_that_mentions_an_issue_is_still_a_condition():
    """#127's and #292's bodies. Reading every number in the sentence would call
    both stale blocks — their references are closed — when what holds them is a
    condition no issue can close."""
    assert blockers_named("**Blocked by:** the owner turning the hypothesis into a "
                          "pre-registered question; its tool shipped in #391.") == []
    assert blockers_named("Blocked by: a `plain` base run superseding the champion "
                          "(no issue for that run exists yet; it follows the #287 LR pair)") == []


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
    assert ranked(busy) == {"tools": [1, 3]}, "gpu-only waits; the rest keep the plain order"
    assert "card idle" not in busy.next_step()


def test_unblockers_still_outrank_the_idle_card_rule():
    issues = [issue(1, "tools", "cpu"), issue(2, "tools", "gpu"),
              issue(3, "tools", "cpu", "blocked", body="Blocked by #1")]
    assert ranked(build_queue(issues, [], FREE))["tools"][0] == 1


def test_a_browsers_gpu_process_does_not_hold_the_card():
    """#389: the board's headless Chromium marked an idle card busy."""
    from instruments.queue import split_compute_processes

    blocking, other = split_compute_processes([
        "208451, /usr/lib/chromium/chromium",
        "226159, /mnt/d_drive/models/.venv-kokoro/bin/python",
        "1170676, /home/lendacerda/Desktop/Repos/TinyRefinementModel/venv/bin/python3.14",
        "",
    ])
    assert blocking == ["226159, /mnt/d_drive/models/.venv-kokoro/bin/python",
                        "1170676, /home/lendacerda/Desktop/Repos/TinyRefinementModel/venv/bin/python3.14"]
    assert other == ["208451, /usr/lib/chromium/chromium"]


def test_a_cloud_session_gets_neither_the_card_nor_this_machines_files():
    """#492: a cloud session can finish neither a gpu-only item nor one that needs the
    weights, corpus or HDD here; it keeps the cpu half of a partial-cpu issue."""
    issues = [issue(1, "tools", "cpu"), issue(2, "tools", "gpu"), issue(3, "tools", "cpu", "gpu"),
              issue(4, "tools", "cpu", "local"), issue(5, "ideas", "cpu", "gpu", "local")]
    q = build_queue(issues, [], CLOUD_CARD, cloud=True)
    assert ranked(q) == {"tools": [1, 3]}
    assert excluded(q) == {2, 4, 5}


def test_local_means_nothing_to_a_session_on_this_machine():
    q = build_queue([issue(1, "tools", "cpu", "local")], [], FREE)
    assert ranked(q) == {"tools": [1]}


# --- issues nobody has touched in weeks (#482) ------------------------------------

def test_an_untouched_issue_is_surfaced_unless_it_is_legitimately_waiting():
    import datetime
    from instruments.queue import STALE_DAYS
    now = datetime.datetime(2026, 9, 28, tzinfo=datetime.timezone.utc)
    old, fresh = "2026-08-01T00:00:00Z", "2026-09-20T00:00:00Z"   # 58 and 8 days
    card_draft = {"number": 90, "title": "the cpu half (#4)", "isDraft": True, "updatedAt": old,
                  "body": "## What changed\nwiring\n\n## What waits on the card (resume protocol)\nrun it"}
    idle_draft = {"number": 91, "title": "a sketch (#5)", "isDraft": True, "updatedAt": old,
                  "body": "## What waits\nthe owner's call\n\n## Notes\nnot on the GPU yet"}
    busy_pr = {"number": 92, "title": "work in flight (#6)", "isDraft": False,
               "updatedAt": fresh, "body": ""}
    prs = [card_draft, idle_draft, busy_pr]
    issues = [issue(1, "tools", "cpu", updated=old),                        # stale
              issue(2, "tools", "cpu", updated=fresh),                      # fresh
              issue(3, "tools", "cpu", "blocked", body="Blocked by #2", updated=old),
              issue(4, "tools", "cpu", "gpu", updated=old),                 # parked on the card
              issue(5, "ideas", "cpu", updated=old),                        # its draft waits on nobody
              issue(6, "tools", "cpu", updated=old),                        # its PR moved last week
              issue(7, "ideas", "cpu", "blocked", body="Blocked by: a condition", updated=old)]
    q = build_queue(issues, prs, FREE, now=now)
    stale = {n for n, why in q.needs_human if why.startswith("untouched")}
    assert stale == {1, 5, 7}, "a condition no issue can close is exactly what rots unasked"
    assert "58 days" in flagged(q)[1] and "keep it (comment why) or close" in flagged(q)[1]
    assert not any(why.startswith("untouched") for _, why in build_queue(issues, prs, FREE).needs_human), \
        "without `now` the check is off"
    at_edge = issue(8, "tools", "cpu", updated=(now - datetime.timedelta(days=STALE_DAYS)).isoformat())
    assert 8 not in flagged(build_queue([at_edge], [], FREE, now=now)), "more than STALE_DAYS, not at it"
