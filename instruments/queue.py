"""What to work on next, and why: CLAUDE.md's ready-queue as a command.

    python -m instruments.queue

CLAUDE.md states the ready-queue as prose precise enough to execute — an issue
is ready when it is open, not blocked, unclaimed, and its lane is free; types
lead in the order architecture > optimization > tools > ideas > documentation;
anything that affects another item leads. Run in a session's head, that
algorithm runs differently depending on what the session happened to read, and
not at all in a fresh one. Here it runs the same every time.

It ranks only what the rules decide, then stops. Within a tier the rules set no
order (for ideas CLAUDE.md says so outright: "any order, your judgment"), so the
output says that instead of sorting by something that looks like authority. The
one mechanical tiebreak is dependency: an issue other open issues are blocked
by affects another item, so it leads its tier.

It also refuses to trust labels it can check. A `blocked` label whose blockers
are all closed is surfaced as stale, not obeyed; an issue with no type label is
surfaced, not guessed into a tier.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass, field

from trm.runtime.gpu_lock import GpuLock, _pid_alive

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # ranks issues and says why; prints no quantities

TYPE_ORDER = ("architecture", "optimization", "tools", "ideas", "documentation")
UNORDERED = {"ideas": "any order, your judgment, per CLAUDE.md"}

# "Blocked by" alone marks a block. When the first thing after it is an issue number,
# every number in the rest of that sentence names a blocker — bodies annotate each
# one, "**Blocked by #440** (the world), **#441** (a base that can pass)", and a
# pattern that wanted a bare `#a, #b and #c` stopped at the first `**` and read #29
# as unblocked while #441 still held it. When the first thing is prose, the block is
# a condition and a number later in the sentence is a reference, not a blocker:
# "Blocked by: the owner turning it into a question; its tool shipped in #391."
_BLOCKED = r"blocked by"
BLOCKED_ON_CONDITION = re.compile(_BLOCKED, re.I)
BLOCKED_BY = re.compile(_BLOCKED + r"[\s:*_]*(#\d+[^\n.]*)", re.I)
CLOSES = re.compile(r"\b(?:closes|fixes|resolves)\s+#(\d+)", re.I)
TITLE_REF = re.compile(r"\(#(\d+)\)")


@dataclass
class Card:
    free: bool
    why: str


@dataclass
class Entry:
    number: int
    title: str
    tier: str
    labels: set[str]
    unblocks: list[int] = field(default_factory=list)

    def reason(self) -> str:
        lane = self.labels & {"cpu", "gpu"}
        parts = ["bug" if "bug" in self.labels else None,
                 "cpu half (partial-cpu)" if lane == {"cpu", "gpu"}
                 else "/".join(sorted(lane)) if lane else "no lane label"]
        if self.unblocks:
            parts.append("unblocks " + ", ".join(f"#{n}" for n in self.unblocks))
        return " · ".join(p for p in parts if p)


@dataclass
class Queue:
    card: Card
    tiers: dict[str, list[Entry]]
    not_ready: list[tuple[int, str]]
    needs_human: list[tuple[int, str]]

    def next_step(self) -> str:
        for tier in TYPE_ORDER:
            entries = self.tiers.get(tier)
            if not entries:
                continue
            lead = entries[0]
            if len(entries) == 1 or lead.unblocks:
                return f"#{lead.number} — first ready {tier} issue ({lead.reason()})"
            if self.card.free and "gpu" in lead.labels:
                return f"#{lead.number} — card idle → gpu items first; first ready {tier} issue ({lead.reason()})"
            return (f"a {tier} issue — {len(entries)} are ready and the rules do not "
                    f"order them; that pick is a judgment call")
        return "nothing is ready"


def blockers_named(body: str) -> list[int]:
    return sorted({int(n) for m in BLOCKED_BY.finditer(body or "")
                   for n in re.findall(r"#(\d+)", m.group(1))})


def claimed_by_pr(prs: list[dict]) -> dict[int, int]:
    """Issue number → the open PR that addresses it. A draft counts: CLAUDE.md
    parks partial-cpu work as a draft precisely to keep it out of the queue."""
    claims = {}
    for pr in prs:
        text = f"{pr.get('title', '')}\n{pr.get('body', '')}"
        for n in CLOSES.findall(text) + TITLE_REF.findall(pr.get("title", "")):
            claims.setdefault(int(n), pr["number"])
    return claims


def build_queue(issues: list[dict], prs: list[dict], card: Card) -> Queue:
    """Pure: the open issues, the open PRs, and the card's state in; the queue out.
    Every issue passed in is taken to be open, so a blocker absent from the list
    is a closed one."""
    open_numbers = {i["number"] for i in issues}
    claims = claimed_by_pr(prs)
    dependents: dict[int, list[int]] = {}
    for issue in issues:
        for n in blockers_named(issue.get("body", "")):
            if n in open_numbers:
                dependents.setdefault(n, []).append(issue["number"])

    tiers: dict[str, list[Entry]] = {t: [] for t in TYPE_ORDER}
    not_ready, needs_human = [], []
    for issue in sorted(issues, key=lambda i: i["number"]):
        n, labels = issue["number"], {lb["name"] for lb in issue.get("labels", [])}
        body = issue.get("body") or ""
        named = blockers_named(body)
        live_blockers = [b for b in named if b in open_numbers]

        on_condition = not named and BLOCKED_ON_CONDITION.search(body)
        if "blocked" in labels and not named and not on_condition:
            needs_human.append((n, "labelled blocked but says nothing about what blocks it"))
        elif "blocked" in labels and named and not live_blockers:
            needs_human.append((n, "stale block — every blocker it names is closed ("
                                + ", ".join(f"#{b}" for b in named) + ")"))
        elif live_blockers and "blocked" not in labels:
            needs_human.append((n, "says blocked by open "
                                + ", ".join(f"#{b}" for b in live_blockers)
                                + " but carries no blocked label"))
        tier = next((t for t in TYPE_ORDER if t in labels), None)
        if tier is None:
            needs_human.append((n, "no type label — cannot be placed in a tier"))

        if issue.get("assignees"):
            not_ready.append((n, "claimed by " + ", ".join(a["login"] for a in issue["assignees"])))
        elif n in claims:
            not_ready.append((n, f"claimed by open PR #{claims[n]}"))
        elif "plan" in labels:
            not_ready.append((n, "plan — not ready to start"))
        elif live_blockers:
            not_ready.append((n, "blocked by open " + ", ".join(f"#{b}" for b in live_blockers)))
        elif on_condition and "blocked" in labels:
            not_ready.append((n, "blocked on a condition, not an issue — the queue cannot check it"))
        elif "blocked" in labels:
            continue  # surfaced above; a label that fails its own check is not obeyed
        elif "gpu" in labels and "cpu" not in labels and not card.free:
            not_ready.append((n, f"gpu lane, card busy ({card.why})"))
        elif tier is not None:
            tiers[tier].append(Entry(n, issue["title"], tier, labels, sorted(dependents.get(n, []))))

    for entries in tiers.values():
        # Unblockers lead. On an idle card, gpu-lane items lead too (#295): the
        # card is the scarce resource, and an idle one is the waste to end first.
        entries.sort(key=lambda e: (-len(e.unblocks), not (card.free and "gpu" in e.labels), e.number))
    return Queue(card, tiers, not_ready, needs_human)


def card_state() -> Card:
    holder = GpuLock().holder()
    if holder and _pid_alive(holder[0]):
        return Card(False, f"GPU lock held by pid {holder[0]} {holder[1]}".strip())
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return Card(False, f"cannot read nvidia-smi ({exc.__class__.__name__}) — assuming busy")
    blocking, other = split_compute_processes(out.splitlines())
    if blocking:
        return Card(False, "compute processes on the card: " + "; ".join(blocking))
    if other:
        return Card(True, "no lock, no training process (also on the card, not counted: "
                          + "; ".join(other) + ")")
    return Card(True, "no lock, no compute processes")


def split_compute_processes(lines: list[str]) -> tuple[list[str], list[str]]:
    """(processes that hold the card, the rest), from nvidia-smi's `pid, name` lines.

    Only a Python process can be a training run, a smoke or another model using the
    card in earnest; a browser's GPU process rasterizing a page is not (#389: the
    stark-hud board's headless Chromium held ~300 MiB and marked an idle card busy,
    so the queue stopped putting gpu items first). The others are still listed."""
    blocking, other = [], []
    for line in (entry.strip() for entry in lines):
        if not line:
            continue
        name = line.split(",", 1)[-1].strip()
        (blocking if "python" in pathlib.Path(name).name else other).append(line)
    return blocking, other


STRAY_AFTER_DAYS = 7


def stray_branches(branches: list[dict], open_pr_heads: set[str], now: datetime.datetime,
                   after_days: int = STRAY_AFTER_DAYS) -> list[tuple[str, int]]:
    """[(branch, days since its last commit)] for pushed work nobody can find.

    #74 was finished — wiring, three seeds, a written finding — and pushed on
    2026-07-05 to a branch that never got a PR, a comment or an assignee. From the
    outside it was indistinguishable from an issue nobody had started, for ten
    weeks. A branch is stray once its newest commit is older than `after_days` and
    no open PR has it as head; younger ones are work in progress.
    `branches` are {"name", "committed_at": ISO-8601}.
    """
    found = []
    for b in branches:
        if b["name"] == "main" or b["name"] in open_pr_heads:
            continue
        committed = datetime.datetime.fromisoformat(b["committed_at"].replace("Z", "+00:00"))
        days = (now - committed).days
        if days >= after_days:
            found.append((b["name"], days))
    return sorted(found, key=lambda x: -x[1])


def remote_branches() -> list[dict]:
    """Every branch on origin with its last commit date, from the GitHub API."""
    rows = gh_json("api", "--paginate", "repos/{owner}/{repo}/branches?per_page=100")
    out = []
    for row in rows:
        commit = gh_json("api", f"repos/{{owner}}/{{repo}}/commits/{row['commit']['sha']}")
        out.append({"name": row["name"], "committed_at": commit["commit"]["committer"]["date"]})
    return out


def gh_json(*args: str) -> list[dict]:
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True,
                                     check=True).stdout)


def render(q: Queue) -> str:
    lines = [f"card: {'free' if q.card.free else 'busy'} — {q.card.why}", "",
             f"NEXT: {q.next_step()}", ""]
    for tier in TYPE_ORDER:
        entries = q.tiers[tier]
        if not entries:
            continue
        note = UNORDERED.get(tier, "rules set no order within a tier; unblockers lead")
        if q.card.free and any("gpu" in e.labels for e in entries):
            note += "; card idle → gpu items first"
        lines.append(f"{tier}  ({note})")
        lines += [f"  #{e.number:<4} {e.title[:72]}\n        {e.reason()}" for e in entries]
        lines.append("")
    for heading, rows in (("NEEDS A HUMAN", q.needs_human), ("not ready", q.not_ready)):
        if rows:
            lines.append(heading)
            lines += [f"  #{n:<4} {why}" for n, why in rows]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strays", action="store_true",
                    help=f"only list branches with no open PR and no commit in {STRAY_AFTER_DAYS} days; "
                         "exit 1 if any (CI runs this on every push)")
    args = ap.parse_args(argv)
    prs = gh_json("pr", "list", "--state", "open", "--limit", "200", "--json", "number,title,body,headRefName")
    strays = stray_branches(remote_branches(), {p["headRefName"] for p in prs},
                            datetime.datetime.now(datetime.timezone.utc))
    if args.strays:
        for name, days in strays:
            print(f"stray branch: {name} — last commit {days} days ago, no open PR. Open a PR "
                  f"(a draft counts), or delete it; a pushed branch nobody can find is #74 again.")
        return 1 if strays else 0
    issues = gh_json("issue", "list", "--state", "open", "--limit", "500",
                     "--json", "number,title,labels,assignees,body")
    q = build_queue(issues, prs, card_state())
    q.needs_human += [(0, f"stray branch {name}: last commit {days} days ago, no open PR") for name, days in strays]
    print(render(q), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
