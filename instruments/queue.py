"""What to work on next, and why: CLAUDE.md's ready-queue as a command.

    python -m instruments.queue

CLAUDE.md states the ready-queue as prose precise enough to execute — an issue
is ready when it is open, not blocked, unclaimed, and its lane is free; types
lead in the order architecture > tools > ideas > optimization > documentation;
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
import json
import re
import subprocess
from dataclasses import dataclass, field

from trm.runtime.supervisor import GpuLock, _pid_alive

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # ranks issues and says why; prints no quantities

TYPE_ORDER = ("architecture", "tools", "ideas", "optimization", "documentation")
UNORDERED = {"ideas": "any order, your judgment, per CLAUDE.md"}

BLOCKED_BY = re.compile(r"blocked by\s+((?:#\d+(?:\s*(?:,|and|&|/)\s*)?)+)", re.I)
BLOCKED_ON_CONDITION = re.compile(r"blocked by", re.I)
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
        live_blockers = [b for b in blockers_named(issue.get("body", "")) if b in open_numbers]
        named = blockers_named(issue.get("body", ""))

        on_condition = not named and BLOCKED_ON_CONDITION.search(issue.get("body") or "")
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
        entries.sort(key=lambda e: (-len(e.unblocks), e.number))
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
    if out:
        return Card(False, "compute processes on the card: " + "; ".join(out.splitlines()))
    return Card(True, "no lock, no compute processes")


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
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    issues = gh_json("issue", "list", "--state", "open", "--limit", "500",
                     "--json", "number,title,labels,assignees,body")
    prs = gh_json("pr", "list", "--state", "open", "--limit", "200", "--json", "number,title,body")
    print(render(build_queue(issues, prs, card_state())), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
