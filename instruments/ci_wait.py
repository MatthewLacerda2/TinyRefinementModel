"""Wait for the CI run a PR event actually triggered, and print how its jobs ended (#539).

    python -m instruments.ci_wait --ready 123    undraft #123, then wait for the run that started
    python -m instruments.ci_wait 123            wait for CI on #123's current head

(`make ready PR=123` and `make ci-wait PR=123` are the same two commands.)

Why it exists: CI skips drafts and re-triggers on `ready_for_review` (ci.yml's trigger
policy, #110). The draft push leaves a run whose jobs were all skipped, and the real run
starts a few seconds after the undraft. A session that read the checks right away saw
the skipped set, concluded CI does not run on undraft, and pushed empty "retrigger"
commits. So this never trusts whatever check set happens to exist: a run GitHub concluded
`skipped` is the draft guard firing and is never selected, and in `--ready` mode only a
run that did not exist before the undraft counts.

Exit status: 0 when every selected run's jobs ended success/skipped/neutral, 1 when any
failed, 2 when no run appeared (a markdown-only diff skips CI by `paths-ignore`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {}  # relays GitHub's job conclusions; prints no quantities

RUN_FIELDS = "databaseId,event,headSha,createdAt,status,conclusion,workflowName"
OK_CONCLUSIONS = {"success", "skipped", "neutral"}
POLL_S = 5
# How long to keep looking for sibling workflows (CI, Review Gate) once the first new run
# shows up; they are created within seconds of each other (#539: 08:43:48 vs 08:43:52).
SETTLE_S = 15
# How long to wait for any run at all before saying none came.
APPEAR_S = 120


def select_runs(runs: list[dict], sha: str, exclude: frozenset[int] = frozenset()) -> list[dict]:
    """The newest `pull_request` run per workflow for `sha` that actually tested something.

    A run concluded `skipped` is a draft run (every job's draft guard fired). `exclude` is
    the set of runs that existed before the event being waited on, so a run from before it
    can never be picked, whatever the clocks say.
    """
    latest: dict[str, dict] = {}
    for run in runs:
        if (run["headSha"] != sha or run["event"] != "pull_request" or run["conclusion"] == "skipped"
                or run["databaseId"] in exclude):
            continue
        held = latest.get(run["workflowName"])
        if held is None or (run["createdAt"], run["databaseId"]) > (held["createdAt"], held["databaseId"]):
            latest[run["workflowName"]] = run
    return sorted(latest.values(), key=lambda r: r["workflowName"])


def verdict(jobs_by_run: dict[str, list[dict]]) -> tuple[bool, list[str]]:
    """(passed, report lines) over the finished runs' jobs."""
    lines, passed = [], True
    for workflow, jobs in jobs_by_run.items():
        for job in jobs:
            ok = job["conclusion"] in OK_CONCLUSIONS
            passed &= ok
            lines.append(f"{workflow} / {job['name']}: {job['conclusion']}{'' if ok else '  <-- FAILED'}")
    return passed, lines


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def pr_head(pr: int) -> tuple[str, bool]:
    """(head sha, is draft), through REST, which every session can reach."""
    pull = json.loads(gh("api", f"repos/{{owner}}/{{repo}}/pulls/{pr}"))
    return pull["head"]["sha"], pull["draft"]


def run_list(sha: str) -> list[dict]:
    return json.loads(gh("run", "list", "--commit", sha, "--json", RUN_FIELDS, "--limit", "50"))


def undraft(pr: int) -> None:
    try:
        gh("pr", "ready", str(pr))
    except subprocess.CalledProcessError:
        # Cloud sessions have no GraphQL, which `gh pr ready` uses; they have this REST route.
        gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/pulls/{pr}/ccr/ready_for_review")


def wait(sha: str, exclude: frozenset[int]) -> int:
    start, first_seen = time.monotonic(), None
    while True:
        runs = select_runs(run_list(sha), sha, exclude)
        now = time.monotonic()
        if runs and first_seen is None:
            first_seen = now
        if not runs and now - start > APPEAR_S:
            print(f"no CI run for {sha[:10]} after {APPEAR_S}s (a markdown-only diff skips CI)")
            return 2
        settled = first_seen is not None and now - first_seen >= SETTLE_S
        if settled and all(r["status"] == "completed" for r in runs):
            break
        pending = ", ".join(f"{r['workflowName']}={r['status']}" for r in runs) or "no run yet"
        print(f"waiting on {sha[:10]}: {pending}", flush=True)
        time.sleep(POLL_S)
    jobs = {r["workflowName"]: json.loads(gh("run", "view", str(r["databaseId"]), "--json", "jobs"))["jobs"]
            for r in runs}
    passed, lines = verdict(jobs)
    print("\n".join(lines))
    print("CI green" if passed else "CI FAILED")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pr", type=int)
    ap.add_argument("--ready", action="store_true", help="undraft the PR first, then wait for the run that starts")
    args = ap.parse_args(argv)
    sha, draft = pr_head(args.pr)
    if args.ready and draft:
        before = frozenset(r["databaseId"] for r in run_list(sha))
        undraft(args.pr)
        print(f"#{args.pr} undrafted; waiting for the run created after it on {sha[:10]}")
        return wait(sha, before)
    if draft:
        print(f"#{args.pr} is a draft and CI skips drafts: `make ready PR={args.pr}` undrafts and waits")
        return 2
    return wait(sha, frozenset())


if __name__ == "__main__":
    sys.exit(main())
