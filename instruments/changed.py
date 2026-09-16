"""What a change touches, as git sees it — the one diff both scoped tools share.

`tests/affected.py` (which tests can a change reach) and `instruments/audit.py`
(which experiment specs can a change reach) answer different questions from the
same list of paths. Keeping the git call here means the two can never disagree
about what "changed" means.

Two ways to ask:

- `changed_paths(base="main")` — everything since the merge-base with a branch,
  plus uncommitted and untracked files. The local loop, and a pull request.
- `changed_paths(since="<sha>")` — everything a range of commits touched. A push
  to main, where CI gets the range and there is no branch to diff against.
"""

from __future__ import annotations

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]


def _git(repo: pathlib.Path, *args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.split()


def changed_paths(repo: pathlib.Path = REPO, *, base: str = "main",
                  since: str | None = None) -> list[str]:
    """Repo-relative POSIX paths, sorted, that the change touches.

    Raises `subprocess.CalledProcessError` when git cannot resolve the base or the
    range; callers decide whether that fails open (audit everything) or loud.
    """
    if since is not None:
        return sorted(set(_git(repo, "diff", "--name-only", since, "HEAD")))
    merge_base = _git(repo, "merge-base", "HEAD", base)[0]
    committed = set(_git(repo, "diff", "--name-only", merge_base))
    untracked = set(_git(repo, "ls-files", "--others", "--exclude-standard"))
    return sorted(committed | untracked)
