"""`.env` is read from this checkout only, and its relative locations resolve against it (#541).

python-dotenv's `load_dotenv()` walks up the tree: tests run from a worktree nested at
`.claude/worktrees/<x>/` found the root's `.env`, whose `DATA_ROOT=./runs/data` then
resolved inside the worktree, where there is no corpus.
"""

import pathlib

from trm.settings import REPO_ROOT, env_file, load_env


def tree(tmp_path):
    """A parent checkout with a .env, and a child checkout nested inside it without one."""
    parent = tmp_path / "repo"
    child = parent / ".claude" / "worktrees" / "x"
    child.mkdir(parents=True)
    (parent / ".env").write_text("DATA_ROOT=./runs/data\nHF_TOKEN=secret\nCOLD_ROOT=gs://bucket/cold\n")
    return parent, child


def test_a_nested_checkout_does_not_inherit_the_parents_env(tmp_path):
    _, child = tree(tmp_path)
    assert env_file(child) == {}
    environ = {}
    load_env(child, environ)
    assert environ == {}


def test_a_relative_location_resolves_against_the_file_that_declares_it(tmp_path, monkeypatch):
    parent, child = tree(tmp_path)
    monkeypatch.chdir(child)  # the cwd is irrelevant: the old abspath resolved it here
    values = env_file(parent)
    assert values["DATA_ROOT"] == str((parent / "runs" / "data").resolve())
    assert values["HF_TOKEN"] == "secret"  # not a location: untouched
    assert values["COLD_ROOT"] == "gs://bucket/cold"  # remote: untouched


def test_the_shell_wins_over_the_file(tmp_path):
    parent, _ = tree(tmp_path)
    environ = {"DATA_ROOT": "/mnt/elsewhere"}  # a sibling worktree given an absolute root
    load_env(parent, environ)
    assert environ["DATA_ROOT"] == "/mnt/elsewhere"
    assert environ["HF_TOKEN"] == "secret"


def test_the_default_root_is_this_checkout():
    assert REPO_ROOT == pathlib.Path(__file__).resolve().parents[2]
