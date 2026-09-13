"""The launch incantation lives in the repo (#169): a budget in, a correct supervised launch out."""

import ast
import pathlib
import sys

import pytest

from trm.runtime import launch

REPO = pathlib.Path(__file__).resolve().parents[2]
TOKENS = 131_072  # TOKENS_PER_OPT_STEP at batch 1, accumulation 128, two 512 windows


def test_the_stop_step_is_a_checkpoint_boundary_that_covers_the_budget():
    """The 4B run stopped at 30,518 and its last checkpoint landed at 30,464 — 56
    trained steps thrown away. The stop must be a boundary at or past the budget."""
    stop = launch.stop_step_for(4_000_000_000, TOKENS)
    assert stop == 30_528
    assert stop % launch.CHECKPOINT_EVERY_OPT_STEPS == 0
    assert stop * TOKENS >= 4_000_000_000 > (stop - launch.CHECKPOINT_EVERY_OPT_STEPS) * TOKENS


def test_the_checkpoint_cadence_matches_the_trainers():
    tree = ast.parse((REPO / "trm/train/trainer.py").read_text())
    value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", None) == "CHECKPOINT_EVERY_OPT_STEPS")
    assert value == launch.CHECKPOINT_EVERY_OPT_STEPS


def test_the_plan_pins_the_checkpoint_path_and_never_passes_new_run(tmp_path):
    p = launch.plan(4_000_000_000, TOKENS, run_id="run_x", issue=157, runs_dir=tmp_path, python="py")
    assert "--new-run" not in p.argv, "every crash relaunch would replay it and restart from scratch"
    trainer_args = p.argv[p.argv.index("--") + 1:]
    assert trainer_args == ["--checkpoint-path", str(tmp_path / "run_x" / "checkpoints")]
    assert p.argv[p.argv.index("--stop-step") + 1] == "30528"
    assert p.argv[p.argv.index("--run-dir") + 1] == str(tmp_path / "run_x")
    assert p.env == {"TRAIN_TOKEN_BUDGET": "4000000000"}


def test_it_refuses_rather_than_guesses(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no BUDGET"):
        launch.main([])
    with pytest.raises(SystemExit, match="positive"):
        launch.stop_step_for(0, TOKENS)
    (tmp_path / "run_x").mkdir()
    p = launch.plan(10**9, TOKENS, run_id="run_x", runs_dir=tmp_path)
    assert "already exists" in launch.refusal(p)


# --- test-affected --------------------------------------------------------------

sys.path.insert(0, str(REPO / "tests"))
import affected  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    def write(rel, text=""):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    write("trm/a.py", "X = 1\n")
    write("trm/b.py", "from trm.a import X\n")
    write("trm/lonely.py", "Y = 2\n")
    write("instruments/tool.py", "import trm.b\n")
    write("tests/core/test_b.py", "from trm import b\n")
    write("tests/core/test_other.py", "import os\n")
    write("tests/apparatus/test_tool.py", "from instruments import tool\n")
    write("tests/apparatus/test_reads_instruments.py", "import pathlib\nROOT = pathlib.Path('.') / 'instruments'\n")
    return tmp_path


def test_a_change_selects_every_test_that_can_reach_it_transitively(repo):
    assert affected.select(["trm/a.py"], repo) == ["tests/apparatus/test_tool.py", "tests/core/test_b.py"]


def test_a_lint_that_reads_a_tree_by_path_is_selected_for_that_tree(repo):
    assert "tests/apparatus/test_reads_instruments.py" in affected.select(["instruments/tool.py"], repo)


def test_it_fails_open(repo):
    suite = affected.SUITE
    assert affected.select(["Makefile"], repo) == suite, "non-Python, non-doc"
    assert affected.select(["tests/conftest.py"], repo) == suite
    assert affected.select(["trm/lonely.py"], repo) == suite, "imported by nothing: maybe run as a module"
    (repo / "trm/broken.py").write_text("def (:\n")
    assert affected.select(["trm/a.py"], repo) == suite, "an unparseable file breaks the graph"


def test_a_changed_test_runs_itself_and_docs_run_nothing(repo):
    assert affected.select(["tests/core/test_other.py"], repo) == ["tests/core/test_other.py"]
    assert affected.select(["docs/notes.md"], repo) == []
