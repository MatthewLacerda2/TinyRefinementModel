"""The cold tier (#458): what a run copies to the HDD, and the one deletion that copy
permits on the SSD. Driven against fake checkpoint trees in tmp dirs; a step dir here
is what orbax leaves (a dir named by the step, finalized by its _CHECKPOINT_METADATA).

The rules that matter most are the deletion's: pruning never removes a rolling or best
checkpoint, never a milestone without a byte-identical cold copy, never anything under
runs/data/, and nothing else in trm/runtime/cold.py deletes at all.
"""

import ast
import json
import pathlib
import shutil

import pytest

from trm.runtime import cold
from trm.runtime.cold import (FULLSTATE_SUBDIR, MARKER, ColdTier, cold_root_problem, committed_at,
                              finalized_steps, mirror, stall_window_hours)
from trm.runtime.layout import BEST_SUBDIR, MILESTONE_SUBDIR

REPO = pathlib.Path(__file__).resolve().parents[2]
HOUR = 3600
EPOCH = 10**6  # seconds; the fake run's hour 0


def _step(parent, step, hours=0.0, payload=b"weights", finalized=True):
    """A fake orbax step dir, committed `hours` into the run."""
    d = parent / str(step)
    (d / "model").mkdir(parents=True)
    (d / "model" / "shard").write_bytes(payload * 100)
    (d / "monitor_state").mkdir()
    (d / "monitor_state" / "metadata").write_text(json.dumps({"data_state": {"sources": {"web": step}}}))
    (d / "step").mkdir()
    if finalized:
        # Integer nanoseconds, so a window boundary is exact.
        (d / MARKER).write_text(json.dumps({"commit_timestamp_nsecs": (EPOCH + round(hours * HOUR)) * 10**9}))
    return d


def _tree(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def ssd(tmp_path, monkeypatch):
    """runs/ with the corpus, one run's rolling, best and milestone checkpoints, and a
    cold root beside it. Both are on one tmp disk, so the other-disk check is stubbed to
    'is it there'; its refusals are tested on their own below."""
    monkeypatch.setattr(cold, "cold_root_problem",
                        lambda root, live: None if root.is_dir() else f"{root} is not a directory")
    runs = tmp_path / "runs"
    data = runs / "data" / "pretrain" / "fineweb-edu"
    data.mkdir(parents=True)
    (data / "chunk_0.npy").write_bytes(b"corpus")
    run = runs / "run_x"
    ckpts = run / "checkpoints"
    for s, h in ((639, 1), (703, 2), (767, 3)):
        _step(ckpts, s, h)
    for s in (511, 767):
        _step(ckpts / BEST_SUBDIR, s)
    for s in (60, 121, 243, 487):
        _step(ckpts / MILESTONE_SUBDIR, s)
    (tmp_path / "cold").mkdir()
    return tmp_path


def _tier(root, free=lambda: 1.0, hours=12.0):
    return ColdTier(run_dir=root / "runs" / "run_x", cold_root=root / "cold", keep_free_gb=20.0,
                    fullstate_every_hours=hours, protected=(root / "runs" / "data",), free_gb=free)


def _everything_but_milestones(root):
    ckpts = root / "runs" / "run_x" / "checkpoints"
    kept = {p: p.read_bytes() for p in ckpts.rglob("*") if p.is_file() and MILESTONE_SUBDIR not in p.parts}
    kept.update({p: p.read_bytes() for p in (root / "runs" / "data").rglob("*") if p.is_file()})
    return kept


# --- copies -----------------------------------------------------------------------

def test_a_copy_lands_whole_and_only_once(tmp_path):
    src = _step(tmp_path / "ssd", 767)
    dst = tmp_path / "hdd" / "767"
    assert mirror(src, dst) is True
    assert _tree(dst) == _tree(src)
    assert not list(dst.parent.glob(".*")), "no partial copy left behind"
    assert mirror(src, dst) is False, "a copy that is there is not redone"


def test_an_unfinished_or_vanished_checkpoint_is_not_copied(tmp_path):
    unfinished = _step(tmp_path / "ssd", 767, finalized=False)
    with pytest.raises(OSError):
        mirror(unfinished, tmp_path / "hdd" / "767")
    with pytest.raises(OSError):
        mirror(tmp_path / "ssd" / "999", tmp_path / "hdd" / "999")
    assert not (tmp_path / "hdd" / "767").exists() and not (tmp_path / "hdd" / "999").exists()


def test_a_copy_interrupted_midway_leaves_nothing_named_by_its_step(tmp_path, monkeypatch):
    """Rolling retention can evict a checkpoint under the copy, and the power can go."""
    src = _step(tmp_path / "ssd", 767)
    real, calls = shutil.copyfile, []

    def dies_on_the_second_file(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise FileNotFoundError(a)
        return real(a, b)

    monkeypatch.setattr(cold.shutil, "copyfile", dies_on_the_second_file)
    with pytest.raises(OSError):
        mirror(src, tmp_path / "hdd" / "767")
    assert not (tmp_path / "hdd" / "767").exists()
    assert not list((tmp_path / "hdd").glob(".*")), "the partial copy is cleaned up"
    monkeypatch.setattr(cold.shutil, "copyfile", real)
    assert mirror(src, tmp_path / "hdd" / "767") and _tree(tmp_path / "hdd" / "767") == _tree(src)


def test_the_commit_time_travels_with_the_copy(tmp_path):
    """The cadence reads orbax's commit time, not mtime: on the NTFS HDD a copy's
    mtime is when it was copied."""
    src = _step(tmp_path / "ssd", 767, hours=5)
    mirror(src, tmp_path / "hdd" / "767")
    assert committed_at(tmp_path / "hdd" / "767") == committed_at(src) == EPOCH + 5 * HOUR


# --- a tick ---------------------------------------------------------------------------

def test_a_tick_mirrors_one_milestone_per_poll_and_the_newest_full_state(ssd):
    """One milestone per tick: the copy runs inside the supervisor's poll, and a resumed
    run's backlog (16 x ~550MB) must not hold its guards for minutes."""
    tier = _tier(ssd, free=lambda: 100.0)
    assert len(tier.tick()) == 2, "milestone 60 and the full state"
    for _ in range(3):
        assert len(tier.tick()) == 1
    assert [p.name for p in finalized_steps(tier.cold_run / MILESTONE_SUBDIR)] == ["60", "121", "243", "487"]
    assert [p.name for p in finalized_steps(tier.cold_run / FULLSTATE_SUBDIR)] == ["767"]
    assert tier.tick() == [], "nothing new, nothing done"


def test_a_rewound_run_saving_a_step_again_replaces_its_copy_and_keeps_the_old_one(ssd):
    """`python -m trm.runtime.rewind` sets checkpoints aside and the resumed run saves
    the same step numbers again, as other saves. The cold copy must follow the live
    run, or a decay branched from it starts from the abandoned state."""
    tier = _tier(ssd, free=lambda: 100.0)
    tier.tick(final=True)
    ckpts = tier.run_dir / "checkpoints"
    for d in (ckpts / "767", ckpts / MILESTONE_SUBDIR / "487"):
        shutil.rmtree(d)
    _step(ckpts, 767, hours=40, payload=b"retrained")
    _step(ckpts / MILESTONE_SUBDIR, 487, hours=39, payload=b"retrained")
    tier.tick(final=True)
    for kind, live in ((FULLSTATE_SUBDIR, ckpts / "767"), (MILESTONE_SUBDIR, ckpts / MILESTONE_SUBDIR / "487")):
        assert _tree(tier.cold_run / kind / live.name) == _tree(live)
        stale = tier.cold_run / kind / f".{live.name}.superseded-0"
        assert (stale / "model" / "shard").read_bytes() == b"weights" * 100, "the abandoned save, kept aside"


def test_full_state_is_copied_once_per_stall_window_and_at_the_end(ssd):
    tier = _tier(ssd, free=lambda: 100.0, hours=12.0)
    ckpts = tier.run_dir / "checkpoints"
    tier.tick()  # the first copy: 767, committed at hour 3
    _step(ckpts, 831, hours=14.9)
    assert tier.fullstate_due() is None, "11.9h after the last copy: inside the window"
    _step(ckpts, 895, hours=15.0)
    assert tier.fullstate_due().name == "895", "12h after the last copy"
    tier.tick()
    _step(ckpts, 959, hours=16.0)
    assert tier.fullstate_due() is None
    assert tier.fullstate_due(final=True).name == "959", "the run's last state, whatever the window"
    tier.tick(final=True)
    assert [p.name for p in finalized_steps(tier.cold_run / FULLSTATE_SUBDIR)] == ["767", "895", "959"]


def test_without_a_stall_window_only_the_last_state_is_copied(ssd):
    tier = _tier(ssd, free=lambda: 100.0, hours=None)
    tier.tick()
    assert finalized_steps(tier.cold_run / FULLSTATE_SUBDIR) == []
    tier.tick(final=True)
    assert [p.name for p in finalized_steps(tier.cold_run / FULLSTATE_SUBDIR)] == ["767"]


def test_the_window_is_the_base_specs_stall_window(tmp_path):
    assert stall_window_hours(REPO / "experiments" / "base" / "specs" / "001-plain-base.toml") == 12.0
    bare = tmp_path / "spec.toml"
    bare.write_text("[protocol]\nbudget_tokens = 1\n")
    assert stall_window_hours(bare) is None and stall_window_hours(None) is None


# --- pruning --------------------------------------------------------------------------

def test_pruning_never_removes_a_rolling_or_best_checkpoint_nor_the_only_copy(ssd):
    """The worst case: the SSD is below its margin forever, so pruning removes all it
    may. What it may: mirrored milestones, and not the newest (a CPU scorer may be
    reading it). Not a milestone without a copy, nor one whose copy differs."""
    free = [100.0]
    tier = _tier(ssd, free=lambda: free[0])
    milestones = tier.run_dir / "checkpoints" / MILESTONE_SUBDIR
    tier.tick(final=True)  # mirrors 60, 121, 243, 487 with room to spare
    _step(milestones, 975)  # a new milestone, not yet mirrored...
    _step(milestones, 1951)  # ...and the newest
    (tier.cold_run / MILESTONE_SUBDIR / "243" / "model" / "shard").write_bytes(b"torn")  # a copy gone bad
    kept = _everything_but_milestones(ssd)
    free[0] = 1.0
    tier.prune()

    assert [p.name for p in finalized_steps(milestones)] == ["243", "975", "1951"]
    assert not (tier.cold_run / MILESTONE_SUBDIR / "243").exists(), "the torn copy set aside, to redo"
    free[0] = 100.0
    tier.tick(final=True)
    assert _tree(tier.cold_run / MILESTONE_SUBDIR / "243") == _tree(milestones / "243")
    for name in ("60", "121", "487"):
        assert (tier.cold_run / MILESTONE_SUBDIR / name / MARKER).is_file(), "removed only with a copy"
    assert {p: p.read_bytes() for p in kept} == kept, "rolling, best and runs/data/ untouched"


def test_pruning_stops_at_the_margin(ssd):
    free = iter([100.0, 5.0, 12.0, 25.0, 25.0])
    tier = _tier(ssd, free=lambda: next(free))
    tier.tick(final=True)
    tier.prune()
    assert [p.name for p in finalized_steps(tier.run_dir / "checkpoints" / MILESTONE_SUBDIR)] == ["243", "487"]


def test_nothing_is_pruned_while_the_cold_root_is_gone(ssd):
    tier = _tier(ssd, free=lambda: 1.0)
    tier.tick()
    before = sorted(p.name for p in finalized_steps(tier.run_dir / "checkpoints" / MILESTONE_SUBDIR))
    shutil.rmtree(ssd / "cold")
    with pytest.raises(OSError, match="not a directory"):
        tier.tick()
    assert sorted(p.name for p in finalized_steps(tier.run_dir / "checkpoints" / MILESTONE_SUBDIR)) == before


# --- the one deletion, asked directly ----------------------------------------------------

def test_the_removal_refuses_anything_but_a_mirrored_milestone(ssd):
    tier = _tier(ssd)
    tier.tick()
    ckpts = tier.run_dir / "checkpoints"
    mirror(ckpts / "639", tier.cold_run / MILESTONE_SUBDIR / "639")  # a copy that would pass
    link = ckpts / MILESTONE_SUBDIR / "9999"
    link.symlink_to(ssd / "runs" / "data", target_is_directory=True)
    for target in (ckpts / "639", ckpts / BEST_SUBDIR / "767", link, ssd / "runs" / "data",
                   _step(ckpts / MILESTONE_SUBDIR, 3000)):  # the last: no copy
        with pytest.raises(ValueError, match="refusing"):
            tier._remove_mirrored_milestone(target)
    assert (ssd / "runs" / "data" / "pretrain" / "fineweb-edu" / "chunk_0.npy").read_bytes() == b"corpus"
    assert (ckpts / "639" / MARKER).is_file() and (ckpts / MILESTONE_SUBDIR / "3000").is_dir()


def test_a_run_inside_the_corpus_has_nothing_removed(ssd):
    """If a run dir ever sat under runs/data/, its milestones are the corpus's tree."""
    shutil.move(ssd / "runs" / "run_x", ssd / "runs" / "data" / "run_x")
    tier = ColdTier(run_dir=ssd / "runs" / "data" / "run_x", cold_root=ssd / "cold", keep_free_gb=20.0,
                    fullstate_every_hours=None, protected=(ssd / "runs" / "data",), free_gb=lambda: 1.0)
    with pytest.raises(ValueError, match="protected"):
        tier.tick()
    assert len(finalized_steps(tier.run_dir / "checkpoints" / MILESTONE_SUBDIR)) == 4


def test_a_cold_root_on_the_same_disk_or_missing_is_refused(tmp_path):
    (tmp_path / "runs").mkdir()
    assert "not a directory" in cold_root_problem(tmp_path / "gone", tmp_path / "runs")
    (tmp_path / "cold").mkdir()
    assert "same disk" in cold_root_problem(tmp_path / "cold", tmp_path / "runs")


def test_nothing_else_in_the_cold_tier_deletes():
    """Every call that can remove a file lives in one of the two guarded functions:
    the milestone removal, and the cleanup of this module's own partial copies."""
    source = (REPO / "trm" / "runtime" / "cold.py").read_text()
    tree = ast.parse(source)
    deleting = {"rmtree", "remove", "unlink", "rmdir", "removedirs", "replace", "move"}
    allowed = {"_remove_mirrored_milestone", "_discard_partial"}
    found = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in deleting):
                    found.append((fn.name, node.func.attr))
    assert found and {name for name, _ in found} <= allowed, found
    # os.rename moves a copy on the cold side only: a finished copy into its step's
    # name, or a superseded one out of it.
    renames = [fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
               for node in ast.walk(fn) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "rename"]
    assert sorted(renames) == ["mirror", "set_aside"]


# --- in the supervisor ------------------------------------------------------------------

def _supervised(tmp_path, tier, **kw):
    import sys
    import textwrap

    from trm.runtime.supervisor import Limits, Supervisor
    from trm.settings import Config

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import time, pathlib
        p = pathlib.Path({str(tier.run_dir / "metrics.csv")!r})
        for step in range(0, 200, 10):
            p.write_text("step,ce\\n" + f"{{step}},3.0\\n")
            time.sleep(0.1)
        time.sleep(60)
    """))
    reported = []
    sup = Supervisor(command=(sys.executable, str(child)), limits=Limits(stop_step=50, max_retries=0),
                     log_path=tmp_path / "train.log", metrics_csv=tier.run_dir / "metrics.csv",
                     config=Config.from_env({}), poll_seconds=0.2, heartbeat_every=1000, report=reported.append, cold=tier, **kw)
    return sup, reported


def test_a_supervised_run_leaves_its_last_full_state_on_the_cold_tier(ssd):
    from trm.runtime.supervisor import BUDGET_COMPLETE

    tier = _tier(ssd, free=lambda: 100.0, hours=None)
    sup, _ = _supervised(ssd, tier, heartbeat_log=ssd / "heartbeat.log")
    assert sup.run() == BUDGET_COMPLETE
    assert [p.name for p in finalized_steps(tier.cold_run / FULLSTATE_SUBDIR)] == ["767"]
    assert len(finalized_steps(tier.cold_run / MILESTONE_SUBDIR)) == 4
    assert "milestone 60 mirrored" in (ssd / "heartbeat.log").read_text()


def test_an_unmounted_cold_tier_is_announced_once_and_the_run_goes_on(ssd):
    from trm.runtime.supervisor import BUDGET_COMPLETE

    shutil.rmtree(ssd / "cold")
    sup, reported = _supervised(ssd, _tier(ssd))
    assert sup.run() == BUDGET_COMPLETE
    assert sum("cold tier" in line for line in reported) == 1, reported
