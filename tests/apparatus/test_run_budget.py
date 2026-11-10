"""A resume must inherit its LR horizon from the run, not from the shell (#197).

The bug these guard against was silent by construction: a power cut killed the #157
base run, the relaunch found no `TRAIN_TOKEN_BUDGET` in its environment, and the
cosine fell back to a 15,000-step horizon for a run built with 30,518. Nothing
raised. The only symptom would have been a learning rate at zero for the back half
of a ten-day run.

So the properties worth pinning are the two that make silence impossible: the
budget is recovered when it can be, and the mismatch is fatal when it can't. The
budget is recovered into the Config the trainer is handed (#475), never into the
environment.
"""

import json
import os

from trm.runtime.run_budget import BUDGET_ENV, horizon_mismatch, with_recorded_budget
from trm.settings import Config
from trm.train.schedules import Schedules

UNSET = Config.from_env({})


def _run_dir(tmp_path, budget=4_000_000_000, decay_steps=30_518):
    """A run directory shaped like a real one: metadata beside a checkpoints folder."""
    run = tmp_path / "run_20260813_214725"
    (run / "checkpoints").mkdir(parents=True)
    parameters = {}
    if budget is not None:
        parameters[BUDGET_ENV] = budget
    if decay_steps is not None:
        parameters["DECAY_STEPS"] = decay_steps
    (run / "run_metadata.json").write_text(json.dumps({"run_id": run.name, "parameters": parameters}))
    return run


def test_the_budget_is_recovered_from_the_run_being_resumed(tmp_path):
    """The #157 case exactly: relaunch with an empty environment, get 4B back."""
    run = _run_dir(tmp_path)
    assert with_recorded_budget(UNSET, str(run / "checkpoints")).TRAIN_TOKEN_BUDGET == 4_000_000_000


def test_recovering_the_budget_changes_nothing_else(tmp_path):
    """The run records the Config it was handed; a resume's must differ from the
    launch's only in what the launch left unset."""
    run = _run_dir(tmp_path)
    recovered = with_recorded_budget(UNSET, str(run / "checkpoints"))
    assert recovered.model_dump(exclude={BUDGET_ENV}) == UNSET.model_dump(exclude={BUDGET_ENV})


def test_an_explicit_budget_still_wins(tmp_path):
    """Deliberately extending a run's horizon has to keep working — the fix is for
    the absent case, not an override lock. The mismatch check below then judges it."""
    run = _run_dir(tmp_path)
    explicit = Config.from_env({BUDGET_ENV: "8000000000"})
    assert with_recorded_budget(explicit, str(run / "checkpoints")) is explicit


def test_a_new_run_is_untouched(tmp_path):
    """No metadata, no adoption — a first launch resolves from its own Config
    exactly as it did before, including resolving to nothing."""
    assert with_recorded_budget(UNSET, str(tmp_path / "checkpoints")) is UNSET


def test_a_run_that_genuinely_had_no_budget_stays_that_way(tmp_path):
    """Metadata present, budget null: the historical default is the *correct*
    answer here, so adoption must not invent one."""
    run = _run_dir(tmp_path, budget=None)
    assert with_recorded_budget(UNSET, str(run / "checkpoints")) is UNSET


def test_no_checkpoint_path_is_not_an_error():
    assert with_recorded_budget(UNSET, None) is UNSET


def test_corrupt_metadata_falls_back_instead_of_crashing(tmp_path):
    """Half-written JSON (a power cut mid-write is exactly how we got here) must
    not stop a run from starting."""
    run = tmp_path / "run_x"
    (run / "checkpoints").mkdir(parents=True)
    (run / "run_metadata.json").write_text('{"parameters": {"TRAIN_TOKEN')
    assert with_recorded_budget(UNSET, str(run / "checkpoints")) is UNSET


def test_a_horizon_that_disagrees_with_the_run_is_fatal(tmp_path):
    """The 15,000-vs-30,518 case. It must name both numbers — a resume at 3am is
    read by whoever is awake, and 'mismatch' alone tells them nothing."""
    run = _run_dir(tmp_path)
    complaint = horizon_mismatch(str(run), 15_000)
    assert complaint is not None
    assert "30,518" in complaint and "15,000" in complaint


def test_the_matching_horizon_says_nothing(tmp_path):
    """A check that fires on a healthy resume would be trained away in a week."""
    run = _run_dir(tmp_path)
    assert horizon_mismatch(str(run), 30_518) is None


def test_a_run_predating_the_recorded_horizon_is_not_condemned(tmp_path):
    """Older metadata has no DECAY_STEPS. Unknown is not the same as wrong, and
    refusing to resume those runs would be a worse bug than the one being fixed."""
    run = _run_dir(tmp_path, decay_steps=None)
    assert horizon_mismatch(str(run), 15_000) is None


def test_adoption_and_the_guard_agree_on_a_real_resume(tmp_path):
    """End to end: adopt from the checkpoint path, resolve, and the guard is quiet.
    The two halves are separate functions and could drift apart; this is the seam."""
    run = _run_dir(tmp_path)
    config = with_recorded_budget(UNSET, str(run / "checkpoints"))
    assert horizon_mismatch(str(run), Schedules.of(config).decay_steps) is None


def test_the_live_run_would_survive_its_own_relaunch(champion_run):
    """Against the #157 run's recorded metadata (tests/apparatus/fixtures). This is the
    exact scenario the power cut produced, and the reason the issue exists."""
    run = str(champion_run)
    config = with_recorded_budget(UNSET, os.path.join(run, "checkpoints"))
    assert config.TRAIN_TOKEN_BUDGET == 4_000_000_000
    assert horizon_mismatch(run, Schedules.of(config).decay_steps) is None
