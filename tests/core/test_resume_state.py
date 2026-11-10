"""The state a resume rebuilds the run from has one shape, checked at load (#477).

Jax-free: the loader's wiring into a real checkpoint is guarded in
test_rolling_checkpoint.py; this file holds the shape itself.
"""

import json

import numpy as np
import pytest

from trm.runtime.resume_state import ResumeState

# The three shapes on disk when #477 landed, one real checkpoint's state each
# (ce_history shortened, data_state cut to its keys' types).
PRE_24 = {  # the 4B champion: before samples_seen (its SFT phase fields are gone, #552)
    "ce_history": [4.71, 4.70], "best_ce": 4.52, "best_loss": 4.60, "best_avg_ce": 4.70,
    "last_improvement_step": 2176, "run_id": "run_20260705_120000"}
PRE_222 = {**PRE_24, "samples_seen": 3899392}  # the 4B base: no best_val_ce, no data_state
CURRENT = {  # the live plain base run
    "ce_history": [3.41, 3.40, 3.40, 3.39], "best_ce": 3.12, "best_loss": 3.20,
    "best_avg_ce": 3.39, "best_val_ce": 3.38, "last_improvement_step": 19072,
    "run_id": "run_20260920_191351", "samples_seen": 4890624,
    "data_state": {"rng": np.random.default_rng(3).bit_generator.state, "alive": [0, 2],
                   "sources": {"pretrain/fineweb-edu": 17}}}


def _through_json(saved):
    """What orbax's JsonSave/JsonRestore does to a state on its way to disk."""
    return json.loads(json.dumps(saved))


def test_a_save_loads_back_equal():
    state = ResumeState.load(CURRENT, "test")
    assert state.saved() == CURRENT
    assert ResumeState.load(_through_json(state.saved()), "test") == state
    # The 128-bit PCG64 state survives, not rounded through a float (#424).
    assert state.data_state["rng"] == np.random.default_rng(3).bit_generator.state


def test_a_fresh_monitor_state_survives_json_with_its_infinities():
    """A save before the first val probe carries inf bests; json writes them as
    Infinity, and they must come back as inf, not fail."""
    fresh = {**CURRENT, "best_ce": float("inf"), "best_val_ce": float("inf"), "ce_history": []}
    assert ResumeState.load(_through_json(fresh), "test").best_val_ce == float("inf")


@pytest.mark.parametrize("key", sorted(CURRENT))
def test_a_misspelled_key_fails_loudly_and_names_it(key):
    """A renamed key used to resume with its .get() default: best_val_ce as inf
    overwrote the real best, samples_seen as the micro-step mis-seeked (#24),
    data_state as None fell back to the estimate #424 replaced."""
    typo = {(k + "_" if k == key else k): v for k, v in CURRENT.items()}
    with pytest.raises(SystemExit, match=f"{key}_: Extra inputs are not permitted"):
        ResumeState.load(typo, "checkpoint step 255")


def test_a_wrong_type_fails_and_names_the_field():
    with pytest.raises(SystemExit, match="checkpoint step 255.*\n.*samples_seen"):
        ResumeState.load({**CURRENT, "samples_seen": "many"}, "checkpoint step 255")


@pytest.mark.parametrize("legacy", [PRE_24, PRE_222], ids=["pre-24", "pre-222"])
def test_legacy_checkpoints_load_through_the_declared_optional_fields(legacy):
    state = ResumeState.load(legacy, "test")
    assert state.best_val_ce == float("inf"), "#222: the first probe after resume sets a best"
    assert state.data_state is None, "#424: the resume estimates the data position"


def test_a_pre_24_checkpoint_resumes_at_its_micro_step():
    """Every checkpoint without samples_seen trained at BATCH_SIZE=1, so the
    micro-step count is its exact position."""
    class Monitor:
        pass
    pre, post = Monitor(), Monitor()
    ResumeState.load(PRE_24, "test").restore(pre, micro_step=286719)
    ResumeState.load(PRE_222, "test").restore(post, micro_step=286719)
    assert pre.samples_seen == 286719
    assert post.samples_seen == 3899392, "a recorded count is never replaced by the step"
