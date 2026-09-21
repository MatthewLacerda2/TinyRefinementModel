"""The training mixture is a named knob, and today's default is exactly today (#439).

Every run used to read the same three sources through one hard-coded ramp, so a
pair asking a question about prose still trained on 65% code and math by its end
(#362), and nothing let a spec say otherwise. The mixture is now `DATA_MIXTURE`:
buckets and weights, chosen in the spec, recorded in the run's metadata.

Three things have to hold for that to be safe to adopt mid-project:

- an unset environment resolves to the ramp every run so far trained with, down to
  the last bit, or the change silently alters every future run;
- two arms that differ only in their mixture differ in nothing else, or a mixture
  pair is not a matched pair;
- a resume refuses a data state written for a different mixture, and still accepts
  one written before mixtures had names — the live base run's checkpoints are those.
"""

import json

import numpy as np
import pytest

from trm.config import MAX_SEQ_LEN
from trm.train import schedules

STRIDE = 2 * MAX_SEQ_LEN + 1


# ── today, exactly ───────────────────────────────────────────────────────────

def _todays_curriculum(step, ramp_steps):
    """The ramp as it was written before #439, with its literals, frozen here."""
    start, end = [0.85, 0.10, 0.05], [0.35, 0.40, 0.25]
    if step >= ramp_steps:
        return list(end)
    fraction = float(step) / ramp_steps
    return [s + (e - s) * fraction for s, e in zip(start, end)]


def test_the_default_mixture_is_the_one_every_run_trained_with():
    assert schedules.PRETRAIN_SOURCES == (
        "pretrain/fineweb-edu", "pretrain/codeparrot", "pretrain/finemath")
    assert schedules.CURRICULUM_START_WEIGHTS == [0.85, 0.10, 0.05]
    assert schedules.CURRICULUM_END_WEIGHTS == [0.35, 0.40, 0.25]
    from trm.config import MIXTURE_RAMP_FRACTION
    assert MIXTURE_RAMP_FRACTION == 10000 / 30518


def test_the_default_ramp_reproduces_bit_for_bit():
    """Not approximately: `==` on every weight at every step checked. A float that
    moved in its last bit would change which corpus a multinomial draw picks."""
    ramp = schedules.CURRICULUM_STEPS
    for step in [0, 1, 7, ramp // 3, ramp // 2, ramp - 1, ramp, ramp + 1, 10 * ramp]:
        assert schedules.get_curriculum_weights(step) == _todays_curriculum(step, ramp), step


# ── the knob ─────────────────────────────────────────────────────────────────

def test_a_mixture_parses_into_buckets_and_both_ends_of_its_ramp():
    buckets, start, end = schedules.parse_mixture("a/web=0.7:0.2, b/code=0.3:0.8")
    assert buckets == ("a/web", "b/code")
    assert start == [0.7, 0.3] and end == [0.2, 0.8]


def test_a_single_weight_holds_for_the_whole_run():
    buckets, start, end = schedules.parse_mixture("pretrain/fineweb-edu=1")
    assert buckets == ("pretrain/fineweb-edu",) and start == end == [1.0]


@pytest.mark.parametrize("text, why", [
    ("a=0.5:0.5,b=0.4:0.5", "sum to"),        # a start column that does not sum to 1
    ("a=0.5:0.6,b=0.5:0.5", "sum to"),        # nor an end one
    ("a=-0.5,b=1.5", "negative"),
    ("a=0.5,a=0.5", "twice"),
    ("a=lots", "not a weight"),
    ("just-a-name", "bucket=start:end"),
    ("", "no bucket"),
])
def test_a_mixture_that_does_not_say_what_it_means_is_refused(text, why):
    """At import, before a model is built. A renormalization nobody wrote down would
    be a hidden default on the hot path, so a column that misses 1 is an error."""
    with pytest.raises(SystemExit, match=why):
        schedules.parse_mixture(text)


def test_two_arms_that_differ_only_in_their_mixture_differ_in_nothing_else(import_config_under):
    """The matched-pair property. Every other schedule the run reads must resolve the
    same, so the mixture is the only variable a mixture pair moves."""
    other = "pretrain/fineweb-edu=1"
    names = ["PRETRAIN_SOURCES", "CURRICULUM_START_WEIGHTS", "CURRICULUM_END_WEIGHTS",
             "CURRICULUM_STEPS", "DECAY_STEPS", "WARMUP_STEPS", "PEAK_LR", "LR_SCHEDULE"]
    default, prose = import_config_under(
        [{"DATA_MIXTURE": None}, {"DATA_MIXTURE": other}],
        [f"trm.train.schedules:{name}" for name in names])
    assert default["ok"] and prose["ok"]
    moved = {name for name in names
             if default["values"][f"trm.train.schedules:{name}"]
             != prose["values"][f"trm.train.schedules:{name}"]}
    assert moved == {"PRETRAIN_SOURCES", "CURRICULUM_START_WEIGHTS", "CURRICULUM_END_WEIGHTS"}


def test_the_run_records_the_mixture_it_read():
    from trm.config import DATA_MIXTURE, MIXTURE_RAMP_FRACTION
    from trm.runtime.run_tracker import RunTracker
    recorded = RunTracker.get_hyperparameters()
    assert recorded["DATA_MIXTURE"] == DATA_MIXTURE
    assert recorded["MIXTURE_RAMP_FRACTION"] == MIXTURE_RAMP_FRACTION



# ── resuming ─────────────────────────────────────────────────────────────────

def _corpus(root, name, base):
    d = root / name
    d.mkdir(exist_ok=True)
    np.save(d / "chunk_0.npy", (np.arange(base, base + 6 * STRIDE) % 50000).astype(np.int32))
    return str(d)


def _mixer(tmp_path, names=("web", "code")):
    from trm.data.loaders import DataMixer, TextDataGenerator
    dirs = [_corpus(tmp_path, "web", 0), _corpus(tmp_path, "code", 20000)]
    return DataMixer([TextDataGenerator(d, rng=np.random.default_rng(7)) for d in dirs],
                     [0.6, 0.4], rng=np.random.default_rng(11), names=names)


def test_a_state_carries_the_names_of_its_buckets(tmp_path):
    mixer = _mixer(tmp_path)
    mixer.get_batch(1)
    state = json.loads(json.dumps(mixer.state()))
    assert state["names"] == ["web", "code"]
    _mixer(tmp_path).load_state(state)  # and the same mixture takes it back


def test_a_state_written_for_another_mixture_is_refused(tmp_path):
    """Same count, different buckets: a positional restore would put the code
    reader's position onto a web shard and train on a stream nobody chose."""
    state = _mixer(tmp_path, names=("web", "code")).state()
    with pytest.raises(ValueError, match="DATA_MIXTURE"):
        _mixer(tmp_path, names=("web", "math")).load_state(state)


def test_the_same_buckets_in_another_order_are_another_mixture(tmp_path):
    state = _mixer(tmp_path, names=("web", "code")).state()
    with pytest.raises(ValueError, match="DATA_MIXTURE"):
        _mixer(tmp_path, names=("code", "web")).load_state(state)


def test_a_state_from_before_buckets_had_names_still_resumes(tmp_path):
    """The live base run's checkpoints carry no names. A relaunch after this lands
    must still pick up exactly where it stopped."""
    reference = _mixer(tmp_path)
    expected = [reference.get_batch(1)[0] for _ in range(6)]

    first = _mixer(tmp_path)
    for _ in range(3):
        first.get_batch(1)
    legacy = json.loads(json.dumps(first.state()))
    del legacy["names"]

    resumed = _mixer(tmp_path)
    resumed.load_state(legacy)
    for want in expected[3:]:
        np.testing.assert_array_equal(np.asarray(resumed.get_batch(1)[0]), np.asarray(want))
