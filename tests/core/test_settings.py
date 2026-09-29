"""The run's knobs are one frozen Config, read once and recorded whole (#475).

Building a Config is a pure function of a mapping, so these cases need no fresh
interpreter. This file holds the reader itself; tests/core/test_config_read_once.py
holds who may read it.
"""

import pytest

from trm.settings import CONFIG, DEFAULT_DATA_MIXTURE, Config

# What an unset environment resolved to before the Config existed, value and type:
# every run so far, the golden run and every recorded recipe pair trained on these.
TODAYS_DEFAULTS = {
    "FORCE_F32_COMPUTE": False,
    "LATENT_DIM": 960, "MAX_SEQ_LEN": 512, "NUM_HEADS": 15, "MODEL_ARCH": "plain",
    "POSITION_ENCODING": "rope",
    "POST_NORM": False, "PLAIN_LAYERS": 8,
    "TIME_SIGNAL": "sinusoidal", "REFINER_ENCODER_LAYERS": 7, "INFERENCE_DEPTH": 6,
    "TRM_OPTIMIZER": "muon", "MUON_LR_MULT": 16.666667,
    "ADAM_B1": 0.9, "ADAM_B2": 0.999, "ADAM_EPS": 1e-8, "WEIGHT_DECAY": 1e-2, "CLIP_NORM": 1.0,
    "MUON_BETA": 0.95, "MUON_NS_STEPS": 5, "MUON_EPS": 1e-8, "MUON_NESTEROV": True,
    "LOSS_SCALE_GROWTH_INTERVAL": 256,
    "BATCH_SIZE": 2, "ACCUMULATION_STEPS": 64, "TOKENS_PER_OPT_STEP": 131072,
    "EVAL_ROWS": 64, "VAL_SKIP_SAMPLES": 3_000_000,
    "PLATEAU_MIN_DELTA": 0.01, "PLATEAU_PATIENCE": 400,
    "TRAIN_TOKEN_BUDGET": None, "WARMUP_STEPS": 1000, "PEAK_LR": 6e-4,
    "LR_SCHEDULE": "wsd", "WSD_DECAY_FRACTION": 0.2, "WSD_DECAY_START": None,
    "PAD_TOKEN_ID": 50257, "DATA_SEED": 42, "MODEL_SEED": 42,
    "DATA_MIXTURE": DEFAULT_DATA_MIXTURE, "MIXTURE_RAMP_FRACTION": 10000 / 30518,
    "DATA_BRANCH": False, "DATA_BRANCH_SEED_STRIDE": 80_000,
    "VAL_EVERY_OPT_STEPS": 64, "CHECKPOINT_EVERY_OPT_STEPS": 64, "VAL_BY_SOURCE_EVERY_OPT_STEPS": 128,
    "MILESTONE_FIRST_TOKENS": 8_000_000, "MILESTONE_RATIO": 2.0, "MILESTONE_MAX_COUNT": 16,
    "ACT_MAX_ALARM": 16376.0, "LOSS_SCALE_FLOOR_ALARM": 4.0, "ZERO_GRAD_ALARM": 0.05,
    "VRAM_HEADROOM_ALARM_MIB": 150.0, "SSD_KEEP_FREE_GB": 20.0, "MILESTONE_SCORERS": 1,
}


def test_an_unset_environment_resolves_to_todays_values():
    resolved = Config.from_env({}).model_dump()
    assert resolved.keys() == TODAYS_DEFAULTS.keys(), "a knob was added or dropped: pin its default here"
    moved = {k: (TODAYS_DEFAULTS[k], v) for k, v in resolved.items()
             if v != TODAYS_DEFAULTS[k] or type(v) is not type(TODAYS_DEFAULTS[k])}
    assert not moved, f"(before, now): {moved}"


def test_only_the_given_environment_is_read():
    """from_env reads the mapping it is handed, nothing behind it, and ignores
    variables that are not knobs."""
    config = Config.from_env({"BATCH_SIZE": "4", "PEAK_LR": "3e-4", "HOME": "/root"})
    assert (config.BATCH_SIZE, config.ACCUMULATION_STEPS, config.PEAK_LR) == (4, 32, 3e-4)
    assert config.TOKENS_PER_OPT_STEP == TODAYS_DEFAULTS["TOKENS_PER_OPT_STEP"], "the product stays fixed"
    assert Config.from_env({}) == Config(), "an empty environment is the defaults"


@pytest.mark.parametrize("value, parsed", [("2e9", 2_000_000_000), ("67108864", 67_108_864), ("", None)])
def test_the_token_budget_accepts_scientific_notation(value, parsed):
    assert Config.from_env({"TRAIN_TOKEN_BUDGET": value}).TRAIN_TOKEN_BUDGET == parsed


def test_flags():
    assert Config.from_env({"POST_NORM": "1", "MUON_NESTEROV": "0"}).model_dump(
        include={"POST_NORM", "MUON_NESTEROV"}) == {"POST_NORM": True, "MUON_NESTEROV": False}
    # The test suite's escape hatch has always meant "set", whatever it is set to.
    assert Config.from_env({"FORCE_F32_COMPUTE": "1"}).FORCE_F32_COMPUTE
    assert Config.from_env({"FORCE_F32_COMPUTE": "0"}).FORCE_F32_COMPUTE


@pytest.mark.parametrize("knob, value", [
    ("MODEL_ARCH", "refnier"), ("TIME_SIGNAL", "sinsuoidal"), ("TRM_OPTIMIZER", "adam"),
    ("LR_SCHEDULE", "linear"), ("BATCH_SIZE", "3"), ("BATCH_SIZE", "0"), ("PAD_TOKEN_ID", "0"),
    ("LATENT_DIM", "wide"), ("POST_NORM", "maybe"), ("MILESTONE_SCORERS", "0"),
    ("POSITION_ENCODING", "none"),
])
def test_a_bad_knob_refuses_to_start_and_says_which(knob, value):
    """Fail closed (#104): a typo must not silently train a different run."""
    with pytest.raises(SystemExit) as refused:
        Config.from_env({knob: value})
    assert f"{knob}={value!r}" in str(refused.value.code)


@pytest.mark.parametrize("arch", ["refiner", "reasoner"])
def test_nope_refuses_an_arch_that_would_ignore_it(arch):
    """#444: only the plain stack reads POSITION_ENCODING. A refiner launched with
    "nope" would train RoPE under a record that says otherwise."""
    with pytest.raises(SystemExit) as refused:
        Config.from_env({"MODEL_ARCH": arch, "POSITION_ENCODING": "nope"})
    assert "POSITION_ENCODING='nope'" in str(refused.value.code)
    assert Config.from_env({"MODEL_ARCH": arch}).POSITION_ENCODING == "rope"
    assert Config.from_env({"POSITION_ENCODING": "nope"}).POSITION_ENCODING == "nope"


def test_every_bad_knob_is_named_at_once():
    with pytest.raises(SystemExit) as refused:
        Config.from_env({"MODEL_ARCH": "x", "BATCH_SIZE": "3"})
    assert "MODEL_ARCH" in refused.value.code and "BATCH_SIZE" in refused.value.code


def test_frozen():
    with pytest.raises(Exception, match="frozen"):
        CONFIG.BATCH_SIZE = 4


def test_the_run_records_every_knob():
    """"Every knob recorded" (#358) by construction: run_metadata.json's parameters
    are the Config's dump, so a new field is recorded without anyone listing it."""
    from trm.runtime.run_tracker import RunTracker

    recorded = RunTracker.get_hyperparameters(CONFIG)
    dumped = CONFIG.model_dump()
    assert {k: recorded.get(k) for k in dumped} == dumped


def test_no_module_re_exports_a_knob():
    """A knob is read from the Config its caller hands down (#475), never from a module
    constant frozen at import: none of the modules that used to re-export one does."""
    import importlib

    knobs = set(Config.model_fields) | set(Config.model_computed_fields)
    for name in ("trm.config", "trm.runtime.layout", "trm.train.schedules", "trm.train.validation"):
        exported = knobs & set(vars(importlib.import_module(name)))
        assert not exported, f"{name} re-exports {sorted(exported)}: pass a Config down instead"
