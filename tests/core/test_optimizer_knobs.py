"""No hidden defaults on the hot path (#358): every numeric knob of the optimizer is
named in trm/config.py and handed to optax by name.

Before this, AdamW's b1/b2/eps and all of Muon's own knobs were whatever optax
defaulted to, recorded nowhere. An optax upgrade that moved one would have changed
the recipe with no diff in this repo. These tests catch the call sites leaving any
of them to the library again, and pin the values to what every run so far used.
"""

import optax

import trm.config as config
from trm.train import optimizers


def _captured(monkeypatch, target, name):
    calls = []
    original = getattr(target, name)

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, spy)
    return calls


def test_adamw_gets_every_knob_by_name(monkeypatch):
    calls = _captured(monkeypatch, optax, "adamw")
    optimizers._adamw(1e-4)

    (kwargs,) = calls
    assert (kwargs["b1"], kwargs["b2"], kwargs["eps"]) == (config.ADAM_B1, config.ADAM_B2, config.ADAM_EPS)
    assert "weight_decay" in kwargs


def test_muon_gets_every_knob_by_name(monkeypatch):
    calls = _captured(monkeypatch, optax.contrib, "scale_by_muon")
    optimizers._muon(lambda step: 1e-4)

    (kwargs,) = calls
    for knob, value in (("ns_steps", config.MUON_NS_STEPS), ("beta", config.MUON_BETA),
                        ("eps", config.MUON_EPS), ("nesterov", config.MUON_NESTEROV)):
        assert kwargs[knob] == value, knob
    assert "ns_coeffs" in kwargs, "the Newton-Schulz coefficients are stated, not defaulted (#375)"


def test_the_defaults_are_what_every_run_so_far_trained_with():
    """Naming the knobs must not move them: the golden run and every recorded
    recipe pair resolve to exactly these."""
    assert (config.ADAM_B1, config.ADAM_B2, config.ADAM_EPS) == (0.9, 0.999, 1e-8)
    assert (config.WEIGHT_DECAY, config.CLIP_NORM) == (1e-2, 1.0)
    assert (config.MUON_BETA, config.MUON_NS_STEPS, config.MUON_EPS, config.MUON_NESTEROV) == (0.95, 5, 1e-8, True)


def test_the_run_records_every_knob():
    """A model card must be able to rebuild the optimizer from run_metadata.json."""
    import inspect

    from trm.runtime import run_tracker

    source = inspect.getsource(run_tracker)
    for knob in ("ADAM_B1", "ADAM_B2", "ADAM_EPS", "WEIGHT_DECAY", "CLIP_NORM",
                 "MUON_BETA", "MUON_NS_STEPS", "MUON_EPS", "MUON_NESTEROV"):
        assert f'"{knob}": {knob}' in source, knob


def test_the_loss_scaler_growth_interval_is_named_and_unchanged():
    """#368 names it in config; the trainer passes it, and it is still the 256 every
    run so far used (the class default stays the same number)."""
    import inspect

    from trm.runtime import run_tracker
    from trm.train import loss_scale, trainer

    assert config.LOSS_SCALE_GROWTH_INTERVAL == 256 == loss_scale.LOSS_SCALE_GROWTH_INTERVAL
    assert "DynamicLossScale(growth_interval=LOSS_SCALE_GROWTH_INTERVAL)" in inspect.getsource(trainer)
    assert '"LOSS_SCALE_GROWTH_INTERVAL": LOSS_SCALE_GROWTH_INTERVAL' in inspect.getsource(run_tracker)
