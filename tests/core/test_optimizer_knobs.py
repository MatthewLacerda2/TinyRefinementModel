"""No hidden defaults on the hot path (#358): every numeric knob of the optimizer is
named in trm/config.py and handed to optax by name.

Before this, AdamW's b1/b2/eps and all of Muon's own knobs were whatever optax
defaulted to, recorded nowhere. An optax upgrade that moved one would have changed
the recipe with no diff in this repo. These tests catch the call sites leaving any
of them to the library again, and pin the values to what every run so far used.
"""

import optax

from trm.settings import CONFIG
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
    optimizers._adamw(CONFIG, 1e-4)

    (kwargs,) = calls
    assert (kwargs["b1"], kwargs["b2"], kwargs["eps"]) == (CONFIG.ADAM_B1, CONFIG.ADAM_B2, CONFIG.ADAM_EPS)
    assert "weight_decay" in kwargs


def test_muon_gets_every_knob_by_name(monkeypatch):
    calls = _captured(monkeypatch, optax.contrib, "scale_by_muon")
    optimizers._muon(CONFIG, lambda step: 1e-4)

    (kwargs,) = calls
    for knob, value in (("ns_steps", CONFIG.MUON_NS_STEPS), ("beta", CONFIG.MUON_BETA),
                        ("eps", CONFIG.MUON_EPS), ("nesterov", CONFIG.MUON_NESTEROV)):
        assert kwargs[knob] == value, knob
    assert "ns_coeffs" in kwargs, "the Newton-Schulz coefficients are stated, not defaulted (#375)"


def test_the_defaults_are_what_every_run_so_far_trained_with():
    """Naming the knobs must not move them: every recorded recipe pair resolves to
    exactly these, except b2, which #359 moved to 0.95 on purpose (ADAM_B2=0.999
    reproduces the runs before it; tests/core/test_adopted_recipe.py)."""
    assert (CONFIG.ADAM_B1, CONFIG.ADAM_B2, CONFIG.ADAM_EPS) == (0.9, 0.95, 1e-8)
    assert (CONFIG.WEIGHT_DECAY, CONFIG.CLIP_NORM) == (1e-2, 1.0)
    assert (CONFIG.MUON_BETA, CONFIG.MUON_NS_STEPS, CONFIG.MUON_EPS, CONFIG.MUON_NESTEROV) == (0.95, 5, 1e-8, True)


def test_the_run_records_every_knob():
    """A model card must be able to rebuild the optimizer from run_metadata.json.
    Every Config field is recorded (tests/core/test_settings.py); these are the ones
    the optimizer needs."""
    from trm.runtime.run_tracker import RunTracker

    recorded = RunTracker.get_hyperparameters(CONFIG)
    for knob in ("ADAM_B1", "ADAM_B2", "ADAM_EPS", "WEIGHT_DECAY", "EMBED_WEIGHT_DECAY", "CLIP_NORM",
                 "MUON_BETA", "MUON_NS_STEPS", "MUON_EPS", "MUON_NESTEROV"):
        assert recorded[knob] == getattr(CONFIG, knob), knob


def test_the_loss_scaler_growth_interval_is_named_and_unchanged():
    """#368 names it in config; the trainer passes it, and it is still the 256 every
    run so far used (the class default stays the same number)."""
    import inspect

    from trm.runtime.run_tracker import RunTracker
    from trm.train import loop, loss_scale

    assert CONFIG.LOSS_SCALE_GROWTH_INTERVAL == 256 == loss_scale.LOSS_SCALE_GROWTH_INTERVAL
    assert "DynamicLossScale(growth_interval=config.LOSS_SCALE_GROWTH_INTERVAL)" in inspect.getsource(loop)
    assert RunTracker.get_hyperparameters(CONFIG)["LOSS_SCALE_GROWTH_INTERVAL"] == 256
