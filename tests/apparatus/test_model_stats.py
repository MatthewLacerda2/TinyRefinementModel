"""The truth check for `instruments.model_stats`.

`model_stats` never builds a model — it is arithmetic over `trm/config.py`, so a
parameter count costs no compute and no GPU slot. The obvious way that goes
wrong is drift: someone widens a block or adds a norm, and the formula keeps
confidently reporting the old model. So the guarantee lives here instead. This
builds the **real** model on CPU for both architectures and asserts the formula
matches the instantiated tree exactly — in total, and group by group, so a
mismatch names the piece that moved rather than just the sum.

It is deliberately expensive-ish (two 138M-param CPU builds, ~0.6GB of host RAM
each, freed between). That is the price of the instrument being free, and it is
paid once per test run rather than once per invocation of the tool.
"""

import gc

import pytest
from flax import nnx

from instruments import model_stats
from trm.config import LATENT_DIM, MAX_STEPS_LIMIT, TIME_SIGNAL

# The refiner's group names, and the top-level attribute of the real param tree
# each one owns. `RefinerForTraining` nests everything under `.refiner`; that
# wrapper level is stripped before matching.
REFINER_GROUPS = {
    "embeddings & tied head": {"embed", "time_embed"},
    "encoder layers": {"encoder"},
    "shared refine block": {"refine_block"},
    "heads & norms": {"time_norm", "time_signal_norm", "out_norm", "gate"},
}

REASONER_GROUPS = {
    "embeddings & tied head": {"embed", "time_embed", "shared_token"},
    "encoder stack": {"encoder_stack"},
    "decoder stack": {"decoder_stack"},
    "shared reasoning block": {"reasoning_stack"},
    "heads & norms": {"seq_norm", "meta_proj", "time_norm", "forget_norm",
                      "time_signal_norm", "hunch_norm", "hunch_gate", "raw_tau",
                      "forget_head"},
}


def _real_breakdown(model, groups, strip=None):
    """Group the instantiated model's parameters the way model_stats claims to.

    Any top-level name not listed in `groups` is an error, not an "other"
    bucket: an unclaimed parameter is exactly the drift this test exists to
    catch, and a catch-all would swallow it.
    """
    import jax

    owner = {name: group for group, names in groups.items() for name in names}
    counts = dict.fromkeys(groups, 0)
    state = nnx.state(model, nnx.Param)
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        keys = [str(getattr(part, "key", getattr(part, "idx", part))) for part in path]
        if strip and keys and keys[0] == strip:
            keys = keys[1:]
        top = keys[0]
        assert top in owner, (
            f"parameter tree has a top-level group {top!r} that model_stats does not "
            f"account for (path {keys}). The architecture changed — update the formula "
            f"in instruments/model_stats.py and the mapping here, together."
        )
        counts[owner[top]] += int(leaf.size)
    return counts


def _assert_matches(arch, model, groups, strip=None, **overrides):
    formula = model_stats.param_breakdown(arch, **overrides)
    real = _real_breakdown(model, groups, strip=strip)
    assert set(formula) == set(real)
    for group in real:
        assert formula[group] == real[group], (
            f"{arch}: group {group!r} — formula says {formula[group]:,}, the real model "
            f"has {real[group]:,} (off by {formula[group] - real[group]:+,})"
        )
    assert model_stats.total_params(arch, **overrides) == sum(real.values())


def test_refiner_formula_matches_the_real_model():
    """The shipped refiner, at the exact config the live base run trains."""
    from trm.model.refiner_lm import RefinerForTraining

    model = RefinerForTraining(LATENT_DIM, nnx.Rngs(0))
    try:
        _assert_matches("refiner", model, REFINER_GROUPS, strip="refiner")
    finally:
        del model
        gc.collect()


def test_reasoner_formula_matches_the_real_model():
    """The control arch. Its tree is the one the old plotter counted, so this
    also keeps the formula honest about GQA's narrow K/V projections."""
    from trm.model.reasoner import UniversalReasoner

    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(0), batch_size=1)
    try:
        _assert_matches("reasoner", model, REASONER_GROUPS)
    finally:
        del model
        gc.collect()


@pytest.mark.parametrize(
    "dim,num_heads,encoder_layers",
    # Small enough to build in a blink, and chosen so the MLP-width rounding
    # (8/3 x dim snapped up to a multiple of 64) lands differently each time —
    # the one place in the formula where a transcription slip hides.
    [(128, 4, 1), (192, 4, 2), (240, 6, 3)],
)
def test_refiner_formula_tracks_the_shape_knobs(dim, num_heads, encoder_layers):
    from trm.model.refiner_lm import RefinerForTraining

    overrides = dict(dim=dim, vocab_size=97, num_heads=num_heads,
                     encoder_layers=encoder_layers, max_depth=4)
    model = RefinerForTraining(
        dim, nnx.Rngs(0), vocab_size=97, num_heads=num_heads,
        encoder_layers=encoder_layers, max_depth=4,
    )
    _assert_matches("refiner", model, REFINER_GROUPS, strip="refiner", **overrides)


def test_the_learned_time_table_is_counted_when_it_exists():
    """`TIME_SIGNAL=table` adds a real embedding; `sinusoidal` (the default) is a
    fixed function of the step index and adds nothing. Getting that backwards
    would be a silent few-thousand-parameter lie in either direction."""
    from trm.model.refiner_lm import RefinerForTraining

    common = dict(vocab_size=97, num_heads=4, encoder_layers=1, max_depth=4)
    model = RefinerForTraining(128, nnx.Rngs(0), time_signal="table", **common)
    _assert_matches("refiner", model, REFINER_GROUPS, strip="refiner",
                    dim=128, time_signal="table", **common)

    table = model_stats.total_params("refiner", dim=128, time_signal="table", **common)
    sinusoidal = model_stats.total_params("refiner", dim=128, time_signal="sinusoidal", **common)
    assert table - sinusoidal == (common["max_depth"] + 1) * 128


def test_the_live_run_parameter_count_is_reproduced():
    """The calibration point: the live base run's launch banner prints
    '138.7M parameters' for the refiner at dim960 / 7 encoder layers / 15 heads.
    Pinned exactly so a config change that moves the model shows up as this
    number moving, and not as a quietly different run."""
    assert TIME_SIGNAL in ("table", "sinusoidal", "none")
    assert model_stats.total_params("refiner", time_signal="sinusoidal") == 138_708_224
    # The reasoner control, for the same reason: this is the count the old
    # (model-instantiating) plotter printed, so agreeing with it proves the
    # formula reproduces a number produced by building the thing.
    assert model_stats.total_params("reasoner") == 138_382_849


def test_shared_block_is_one_physical_copy():
    """Looping the shared block costs compute, never weights: doubling the depth
    limit must not change the parameter count."""
    base = model_stats.total_params("refiner")
    assert model_stats.total_params("refiner", max_depth=2 * MAX_STEPS_LIMIT) == base


def test_unknown_override_is_refused():
    """A typo must not silently report the default config under the caller's name."""
    with pytest.raises(TypeError, match="unknown override"):
        model_stats.param_breakdown("refiner", n_heads=8)
    with pytest.raises(ValueError, match="unknown arch"):
        model_stats.param_breakdown("transformer")


def test_vram_terms_are_the_dtypes_the_optimizer_actually_uses():
    """18 bytes per parameter, and each byte is accounted: f32 weights, f32
    gradients, bf16 Adam mu, f32 Adam nu, and the f32 gradient accumulator
    optax.MultiSteps keeps (trm/train/optimizers.py). Miss one and the floor
    stops being a floor."""
    params = model_stats.total_params("refiner")
    train = model_stats.vram_estimate("train", batch=1, depth=8, arch="refiner")

    assert train["parameters (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["AdamW mu (bf16)"] == pytest.approx(params * 2 / model_stats.MIB)
    assert train["AdamW nu (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["gradients (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["MultiSteps accumulated grads (f32)"] == pytest.approx(params * 4 / model_stats.MIB)

    exact_terms = params * 18 / model_stats.MIB
    total = train[model_stats.TOTAL_KEY]
    assert total > exact_terms, "the floor must include the remat-boundary lower bound"
    assert total == pytest.approx(sum(v for k, v in train.items() if k != model_stats.TOTAL_KEY))

    # Inference drops both moments and the gradients; it must be far cheaper.
    infer = model_stats.vram_estimate("infer", batch=1, arch="refiner")
    assert "AdamW mu (bf16)" not in infer and "gradients (f32)" not in infer
    assert infer[model_stats.TOTAL_KEY] < total / 2


def test_the_floor_stays_under_the_measured_peak():
    """A floor that exceeds the measured peak is not a floor, it is a bug. The
    live run's measured peak at this config is ~5.0 GB."""
    floor_gb = (model_stats.vram_estimate("train", batch=1, depth=8, arch="refiner")
                [model_stats.TOTAL_KEY] * model_stats.MIB / 1e9)
    assert 0 < floor_gb < model_stats.MEASURED_PEAK_GB
    assert model_stats.measured_peak_applies("refiner", batch=1)
    assert not model_stats.measured_peak_applies("reasoner", batch=1)
    assert not model_stats.measured_peak_applies("refiner", batch=4)


def test_vram_rejects_a_mode_it_cannot_estimate():
    with pytest.raises(ValueError, match="mode must be"):
        model_stats.vram_estimate("serve")


def test_the_instrument_never_imports_a_model():
    """The whole design in one assertion: importing model_stats must not drag in
    a model module. If this fails, the tool has started building what it is
    supposed to only count."""
    import subprocess
    import sys

    probe = (
        "import sys; import instruments.model_stats as m; "
        "m.param_breakdown('refiner'); m.param_breakdown('reasoner'); "
        "print([k for k in sys.modules if k.startswith('trm.model')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
                         env={**__import__("os").environ, "JAX_PLATFORMS": "cpu",
                              "PYTHONPATH": "."})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("[]"), (
        f"model_stats pulled in model modules: {out.stdout.strip()}"
    )
