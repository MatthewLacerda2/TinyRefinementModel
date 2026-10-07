"""The truth check for `instruments.model_stats`.

`model_stats` never builds a model — it is arithmetic over `trm/config.py`, so a
parameter count costs no compute and no GPU slot. The obvious way that goes
wrong is drift: someone widens a block or adds a norm, and the formula keeps
confidently reporting the old model. So the guarantee lives here instead. This
builds the **real** model on CPU and asserts the formula matches the instantiated
tree exactly — in total, and group by group, so a mismatch names the piece that
moved rather than just the sum.

It is deliberately expensive-ish (a 138M-param CPU build, ~0.6GB of host RAM).
That is the price of the instrument being free, and it is paid once per test run
rather than once per invocation of the tool.
"""

import gc

import pytest
from flax import nnx

from instruments import model_stats
from trm.settings import CONFIG

# Each group name, and the top-level attributes of the real param tree it owns.
GROUPS = {
    "embeddings & tied head": {"embed"},
    "blocks": {"blocks"},
    "heads & norms": {"out_norm"},
}


def _real_breakdown(model):
    """Group the instantiated model's parameters the way model_stats claims to.

    Any top-level name not listed in GROUPS is an error, not an "other"
    bucket: an unclaimed parameter is exactly the drift this test exists to
    catch, and a catch-all would swallow it.
    """
    import jax

    owner = {name: group for group, names in GROUPS.items() for name in names}
    counts = dict.fromkeys(GROUPS, 0)
    state = nnx.state(model, nnx.Param)
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        keys = [str(getattr(part, "key", getattr(part, "idx", part))) for part in path]
        top = keys[0]
        assert top in owner, (
            f"parameter tree has a top-level group {top!r} that model_stats does not "
            f"account for (path {keys}). The model changed — update the formula "
            f"in instruments/model_stats.py and the mapping here, together."
        )
        counts[owner[top]] += int(leaf.size)
    return counts


def _assert_matches(model, **overrides):
    formula = model_stats.param_breakdown(**overrides)
    real = _real_breakdown(model)
    assert set(formula) == set(real)
    for group in real:
        assert formula[group] == real[group], (
            f"group {group!r} — formula says {formula[group]:,}, the real model "
            f"has {real[group]:,} (off by {formula[group] - real[group]:+,})"
        )
    assert model_stats.total_params(**overrides) == sum(real.values())


def test_the_formula_matches_the_real_model():
    from trm.model import build_model

    model = build_model(CONFIG, nnx.Rngs(0))
    try:
        _assert_matches(model)
    finally:
        del model
        gc.collect()


@pytest.mark.parametrize("dim,num_heads,num_layers,post_norm",
                         # Small enough to build in a blink, and chosen so the MLP-width
                         # rounding (8/3 x dim snapped up to a multiple of 64) lands
                         # differently each time — where a transcription slip hides.
                         [(128, 4, 1, False), (192, 4, 3, True), (240, 6, 2, False)])
def test_the_formula_tracks_the_shape_knobs(dim, num_heads, num_layers, post_norm):
    from trm.model import build_model

    model = build_model(CONFIG, nnx.Rngs(0), dim=dim, vocab_size=97, num_heads=num_heads,
                        num_layers=num_layers, post_norm=post_norm)
    _assert_matches(model, dim=dim, vocab_size=97, num_heads=num_heads,
                    num_layers=num_layers, post_norm=post_norm)


def test_the_parameter_count_is_reproduced():
    """136.9M at 8 layers — the count the launch banner prints."""
    assert model_stats.total_params(num_layers=8, post_norm=False) == 136_862_144
    assert model_stats.total_params(num_layers=9, post_norm=False) == 147_933_312
    # 8 is the default again since 2026-09-20: batch 2 buys more than the ninth layer
    # (+39% tok/s, #385). The count is the formula's; test_the_formula_matches_the_real_model
    # ties the formula to the instantiated tree, so this pins the default's size, not a measurement.
    assert CONFIG.PLAIN_LAYERS == 8


def test_unknown_override_is_refused():
    """A typo must not silently report the default config under the caller's name."""
    with pytest.raises(TypeError, match="unknown override"):
        model_stats.param_breakdown(n_heads=8)


def test_vram_terms_are_the_dtypes_the_optimizer_actually_uses():
    """18 bytes per parameter, and each byte is accounted: f32 weights, f32
    gradients, bf16 Adam mu, f32 Adam nu, and the f32 gradient accumulator
    optax.MultiSteps keeps (trm/train/optimizers.py). Miss one and the floor
    stops being a floor."""
    params = model_stats.total_params()
    train = model_stats.vram_estimate("train", batch=1)

    assert train["parameters (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["AdamW mu (bf16)"] == pytest.approx(params * 2 / model_stats.MIB)
    assert train["AdamW nu (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["gradients (f32)"] == pytest.approx(params * 4 / model_stats.MIB)
    assert train["MultiSteps accumulated grads (f32)"] == pytest.approx(params * 4 / model_stats.MIB)

    exact_terms = params * 18 / model_stats.MIB
    total = train[model_stats.TOTAL_KEY]
    assert total > exact_terms, "the floor must include the block-input lower bound"
    assert total == pytest.approx(sum(v for k, v in train.items() if k != model_stats.TOTAL_KEY))

    # Inference drops both moments and the gradients; it must be far cheaper.
    infer = model_stats.vram_estimate("infer", batch=1)
    assert "AdamW mu (bf16)" not in infer and "gradients (f32)" not in infer
    assert infer[model_stats.TOTAL_KEY] < total / 2


def test_the_floor_stays_under_the_measured_peak():
    """A floor that exceeds the measured peak is not a floor, it is a bug."""
    for n in (8, 9, 10):
        overrides = {"dim": 960, "num_heads": 15, "num_layers": n, "post_norm": False}
        peak = model_stats.measured_peak(batch=1, **overrides)
        assert peak is not None, overrides
        floor_gb = (model_stats.vram_estimate("train", batch=1, **overrides)
                    [model_stats.TOTAL_KEY] * model_stats.MIB / 1e9)
        assert 0 < floor_gb < peak.gb, overrides


def test_a_measured_peak_is_quoted_only_for_its_exact_config():
    shape = {"dim": 960, "num_heads": 15, "post_norm": False}
    assert model_stats.measured_peak(batch=4, num_layers=9, **shape) is None
    assert model_stats.measured_peak(batch=1, num_layers=11, **shape) is None
    assert model_stats.measured_peak(batch=1, num_layers=9, **{**shape, "post_norm": True}) is None
    assert model_stats.measured_peak(batch=1, num_layers=9, **shape).gb == pytest.approx(4437 * 2**20 / 1e9)


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
        "m.param_breakdown(); "
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


def test_the_report_runs():
    """The report runs end to end, as `python -m instruments.report --model-only`
    would, in a child interpreter (#325)."""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-m", "instruments.report", "--model-only"],
                          capture_output=True, text=True, timeout=600,
                          cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-2000:]}\nstdout:\n{proc.stdout[-2000:]}"
    assert "total" in proc.stdout
