"""Inference must serve the architecture the run selected.

`trm/infer.py` constructed `UniversalReasoner` unconditionally, while `MODEL_ARCH`
has defaulted to `refiner` since Plan A became the live bet. The two have
different param trees, so serving a refiner checkpoint died on a structure
mismatch — inference was simply unavailable for the architecture we actually
train, and nothing said so until you tried it.

This is the third tool found naming one architecture while the run selects
another (`instruments/plots.py` in #181, `instruments/mem_profile` earlier). The
shape of the bug is always the same: a default written when there was only one
arch, left behind when a second arrived.
"""

import pytest

from trm import infer
from trm.config import MODEL_ARCH


def test_build_model_follows_the_arch_selector():
    """The selector is the single source of truth for which network exists; a
    serving path that ignores it cannot load what the trainer wrote."""
    assert type(infer.build_model("refiner")).__name__ == "RefinerForTraining"
    assert type(infer.build_model("reasoner")).__name__ == "UniversalReasoner"


def test_the_default_is_the_configured_arch():
    """Called with no argument — the way run_inference() calls it — it must build
    what MODEL_ARCH says, not a hardcoded choice."""
    expected = "RefinerForTraining" if MODEL_ARCH == "refiner" else "UniversalReasoner"
    assert type(infer.build_model()).__name__ == expected


def test_infer_does_not_hardcode_a_model_class_at_module_level():
    """Both arches must be imported lazily, inside the branch that needs them —
    otherwise importing the serving path drags in Plan A code on a baseline run,
    and (more to the point) a module-level import is how the old hardcoding
    survived unnoticed."""
    from pathlib import Path
    source = Path(infer.__file__).read_text()
    header = source.split("def build_model")[0]
    for cls in ("UniversalReasoner", "RefinerForTraining"):
        assert f"import {cls}" not in header, (
            f"{cls} should be imported inside build_model, not at module level"
        )


@pytest.mark.parametrize("arch", ["refiner", "reasoner"])
def test_both_arches_satisfy_the_contract_the_serving_loop_uses(arch):
    """run_model_inference calls model(tokens, depth=..., training=False,
    new_document=...) and reads `.logits`. Both arches must honour that, or
    switching MODEL_ARCH would fail at generation time rather than at load."""
    from trm.model.contract import LanguageModel
    assert isinstance(infer.build_model(arch), LanguageModel)


# --- sampling is configurable, and the default is not greedy ------------------

def test_the_default_temperature_is_warm_enough_to_see_the_model():
    """0.5 divided the logits, doubling every gap, and with max|logit| ~21 partway
    through the base run that made sampling effectively greedy — which turns an
    undertrained LM into a repetition loop. What you read then is the decoder's
    failure mode, not the weights'."""
    assert infer.DEFAULT_TEMPERATURE == 0.7
    args = infer.build_arg_parser().parse_args([])
    assert args.temperature == 0.7


def test_every_sampling_knob_is_reachable_from_the_cli():
    args = infer.build_arg_parser().parse_args(
        ["--temperature", "0.9", "--top-k", "0", "--top-p", "1.0",
         "--max-new-tokens", "32", "--depth", "8"])
    assert (args.temperature, args.top_k, args.top_p) == (0.9, 0, 1.0)
    assert (args.max_new_tokens, args.depth) == (32, 8)


def test_depth_defaults_to_the_serving_knee_and_is_overridable():
    """Depth is the architecture's whole bet, and the sinusoidal time signal is
    defined at any step — so serving depth should be a dial, not a constant baked
    into the generation loop."""
    from trm.config import INFERENCE_DEPTH
    assert infer.build_arg_parser().parse_args([]).depth == INFERENCE_DEPTH
    assert infer.build_arg_parser().parse_args(["--depth", "16"]).depth == 16
