"""The VRAM sizer measures the path a launch runs, not an adjacent one (#161).

It once reported batch 2 cheaper than batch 1: platform allocator (cannot
fragment), a 150ms nvidia-smi poll reported as a peak, its own f32-moment
optimizer, and no optimizer apply or validation probe in the measurement. These
guard each of those from coming back; the numbers themselves need the card and
are in the PR.
"""

import ast
import pathlib

from instruments import vram_headroom_smoke as smoke

SOURCE = pathlib.Path(smoke.__file__).read_text()
TREE = ast.parse(SOURCE)


def test_it_uses_the_production_optimizer_not_its_own():
    assert "from trm.train.optimizers import optimizer_chain" in SOURCE
    assert "optax.chain" not in SOURCE and "mu_dtype" not in SOURCE


def test_the_default_run_crosses_an_optimizer_apply():
    parser_defaults = {ast.unparse(kw.value) for call in ast.walk(TREE) if isinstance(call, ast.Call)
                       and call.args and getattr(call.args[0], "value", None) == "--micro-steps"
                       for kw in call.keywords if kw.arg == "default"}
    assert parser_defaults == {"CONFIG.ACCUMULATION_STEPS + 1"}, "default must be ACCUMULATION_STEPS + 1"


def test_the_validation_probe_is_inside_the_measurement():
    assert "_val_ce_sums(" in SOURCE


def test_the_peak_is_read_from_the_allocator_not_a_poll():
    assert "peak_bytes_in_use" in SOURCE and "bytes_limit" in SOURCE
    assert "ENV_DIVERGENCES" not in SOURCE, "it now runs production's allocator; nothing to declare"
