"""An instrument's default may not be a silent second copy of a config.py constant (#167).

`vram_headroom_smoke` defaulted `--batch` to 1 after #24 changed BATCH_SIZE, and
dim/heads to 512/16 after the shipping model became 960/15. On 2026-08-13 it ran at
its defaults and the number was read as clearance for a batch-2 launch; the run
OOM'd. Sweeping configs is what instruments are for — the failure is a *default*
that answers a question about a model we do not ship, where nobody sees it.

So a flag that shadows a config constant defaults to that constant. A deliberate
off-config default stays possible, declared where a reader will see it:

    CONFIG_DIVERGENCES = {"--batch": "examples per eval forward, not the training micro-batch"}
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Flag → the config.py constant it shadows.
SHADOWS = {
    "--dim": "LATENT_DIM",
    "--heads": "NUM_HEADS",
    "--encoder-layers": "REFINER_ENCODER_LAYERS",
    "--layers": "PLAIN_LAYERS",
    "--batch": "BATCH_SIZE",
    "--seq": "MAX_SEQ_LEN",
    "--seq-len": "MAX_SEQ_LEN",
    "--depth": "MAX_STEPS_LIMIT",
}


def _declared_divergences(tree):
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "CONFIG_DIVERGENCES"):
            return {k: v for k, v in ast.literal_eval(node.value).items() if str(v).strip()}
    return {}


def literal_defaults(source):
    """[(flag, literal default, line)] for shadowing flags not declared as divergences."""
    tree = ast.parse(source)
    declared = _declared_divergences(tree)
    found = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "add_argument"
                and call.args and isinstance(call.args[0], ast.Constant)):
            continue
        flag = call.args[0].value
        default = next((kw.value for kw in call.keywords if kw.arg == "default"), None)
        if (flag in SHADOWS and flag not in declared
                and isinstance(default, ast.Constant) and isinstance(default.value, (int, float))
                and not isinstance(default.value, bool)):
            found.append((flag, default.value, call.lineno))
    return found


INSTRUMENTS = sorted(p for p in (ROOT / "instruments").rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", INSTRUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_instrument_defaults_follow_config(path):
    bad = literal_defaults(path.read_text())
    assert not bad, "\n".join(
        f"{path.relative_to(ROOT)}:{line} {flag} default={value} shadows config.{SHADOWS[flag]} — "
        f"use default={SHADOWS[flag]}, or declare it in CONFIG_DIVERGENCES with a reason"
        for flag, value, line in bad)


def test_the_lint_catches_the_case_that_caused_an_oom():
    assert literal_defaults('ap.add_argument("--batch", type=int, default=1)') == [("--batch", 1, 1)]
    assert not literal_defaults('ap.add_argument("--batch", type=int, default=BATCH_SIZE)')
    assert not literal_defaults('ap.add_argument("--layers", type=int, default=None)')


def test_a_declaration_needs_a_reason():
    silent = 'CONFIG_DIVERGENCES = {"--batch": ""}\nap.add_argument("--batch", default=4)'
    stated = 'CONFIG_DIVERGENCES = {"--batch": "eval forward"}\nap.add_argument("--batch", default=4)'
    assert literal_defaults(silent) and not literal_defaults(stated)


def test_every_shadowed_constant_exists_in_config():
    import trm.config as config
    missing = [c for c in set(SHADOWS.values()) if not hasattr(config, c)]
    assert not missing, missing
