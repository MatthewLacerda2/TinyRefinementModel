"""The shared training loop names no architecture's machinery (#105).

The source-scan half of the neutral-contract guard, split out of
test_neutral_contract.py (#325) because it needs neither jax nor conftest, so CI's
lint job runs it in seconds. Why the seam exists, and the behavioural half that
trains both arches through one call: test_neutral_contract.py.
"""

import pathlib
import tokenize

import pytest

# Needs neither jax, numpy nor tests/conftest.py: CI runs it in the seconds-long
# lint job instead of the jax-heavy pytest job (#325).
pytestmark = pytest.mark.jaxfree

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Modules that drive training and must stay architecture-blind. Named explicitly
# rather than globbed over trm/train/, because two siblings there legitimately carry
# this vocabulary: runtime/metrics.py writes a CSV column called `avg_forget_cost`
# (telemetry with a stable name that historical runs and instruments/plots.py still
# read), and train/schedules.py still holds the reasoner's two lambda schedules,
# which its own grade_aux imports. Neither is the shared loop doing an
# architecture's bookkeeping — the three below are the loop.
SHARED_LOOP = (
    "trm/train/trainer.py",
    "trm/train/grad_step.py",
    "trm/train/validation.py",
)

# The reasoner is the arch whose vocabulary this guard hunts for, so it doubles as
# the scan's own control sample (see the detector test below).
REASONER = "trm/model/reasoner.py"

# Vocabulary belonging to one specific architecture. If the loop mentions any of
# it in *code*, some model's internals have leaked back into the shared path.
ARCH_VOCABULARY = ("hunch", "forget", "diversity", "temporal_drift", "slot", "refresh")


def _code_without_comments_or_strings(path):
    """Source with comments and string literals stripped, so prose about the past
    (the forget-cost incident is worth explaining) doesn't trip the scan."""
    kept = []
    with open(path, "rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                kept.append(tok.string)
    return " ".join(kept)


def test_the_scanned_paths_still_exist():
    """These are file paths, not imports, so a package move slides straight past
    every rename tool. Fail with a sentence that says what to do about it."""
    missing = [p for p in (*SHARED_LOOP, REASONER) if not (REPO_ROOT / p).is_file()]
    assert not missing, (
        f"this guard points at files that no longer exist: {missing}. The layout "
        f"moved — update SHARED_LOOP/REASONER, don't delete the check."
    )


@pytest.mark.parametrize("module", SHARED_LOOP)
def test_shared_loop_names_no_architecture_internals(module):
    code = _code_without_comments_or_strings(REPO_ROOT / module).lower()
    leaked = [word for word in ARCH_VOCABULARY if word in code]
    assert not leaked, (
        f"{module} mentions {leaked} in code — the shared loop is doing one "
        f"architecture's bookkeeping again. Move it behind a trm/model/contract.py hook."
    )


def test_the_leak_scan_can_actually_detect_a_leak():
    """The scan above is only worth having if it fails when it should. The reasoner
    is the architecture whose internals this hunts for, so its code is full of the
    vocabulary — if stripping comments and strings ever swallowed the whole file, or
    a rename left the scan pointed at nothing, the guard would pass vacuously and
    notice nothing."""
    code = _code_without_comments_or_strings(REPO_ROOT / REASONER).lower()
    found = [word for word in ARCH_VOCABULARY if word in code]
    assert "hunch" in found and "forget" in found, (
        f"the scan found only {found} in reasoner.py — it is no longer reading real code"
    )
