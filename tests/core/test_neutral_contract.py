"""The training loop speaks plain LM, and keeps speaking it (#105).

Before this, the loop's idea of "a model" was the shape of the original reasoner:
it did hunch bookkeeping, scheduled a forget cost and a diversity loss, and sent a
refresh signal — so every live architecture had to wear that costume, and Plan A
was integrated by dressing up as the arch we had already proved inert.

Two things keep the inversion from quietly rotting back:
  1. the shared loop names no architecture's machinery, and
  2. both architectures really do train through the same call.

The second is the one that matters; the first is what stops the seam from being
re-opened one convenient attribute at a time.
"""

import io
import pathlib
import tokenize

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from config import LATENT_DIM, MAX_SEQ_LEN
from grad_step import compute_grad_step
from lm_contract import LanguageModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Modules that drive training and must stay architecture-blind. metrics_logger is
# deliberately NOT here: a CSV column named `avg_forget_cost` is telemetry with a
# stable name that historical runs and plot_history still read, not the loop doing
# an architecture's bookkeeping.
SHARED_LOOP = ("trainer.py", "grad_step.py", "validation.py")

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


@pytest.mark.parametrize("module", SHARED_LOOP)
def test_shared_loop_names_no_architecture_internals(module):
    code = _code_without_comments_or_strings(REPO_ROOT / module).lower()
    leaked = [word for word in ARCH_VOCABULARY if word in code]
    assert not leaked, (
        f"{module} mentions {leaked} in code — the shared loop is doing one "
        f"architecture's bookkeeping again. Move it behind an lm_contract hook."
    )


def test_the_leak_scan_can_actually_detect_a_leak():
    """The scan above is only worth having if it fails when it should. model.py is
    the reasoner itself, so its code is full of this vocabulary — if stripping
    comments and strings ever swallowed the whole file, the guard would pass
    vacuously and notice nothing."""
    code = _code_without_comments_or_strings(REPO_ROOT / "model.py").lower()
    found = [word for word in ARCH_VOCABULARY if word in code]
    assert "hunch" in found and "forget" in found, (
        f"the scan found only {found} in model.py — it is no longer reading real code"
    )


def test_contract_defaults_describe_a_plain_stateless_lm():
    """A new architecture must be able to implement `__call__` and stop. If these
    defaults ever stop being no-ops, every future model inherits a chore."""

    class Minimal(LanguageModel):
        def __call__(self, tokens, depth, training=False, new_document=True):
            raise AssertionError("not called")

    m = Minimal()
    assert m.grade_aux([{}, {}], 0) == {}
    assert m.legacy_checkpoint_variables() == {}
    m.reset_state()
    m.end_step(True)
    with m.isolated_state():
        pass


def _tiny_refiner():
    from plan_a_trainer import RefinerForTraining

    return RefinerForTraining(
        128, nnx.Rngs(0), vocab_size=37, num_heads=4,
        encoder_layers=2, max_depth=8, max_seq_len=MAX_SEQ_LEN,
    )


def _reasoner():
    from model import UniversalReasoner

    return UniversalReasoner(LATENT_DIM, nnx.Rngs(5), batch_size=1)


@pytest.mark.parametrize("build", [_tiny_refiner, _reasoner], ids=["refiner", "reasoner"])
def test_both_arches_train_through_the_identical_call(build):
    """The same `compute_grad_step` invocation — no arch branch, no adapter
    shim — drives a model that carries state and grades two regularizers, and one
    that does neither."""
    model = build()
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.integers(1, 37, size=(1, 2 * MAX_SEQ_LEN + 1)), dtype=jnp.int32)

    loss, out, grads, grad_norm = compute_grad_step(
        model, batch, step=0, depth=2, doc_boundary=True)

    assert np.isfinite(float(loss)) and np.isfinite(float(grad_norm))
    assert float(grad_norm) > 0.0, "no gradient flowed"
    assert out.logits is None, "training must return pre-head states, not logits"
    # Whatever the arch reported got graded and added; nothing else was required.
    assert isinstance(out.aux, dict)
