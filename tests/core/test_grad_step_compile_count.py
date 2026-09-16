"""The plain grad step compiles once, not once per sampled depth (#316).

`compute_grad_step` takes `depth` as a static jit argument, which is right for the
looped arches: the refiner unrolls its shared block per depth. The trainer used to
draw `sample_reasoning_depth(step)` for every arch, though, and `PlainTransformer`
discards the argument, so the plain grad step was traced and compiled once per
distinct depth drawn (8 over a run): identical programs, each with its own resident
executable and CUDA graphs on a 6GB card.

Depth is now the architecture's answer (`LanguageModel.training_depth`). Plain answers
None every micro-step, so there is one static value and one program. The looped
arches still answer `sample_reasoning_depth(step)`, a pure function of DATA_SEED and
the micro-step, so a resumed run replays the same depths.

A static argument's effect lives in the jit cache key, not in the model, so a tiny
model shows it. Sequence length is the one knob that cannot shrink: the grad step
splits its batch at the config's MAX_SEQ_LEN.
"""

import inspect

import jax.numpy as jnp
import pytest
from flax import nnx

from trm.config import MAX_SEQ_LEN, MAX_STEPS_LIMIT
from trm.model.plain import PlainTransformer
from trm.train.grad_step import compute_grad_step
from trm.train.schedules import sample_reasoning_depth

MICRO_STEPS = 16
TRACES = []  # one entry per forward traced, appended at trace time only


class _CountingPlain(PlainTransformer):
    """PlainTransformer whose forward records each time jit traces it."""

    def __call__(self, *args, **kwargs):
        TRACES.append(kwargs.get("depth"))
        return super().__call__(*args, **kwargs)


@pytest.fixture
def plain():
    TRACES.clear()
    return _CountingPlain(32, nnx.Rngs(0), vocab_size=37, num_heads=4, num_layers=1,
                          max_seq_len=MAX_SEQ_LEN, pad_token_id=36)


def _run(model, depths):
    """The trainer's call, micro-step by micro-step; returns how many grad-step traces
    happened (each trace runs the forward twice, once per window)."""
    batch = jnp.ones((1, 2 * MAX_SEQ_LEN + 1), dtype=jnp.int32)
    doc_boundary = jnp.zeros((1,), dtype=bool)
    for step, depth in enumerate(depths):
        compute_grad_step(model, batch, jnp.array(step), depth, doc_boundary=doc_boundary,
                          loss_scale=jnp.float32(1.0), clip_norm=jnp.float32(jnp.inf))
    return len(TRACES) // 2


def test_the_plain_grad_step_traces_once_across_micro_steps(plain):
    depths = [plain.training_depth(step) for step in range(MICRO_STEPS)]
    assert set(depths) == {None}, "plain has no depth dial"
    assert _run(plain, depths) == 1


def test_the_old_loop_compiled_one_plain_program_per_sampled_depth(plain):
    """The defect this removes, reproduced: the loop-sampled depths over the same 16
    micro-steps give one trace per distinct value, for one computation."""
    sampled = [sample_reasoning_depth(step) for step in range(MICRO_STEPS)]
    assert len(set(sampled)) > 1, "the draw must vary for this to show anything"
    assert _run(plain, sampled) == len(set(sampled))


def test_the_trainer_asks_the_model_for_its_depth():
    from trm.train import trainer
    source = inspect.getsource(trainer.train_loop)
    assert "depth = model.training_depth(step)" in source
    assert "sample_reasoning_depth" not in source, "the loop no longer samples a depth for every arch"


@pytest.mark.parametrize("module, name", [("trm.model.refiner_lm", "RefinerForTraining"),
                                          ("trm.model.reasoner", "UniversalReasoner")])
def test_looped_arches_keep_the_replayable_draw(module, name):
    """Bit-identical to the old loop: the same depth at every micro-step, so a resumed
    run replays the depths the original trained at. Unbound: no model is built."""
    import importlib
    cls = getattr(importlib.import_module(module), name)
    depths = [cls.training_depth(None, step) for step in range(2048)]
    assert depths == [sample_reasoning_depth(step) for step in range(2048)]
    assert set(depths) == set(range(1, MAX_STEPS_LIMIT + 1))
