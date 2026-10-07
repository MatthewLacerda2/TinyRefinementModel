"""Generation must build one executable, not one per token (#207).

Every argument of the sampling step that changes from token to token (the position
being read, the temperature) must be traced: a static argument is part of the jit
cache key, so one that moves compiles the step again — another resident program and
another set of CUDA graphs on a 6GB card (#207 is the incident). The arguments that
gate Python-level branches must stay static, or generation breaks outright.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm import infer
from trm.settings import CONFIG

# Deliberately tiny: the defect lives in the jit cache key, not in the model, so
# nothing here needs the live config's dimensions.
TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
TOY_LAYERS = 1
# generate_text pads to the model's own window, so this could shrink too; it stays at
# MAX_SEQ_LEN so the sampling step runs at the shape it serves.
TOY_SEQ_LEN = CONFIG.MAX_SEQ_LEN
TOY_PAD = TOY_VOCAB - 1
TOY_TOP_K = 8          # must not exceed TOY_VOCAB
PROMPT = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def toy():
    from trm.model.plain import PlainTransformer

    return PlainTransformer(
        TOY_DIM, nnx.Rngs(0), CONFIG, vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS, max_seq_len=TOY_SEQ_LEN, pad_token_id=TOY_PAD,
    )


@pytest.fixture(scope="module")
def padded_tokens():
    tokens = jnp.full((1, TOY_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return tokens.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


def _logits(model, tokens, position, temperature=0.7):
    return infer.get_logits_for_token(model, tokens, position, TOY_TOP_K, 0.9, temperature)


def _compilations_during(thunk):
    """XLA compilations triggered by `thunk`.

    Counted at `compile_or_get_cached`, the single funnel every jit lowering
    passes through. Private API, deliberately: there is no public counter, and a
    test that cannot see compilations cannot guard against an extra one.
    """
    import jax._src.compiler as compiler

    count = 0
    original = compiler.compile_or_get_cached

    def counting(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    compiler.compile_or_get_cached = counting
    try:
        thunk()
    finally:
        compiler.compile_or_get_cached = original
    return count


def test_the_next_token_does_not_trigger_a_second_compile(toy, padded_tokens):
    """The defect, stated directly. Warm the cache at one position and temperature,
    then move both, the way generation does: a traced argument compiles nothing."""
    _logits(toy, padded_tokens, len(PROMPT) - 1).block_until_ready()

    compiles = _compilations_during(
        lambda: _logits(toy, padded_tokens, len(PROMPT), temperature=1.0).block_until_ready())

    assert compiles == 0, (
        f"the next token compiled {compiles} more executable(s) — an argument that "
        f"changes per token is back in the jit cache key")


def test_only_the_branching_arguments_are_static():
    """`top_k` and `top_p` gate Python-level branches in `_temperature_truncate` (and
    `lax.top_k` needs a static k), so they must stay static. Nothing that changes per
    token may be: the behavioural test above needs a cold cache to be meaningful;
    this does not."""
    from pathlib import Path

    source = Path(infer.__file__).read_text()
    signature = source.split("def get_logits_for_token")[0].split("@partial")[-1]

    for name in ("top_k", "top_p"):
        assert f"'{name}'" in signature, f"{name} must remain a static jit argument"
    for name in ("token_idx", "temperature"):
        assert f"'{name}'" not in signature and f'"{name}"' not in signature, (
            f"{name} changes per token, so as a static argument it compiles the step again")
