"""Generation must build one executable, not two (#207).

`refresh` was a static jit argument, and `generate_text` flips it every
`HUNCH_REFRESH_EVERY` tokens. A static argument is part of the compilation cache
key, so alternating it compiled the sampling step twice — two resident programs
and two sets of CUDA graphs for one computation.

The flag is inert on the live architecture: `RefinerForTraining.__call__` accepts
`new_document` and never reads it, because Plan A carries no state between
windows. On the reasoner it *is* read, but only through `jax.lax.cond`, which
takes a traced predicate natively. So the fix is to stop making it static rather
than to special-case an architecture — and these guard both halves of that: the
duplicate compile is gone, and the flag still has no effect on refiner output.

The cost is not abstract. `trm/config.py` blames this class of driver-memory
pressure — "28 alive graphs ... the driver cannot instantiate a CUDA command
buffer" — for squeezing batch-2 from *outside* the BFC arena, on a 6GB card that
has no room to waste on a second copy of the same program.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from trm import infer
from trm.config import MAX_SEQ_LEN

# Deliberately tiny: the defect lives in the jit cache key, not in the model, so
# nothing here needs the live config's dimensions. RefinerForTraining makes every
# architecture knob overridable for exactly this reason.
TOY_DIM = 32
TOY_VOCAB = 37
TOY_HEADS = 4
# Sequence length is the one knob that cannot shrink: generate_text pads to the
# config's MAX_SEQ_LEN, so a toy model has to agree with it. Everything else does.
TOY_SEQ_LEN = MAX_SEQ_LEN
TOY_PAD = TOY_VOCAB - 1
TOY_DEPTH = 2
TOY_TOP_K = 8          # must not exceed TOY_VOCAB
PROMPT = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def toy_refiner():
    from trm.model.refiner_lm import RefinerForTraining

    return RefinerForTraining(
        TOY_DIM, nnx.Rngs(0), vocab_size=TOY_VOCAB, num_heads=TOY_HEADS,
        encoder_layers=1, max_seq_len=TOY_SEQ_LEN, pad_token_id=TOY_PAD,
    )


@pytest.fixture(scope="module")
def padded_tokens():
    tokens = jnp.full((1, TOY_SEQ_LEN), TOY_PAD, dtype=jnp.int32)
    return tokens.at[0, : len(PROMPT)].set(jnp.array(PROMPT, dtype=jnp.int32))


def _logits(model, tokens, refresh):
    return infer.get_logits_for_token(
        model, tokens, len(PROMPT) - 1, refresh,
        TOY_TOP_K, 0.9, 0.7, TOY_DEPTH,
    )


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


def test_flipping_refresh_does_not_trigger_a_second_compile(toy_refiner, padded_tokens):
    """The defect, stated directly. Warm the cache with one value of the flag,
    then flip it: a correctly-traced argument compiles nothing further."""
    _logits(toy_refiner, padded_tokens, True).block_until_ready()

    compiles = _compilations_during(
        lambda: _logits(toy_refiner, padded_tokens, False).block_until_ready())

    assert compiles == 0, (
        f"flipping `refresh` compiled {compiles} more executable(s) — it is back in "
        f"the jit cache key, so generation is building two programs for one computation")


def test_the_flag_does_not_change_what_the_refiner_predicts(toy_refiner, padded_tokens):
    """The premise the fix rests on: Plan A carries no state across windows, so
    `new_document` cannot matter. If this ever fails, dropping the staticness was
    silently changing predictions and the whole change is wrong."""
    fresh = _logits(toy_refiner, padded_tokens, True)
    carried = _logits(toy_refiner, padded_tokens, False)

    assert jnp.array_equal(fresh, carried), (
        "refresh changed the refiner's logits — it is not inert after all")


def test_refresh_is_not_a_static_argument():
    """Structural, and the one that actually fails if someone re-adds it. The
    behavioural test above needs a cold cache to be meaningful; this does not."""
    from pathlib import Path

    source = Path(infer.__file__).read_text()
    signature = source.split("def get_logits_for_token")[0].split("@partial")[-1]

    assert "'refresh'" not in signature and '"refresh"' not in signature, (
        "`refresh` is static again — it flips every HUNCH_REFRESH_EVERY tokens, so "
        "as a cache key it doubles the number of compiled programs")


def test_the_arguments_that_must_stay_static_did_not_get_swept_up():
    """`top_k` and `top_p` gate Python-level branches in `_temperature_truncate`
    (and `lax.top_k` needs a static k); `depth` is the refinement loop's trip
    count. Tracing any of them breaks generation outright, so the fix has to be
    surgical rather than a blanket removal."""
    from pathlib import Path

    source = Path(infer.__file__).read_text()
    signature = source.split("def get_logits_for_token")[0].split("@partial")[-1]

    for name in ("top_k", "top_p", "depth"):
        assert f"'{name}'" in signature, f"{name} must remain a static jit argument"


def test_generation_still_runs_with_the_flag_flipping(toy_refiner, padded_tokens):
    """End to end through `generate_text`, which is where `refresh` actually
    alternates (every HUNCH_REFRESH_EVERY tokens). The unit tests above use one
    call at a time; this is the loop that made the duplicate compile happen, and
    it is also where a traced boolean would surface if anything downstream still
    wanted a concrete one.

    Seeded, so it doubles as a determinism check on the generation path.
    """
    import tiktoken

    enc = tiktoken.get_encoding("r50k_base")
    del padded_tokens  # generate_text tokenizes its own prompt

    def run():
        return infer.generate_text(
            toy_refiner, enc, "ab", max_new_tokens=2 * infer.HUNCH_REFRESH_EVERY,
            temperature=0.7, top_k=TOY_TOP_K, top_p=0.9, depth=TOY_DEPTH,
            seed=42, quiet=True,
        )

    assert run() == run(), "seeded generation is not reproducible"
