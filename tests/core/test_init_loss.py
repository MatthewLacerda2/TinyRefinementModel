"""Loss-at-init canary: an untrained model must score like a blind guesser.

A model with no knowledge should assign roughly uniform probability over the
vocabulary, which pins its expected CE to a known value (about 11.52 for this
vocab). Far above that means broken init or input pipeline; meaningfully below
means information is leaking into the prediction. The tied embedding head makes
init logits slightly non-uniform, so a modest excess over the ideal is expected
(observed ~0.4 at full scale) — the band below allows it.
"""

import jax.numpy as jnp
import numpy as np
import optax

from config import VOCAB_SIZE


def test_untrained_model_scores_near_uniform_guessing(tiny_model, token_batch):
    out = tiny_model(jnp.asarray(token_batch), max_steps=2, training=False, should_refresh=True)
    targets = np.roll(token_batch, -1, axis=1)[:, :-1]
    ce = float(jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            logits=out.logits[:, :-1], labels=jnp.asarray(targets)
        )
    ))

    uniform_ce = float(np.log(VOCAB_SIZE))
    assert ce > uniform_ce - 0.2, (
        f"Init CE {ce:.3f} is below the uniform-guess value {uniform_ce:.3f} — "
        "an untrained model cannot legitimately beat blind guessing; "
        "something leaks target information."
    )
    assert ce < uniform_ce + 1.2, (
        f"Init CE {ce:.3f} is far above the uniform-guess value {uniform_ce:.3f} — "
        "initialization or the input pipeline is broken."
    )
