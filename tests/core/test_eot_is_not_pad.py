"""The document separator is a token, not padding (#373).

pad_token_id was the tokenizer's end-of-text, so every real EOT prefill writes between
documents was masked as pad: never a loss target (no learned "the document ends
here", so nothing ever taught the model to stop), never an attention key (a token
after a boundary could not see that one had passed). With the pad moved to 50257 both
come back; a real pad is still masked."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from trm.config import EOT_TOKEN_ID, MAX_SEQ_LEN, VOCAB_SIZE

NEW_PAD = 50257


def test_eot_is_a_loss_target_when_it_is_not_the_pad():
    from trm.train.losses import chunked_cross_entropy_rows

    hidden = jnp.ones((1, 8, 4))
    embedding = jnp.ones((VOCAB_SIZE, 4)) * 0.01
    targets = jnp.array([[5, 6, EOT_TOKEN_ID, 7, 8, NEW_PAD, NEW_PAD, NEW_PAD]])

    _, counts_new, _ = chunked_cross_entropy_rows(hidden, embedding, targets, NEW_PAD, 4)
    _, counts_old, _ = chunked_cross_entropy_rows(hidden, embedding, targets, EOT_TOKEN_ID, 4)
    assert float(counts_new[0]) == 5, "the EOT is scored, the three trailing pads are not"
    assert float(counts_old[0]) == 7, "the old convention dropped the EOT"


def test_eot_is_an_attention_key_when_it_is_not_the_pad():
    """Change the token AT the EOT position: a later position must notice under the
    new pad (EOT is a key), and must not notice when that position is a real pad."""
    from trm.model.plain import PlainTransformer

    model = PlainTransformer(16, nnx.Rngs(0), num_heads=2, num_layers=1, max_seq_len=MAX_SEQ_LEN,
                             pad_token_id=NEW_PAD)
    base = np.full((1, MAX_SEQ_LEN), 11, dtype=np.int32)
    base[0, :12] = np.arange(100, 112)

    def later_logits(at_five):
        tokens = base.copy()
        tokens[0, 5] = at_five
        return np.asarray(model(jnp.asarray(tokens), training=False).logits[0, 10], dtype=np.float32)

    assert not np.allclose(later_logits(EOT_TOKEN_ID), later_logits(123)), "EOT must be seen"
    # A real pad stays invisible: tests/core/test_model_invariants.py holds that
    # (a run of leading pads cannot move the real tokens after it).


def test_the_heldout_ce_never_scores_the_separator_under_either_pad():
    """Every val CE on record excluded EOT (it was the pad). A pair that changes the
    pad must still compare the same positions, so the probe maps EOT targets to pad."""
    from trm.train.validation import heldout_targets

    targets = jnp.array([[5, EOT_TOKEN_ID, 7, NEW_PAD]])
    assert heldout_targets(targets, NEW_PAD).tolist() == [[5, NEW_PAD, 7, NEW_PAD]]
    assert heldout_targets(targets, EOT_TOKEN_ID).tolist() == [[5, EOT_TOKEN_ID, 7, NEW_PAD]]


def test_the_default_pad_leaves_eot_a_real_token():
    """The shipped default, adopted for the base run (owner, 2026-09-20): the pad is
    the unused id, so EOT is trained like any other token and the model can learn to
    stop. The old id stays reachable to reproduce a run recorded under it."""
    from trm import config

    assert config.PAD_TOKEN_ID == NEW_PAD == 50257
    assert EOT_TOKEN_ID == 50256 and config.EOT_TOKEN_ID != config.PAD_TOKEN_ID
