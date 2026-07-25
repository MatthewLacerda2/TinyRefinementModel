"""Production adapter for Plan A (CausalRefiner).

`plan_a_model.CausalRefiner` is kept deliberately pure and config-free so the same
architecture runs at toy scale in the ablation harness and at real scale here. This
module is the thin seam that wires it to production config and to the trainer's
neutral model contract (`lm_contract.LanguageModel`).

Plan A is a plain causal LM as far as the training loop is concerned: each window
is scored independently, there is no state carried between them, and there are no
auxiliary objectives. So it implements `__call__` and nothing else — the contract's
defaults (no carried state, no graded extras) are already the truth here. Before
#105 this same fact had to be expressed as *impersonation*: a zero forget-cost and
a zero diversity loss for schedules to multiply, a never-read hunch buffer for the
trainer's bookkeeping to write into, and placeholder zeros for the reasoner's
diagnostics. All of it is gone.

`depth` maps to the refinement depth (sampled per step in training, fixed at
inference) — the one dial Plan A actually uses.
"""

import jax.numpy as jnp

from trm.config import (
    VOCAB_SIZE,
    NUM_HEADS,
    MAX_SEQ_LEN,
    MAX_STEPS_LIMIT,
    INFERENCE_DEPTH,
    PAD_TOKEN_ID,
    COMPUTE_DTYPE,
    REFINER_ENCODER_LAYERS,
    CHUNKED_ATTENTION,
    TIME_SIGNAL,
)
from trm.model.contract import LMOutput, LanguageModel
from trm.model.refiner import CausalRefiner


class RefinerForTraining(LanguageModel):
    """CausalRefiner behind the trainer's neutral contract.

    Config constants are the defaults; every architecture knob is overridable so the
    integration test can build a tiny instance without monkeypatching config.
    """

    def __init__(self, latent_dim, rngs, *, vocab_size=VOCAB_SIZE, num_heads=NUM_HEADS,
                 encoder_layers=REFINER_ENCODER_LAYERS, max_depth=MAX_STEPS_LIMIT,
                 max_seq_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID, dtype=COMPUTE_DTYPE,
                 chunked_attention=CHUNKED_ATTENTION, time_signal=TIME_SIGNAL):
        self.pad_token_id = pad_token_id
        self.latent_dim = latent_dim
        self.refiner = CausalRefiner(
            dim=latent_dim, vocab_size=vocab_size, num_heads=num_heads,
            num_encoder_layers=encoder_layers, max_depth=max_depth,
            max_seq_len=max_seq_len, dtype=dtype, rngs=rngs,
            chunked_attention=chunked_attention, time_signal=time_signal,
        )

    def __call__(self, tokens, depth=INFERENCE_DEPTH, training=False, new_document=True):
        # training selects pre-head states vs logits. new_document is part of the
        # contract for models that carry state across windows; Plan A carries none,
        # so each window is a standalone causal LM prediction either way.
        # The default depth is the serving knee (INFERENCE_DEPTH, the 2026-06-19
        # dense-sweep plateau), NOT MAX_STEPS_LIMIT: callers that don't say a depth
        # get the cheapest setting the evidence says is equivalent. Training always
        # passes its sampled depth explicitly.
        pad_mask = tokens != self.pad_token_id
        if training:
            # Return pre-head states; the loss does the chunked LM-head projection
            # (#19), avoiding the full [b, s, vocab] f32 logit peak.
            hidden = self.refiner(tokens, depth=depth, pad_mask=pad_mask, return_hidden=True)
            return LMOutput(hidden=hidden)
        return LMOutput(logits=self.refiner(tokens, depth=depth, pad_mask=pad_mask))

    def legacy_checkpoint_variables(self):
        # Every refiner checkpoint written before #105 carries the vestigial
        # [1, 1, dim] hunch buffer the old trainer wrote into. It never held
        # information — the forward never read it — so restore matches its shape
        # and drops the value. Removing this entry orphans the champion base-run
        # weights (and the runs the #134 time machine revives).
        return {"hunch_cache": jnp.zeros((1, 1, self.latent_dim))}

    @property
    def embed(self):
        # Uniform accessor so grad_step's chunked CE reaches the tied embedding the
        # same way for both arches. A property (not a stored attribute) so nnx does
        # not see a duplicate of the refiner's embedding in the param tree.
        return self.refiner.embed
