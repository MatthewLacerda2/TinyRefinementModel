"""Production adapter for Plan A (CausalRefiner).

`plan_a_model.CausalRefiner` is kept deliberately pure and config-free so the same
architecture runs at toy scale in the ablation harness and at real scale here. This
module is the thin seam that lets the existing production trainer drive it: it wraps
the refiner in the exact interface UniversalReasoner exposes — `__call__(tokens,
max_steps, training, should_refresh) -> ReasonerOutput` plus a `hunch_cache` buffer —
so the grad step, validation probe, and Orbax checkpoint plumbing work unchanged.

Plan A has no cross-window state, no forget gate, and no slot-diversity term, so the
machinery the trainer threads through degrades to honest no-ops:
  - forget_cost and diversity_loss are exactly zero, so their loss schedules
    (forget_lambda, diversity_lambda) multiply zero and contribute nothing;
  - the two training windows are independent (no carried state), so should_refresh
    is ignored — each window is a standalone causal LM prediction;
  - hunch_cache is a vestigial zero buffer, never read in the forward, kept only so
    the trainer's `model.hunch_cache[...]` bookkeeping writes stay valid.
`max_steps` maps to the refinement depth (sampled per step in training, fixed at
inference) — the one dial Plan A actually uses.
"""

import jax.numpy as jnp
from flax import nnx

from config import (
    VOCAB_SIZE,
    NUM_HEADS,
    MAX_SEQ_LEN,
    MAX_STEPS_LIMIT,
    PAD_TOKEN_ID,
    COMPUTE_DTYPE,
    REFINER_ENCODER_LAYERS,
    REFINER_TIME_SIGNAL,
    CHUNKED_ATTENTION,
)
from layers import ReasonerOutput
from plan_a_model import CausalRefiner


class RefinerForTraining(nnx.Module):
    """CausalRefiner in the UniversalReasoner-shaped interface the trainer expects.

    Config constants are the defaults; every architecture knob is overridable so the
    integration test can build a tiny instance without monkeypatching config.
    """

    def __init__(self, latent_dim, rngs, *, vocab_size=VOCAB_SIZE, num_heads=NUM_HEADS,
                 encoder_layers=REFINER_ENCODER_LAYERS, max_depth=MAX_STEPS_LIMIT,
                 max_seq_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID, dtype=COMPUTE_DTYPE,
                 chunked_attention=CHUNKED_ATTENTION, time_signal=REFINER_TIME_SIGNAL):
        self.pad_token_id = pad_token_id
        self.refiner = CausalRefiner(
            dim=latent_dim, vocab_size=vocab_size, num_heads=num_heads,
            num_encoder_layers=encoder_layers, max_depth=max_depth,
            max_seq_len=max_seq_len, dtype=dtype, rngs=rngs,
            chunked_attention=chunked_attention, time_signal=time_signal,
        )
        # Vestigial: written by the trainer's hunch bookkeeping, never read here.
        # Kept tiny ([1, 1, dim]) since it carries no information.
        self.hunch_cache = nnx.Variable(jnp.zeros((1, 1, latent_dim)))

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True):
        # training / should_refresh are part of the baseline interface and have no
        # effect here (no dropout, no carried state) — accepted and ignored.
        pad_mask = tokens != self.pad_token_id
        zero = jnp.array(0.0)
        diag = {"temporal_drift": zero, "forget_density": zero, "tau": zero}
        if training:
            # Return pre-head states; the loss does the chunked LM-head projection
            # (#19), avoiding the full [b, s, vocab] f32 logit peak.
            hidden = self.refiner(tokens, depth=max_steps, pad_mask=pad_mask, return_hidden=True)
            return ReasonerOutput(
                logits=None, hidden=hidden, forget_cost=zero, diversity_loss=zero,
                diag=diag, final_shared=None,
            )
        logits = self.refiner(tokens, depth=max_steps, pad_mask=pad_mask)
        return ReasonerOutput(
            logits=logits, forget_cost=zero, diversity_loss=zero,
            diag=diag, final_shared=None,
        )

    @property
    def embed(self):
        # Uniform accessor so grad_step's chunked CE reaches the tied embedding the
        # same way for both arches. A property (not a stored attribute) so nnx does
        # not see a duplicate of the refiner's embedding in the param tree.
        return self.refiner.embed
