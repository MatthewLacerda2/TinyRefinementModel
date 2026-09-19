"""A plain causal transformer — the architecture after depth recurrence was retired.

Plan A looped ONE shared block K times; why that was retired is
`docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md`.

This is the same stack with the loop unrolled into distinct layers, and with the
machinery that only existed to serve the loop removed:

    gone: the shared refine block and its trip count
    gone: the per-pass time signal (which step am I on)
    gone: the retention/refine gate (2*dim x dim, ~13% of a refine step's params)
    gone: the `depth` dial entirely

`depth` stays in the call signature because the training loop's contract passes it,
and is IGNORED here -- a plain transformer has one fixed amount of compute per token.
Accepting and ignoring it is deliberate: the alternative is an arch-specific branch in
the loop, which is what `trm/model/contract.py` exists to prevent.

Reuses `Block` from refiner.py rather than redefining it. The block is the part that
was never in question, and two copies would drift.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from trm.config import (
    COMPUTE_DTYPE,
    MAX_SEQ_LEN,
    NUM_HEADS,
    PAD_TOKEN_ID,
    PLAIN_LAYERS,
    POST_NORM,
    VOCAB_SIZE,
    ZERO_INIT_ATTN_OUT,
)
from trm.model.contract import LMOutput, LanguageModel
from trm.model.refiner import Block


class PlainTransformer(LanguageModel):
    """N distinct causal blocks, a tied LM head, and nothing else.

    Every architecture knob is overridable so a test can build a tiny instance
    without monkeypatching config — the same arrangement RefinerForTraining uses.
    """

    def __init__(self, latent_dim, rngs, *, vocab_size=VOCAB_SIZE, num_heads=NUM_HEADS,
                 num_layers=PLAIN_LAYERS, max_seq_len=MAX_SEQ_LEN,
                 pad_token_id=PAD_TOKEN_ID, dtype=COMPUTE_DTYPE,
                 post_norm=POST_NORM, zero_init_o=ZERO_INIT_ATTN_OUT):
        self.pad_token_id = pad_token_id
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.embed = nnx.Embed(vocab_size, latent_dim, rngs=rngs, dtype=dtype)
        self.blocks = nnx.List([
            Block(latent_dim, num_heads, max_seq_len, rngs, dtype,
                  post_norm=post_norm, zero_init_o=zero_init_o)
            for _ in range(num_layers)
        ])
        self.out_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

    def _stream(self, tokens, keep_states=False):
        """The residual stream through the stack: (final z, act_max per state,
        residual RMS per state, states).

        One body for both the forward pass and `capture_trajectory`, so what an
        instrument reads is the computation the model actually runs. `states` is
        the list [z_0, z_1, ..., z_N] when `keep_states`, else None."""
        # An id outside the table is NOT a local error here: nnx.Embed lowers to
        # jnp.take, whose default mode="fill" returns NaN for an out-of-range index
        # (direct indexing clamps instead, which is why a quick probe suggests
        # otherwise). One NaN then reaches EVERY position, because attention masking
        # is additive -- NaN + (-1e9) is NaN, so a single poisoned key makes every
        # query's softmax row all-NaN, including causally earlier ones. Measured:
        # one bad id turned 18,944 of 18,944 logits non-finite (#233).
        #
        # Clamping converts that into a wrong token at one position instead of a
        # destroyed window. It does NOT tell you the data was wrong -- that check
        # belongs where it can raise, at the data boundary, and is separate.
        tokens = jnp.clip(tokens, 0, self.embed.embedding.shape[0] - 1)

        pad_mask = tokens != self.pad_token_id
        pad_bias = ((pad_mask.astype(jnp.float32) - 1.0) * 1e9)[:, None, None, :]

        z = self.embed(tokens)
        states = [z] if keep_states else None
        # Peak |activation| through the stack, carried out on `diag` so it lands in
        # metrics.csv with everything else.
        #
        # This is the number that decides whether the model can be served in f16 at
        # all, and it has never been measured DURING a run. The 4B champion trained
        # itself to 65,120 against an f16 ceiling of 65,504 -- 0.6% of headroom, and
        # #229's whole-window NaN is what running out looks like. It was found two
        # weeks after the run ended, by hand, because nothing watched it (#235).
        #
        # One max-reduce per block against a forward pass: unmeasurable. Detached,
        # so it cannot perturb the gradient it is reporting on.
        #
        # Kept PER STATE (#392): index 0 is the embedding, index k the stream after
        # block k, so a hot stream says which block made it hot. Beside the max, the
        # RMS: max is the f16 overflow risk (one channel is enough), RMS the typical
        # scale a block's contribution competes against, which is #357's rounding.
        maxes, rmses = [], []

        def measure(state):
            state = state.astype(jnp.float32)
            maxes.append(jnp.max(jnp.abs(state)))
            rmses.append(jnp.sqrt(jnp.mean(jnp.square(state))))

        measure(z)
        for blk in self.blocks:
            z = blk(z, pad_bias)
            measure(z)
            if keep_states:
                states.append(z)
        return z, jnp.stack(maxes), jnp.stack(rmses), states

    def __call__(self, tokens, depth=None, training=False, new_document=True,
                 logits_at=None):
        # depth and new_document are contract arguments this architecture does not
        # use: compute per token is fixed, and no state crosses windows.
        del depth, new_document

        z, act_maxes, act_rmses, _ = self._stream(tokens)
        z = self.out_norm(z)
        diag = {
            # The scalar every reader since #235 reads, and the #368 alarm watches.
            "act_max": jax.lax.stop_gradient(jnp.max(act_maxes)),
            "act_max_blocks": jax.lax.stop_gradient(act_maxes),
            "act_rms_blocks": jax.lax.stop_gradient(act_rmses),
        }

        if training:
            # Pre-head states; the loss projects the tied head per chunk (#19) so the
            # full [b, s, vocab] f32 logit tensor is never materialized.
            return LMOutput(hidden=z.astype(self.dtype), diag=diag)
        if logits_at is not None:
            # Generation reads one row per forward (#206); slicing before the matmul
            # keeps ~a fifth of per-token compute from being spent on rows nobody
            # reads, which XLA cannot remove because the index is traced.
            z = jax.lax.dynamic_slice_in_dim(z, logits_at, 1, axis=1)
        embed_t = self.embed.embedding[...].astype(self.dtype).T
        logits = jnp.matmul(z.astype(self.dtype), embed_t,
                            preferred_element_type=jnp.float32)
        return LMOutput(logits=logits, diag=diag)

    def capture_trajectory(self, tokens, depth=None):
        """The residual stream after every block, for instruments (#391).

        `[N+1, b, s, dim]` in f32: index 0 is the embedding output, index k the
        stream after block k, so the last entry is the state the out-norm and the
        tied head read. Spatial depth where the refiner had recurrent depth, the
        same object. No gate, so the second value is None; `depth` is ignored as
        in `__call__`.
        """
        del depth
        *_, states = self._stream(tokens, keep_states=True)
        return jnp.stack([state.astype(jnp.float32) for state in states]), None
