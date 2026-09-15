"""A plain causal transformer — the architecture after depth recurrence was retired.

Plan A looped ONE shared block K times. That mechanism works on sequential
composition ([[plan-a-depth-recurrence-works]]) and is actively suppressed on
language: the trained gate routes to 6 of 960 channels on prose, the second refine
pass costs 5.7 nats at 0.66B and nothing at 3.99B, and bounding the activation scale
does not recover it. Full record in
`docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md`.

So this is the same stack with the loop unrolled into distinct layers, and with the
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
                 post_norm=POST_NORM):
        self.pad_token_id = pad_token_id
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.max_seq_len, self.num_heads, self.head_dim = max_seq_len, num_heads, latent_dim // num_heads
        self.embed = nnx.Embed(vocab_size, latent_dim, rngs=rngs, dtype=dtype)
        self.blocks = nnx.List([
            Block(latent_dim, num_heads, max_seq_len, rngs, dtype,
                  post_norm=post_norm)
            for _ in range(num_layers)
        ])
        self.out_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

    def __call__(self, tokens, depth=None, training=False, new_document=True,
                 logits_at=None):
        # depth and new_document are contract arguments this architecture does not
        # use: compute per token is fixed, and no state crosses windows.
        del depth, new_document

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
        act_max = jnp.max(jnp.abs(z.astype(jnp.float32)))
        for blk in self.blocks:
            z = blk(z, pad_bias)
            act_max = jnp.maximum(act_max, jnp.max(jnp.abs(z.astype(jnp.float32))))
        z = self.out_norm(z)
        diag = {"act_max": jax.lax.stop_gradient(act_max)}

        if training:
            # Pre-head states; the loss projects the tied head per chunk (#19) so the
            # full [b, s, vocab] f32 logit tensor is never materialized.
            return LMOutput(hidden=z.astype(self.dtype), diag=diag)
        if logits_at is not None:
            # Generation reads one row per forward (#206); slicing before the matmul
            # keeps ~a fifth of per-token compute from being spent on rows nobody
            # reads, which XLA cannot remove because the index is traced.
            z = jax.lax.dynamic_slice_in_dim(z, logits_at, 1, axis=1)
        return LMOutput(logits=self._head(z), diag=diag)

    def _head(self, z):
        embed_t = self.embed.embedding[...].astype(self.dtype).T
        return jnp.matmul(z.astype(self.dtype), embed_t, preferred_element_type=jnp.float32)

    # --- decoding with a KV cache (#153) -------------------------------------
    # Generation used to re-run the whole padded window through every block for
    # every emitted token: O(N * S^2) where O(N * S) is the honest cost. The cache
    # holds every block's rotated K and V for the positions seen so far, and a
    # step attends its one new query against them. Same math as the full window:
    # a position's keys are exactly those at or before it either way; only the
    # float reduction order over the (masked, zero-weight) tail differs.

    def prefill(self, tokens):
        """Run an unpadded prompt [b, n], fill the caches, return the logits for
        the next token ([b, vocab]) and the caches."""
        b, n = tokens.shape
        tokens = jnp.clip(tokens, 0, self.embed.embedding.shape[0] - 1)
        positions = jnp.arange(n)
        z = self.embed(tokens)
        caches = []
        for blk in self.blocks:
            k, v = blk.attn.kv(blk.norm1(z), positions)
            cache_k = jnp.zeros((b, self.max_seq_len, self.num_heads, self.head_dim), k.dtype).at[:, :n].set(k)
            cache_v = jnp.zeros((b, self.max_seq_len, self.num_heads, self.head_dim), v.dtype).at[:, :n].set(v)
            caches.append((cache_k, cache_v))
            z = blk(z)
        z = self.out_norm(z[:, -1:, :])
        return self._head(z)[:, 0, :], caches

    def decode_step(self, token, caches, pos):
        """One new token [b, 1] at absolute position `pos` (traced): logits for
        the token after it, and the caches with this position written."""
        token = jnp.clip(token, 0, self.embed.embedding.shape[0] - 1)
        positions = jnp.full((1,), pos, dtype=jnp.int32)
        keep = jnp.arange(self.max_seq_len) <= pos                       # keys at or before pos
        bias = jnp.where(keep, 0.0, -1e9)[None, None, None, :]           # [1, 1, 1, L]
        z = self.embed(token)
        new_caches = []
        for blk, (cache_k, cache_v) in zip(self.blocks, caches):
            h = blk.norm1(z)
            k, v = blk.attn.kv(h, positions)
            cache_k = jax.lax.dynamic_update_slice_in_dim(cache_k, k, pos, axis=1)
            cache_v = jax.lax.dynamic_update_slice_in_dim(cache_v, v, pos, axis=1)
            new_caches.append((cache_k, cache_v))
            attn_out = blk.attn.attend(h, cache_k, cache_v, bias, positions)
            z = z + (blk.attn_out_norm(attn_out) if blk.post_norm else attn_out)
            h2 = blk.norm2(z)
            mlp_out = blk.down_proj(jax.nn.silu(blk.gate_proj(h2)) * blk.up_proj(h2))
            z = z + (blk.mlp_out_norm(mlp_out) if blk.post_norm else mlp_out)
        z = self.out_norm(z)
        return self._head(z)[:, 0, :], new_caches
