"""The model: a plain causal transformer.

N distinct pre-norm blocks (RoPE attention with QK-norm, SwiGLU MLP) over a token
embedding, and the embedding again as the LM head. What came before it (a shared block
looped to a sampled depth, retired 2026-09-12) is in docs/findings/ and the ROADMAP
graveyard, not here.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import nnx, struct

from trm.config import COMPUTE_DTYPE, VOCAB_SIZE
from trm.model.rope import rope_tables, apply_rope


@struct.dataclass
class LMOutput:
    """One forward pass. Exactly one of `logits` / `hidden` is filled.

    `hidden` (pre-head states, [b, s, d]) is what training returns: the loss
    projects the LM head chunk-by-chunk (#19) so the full [b, s, vocab] f32
    logit tensor is never materialized. Inference fills `logits` instead.
    """

    logits: jnp.ndarray | None = None
    hidden: jnp.ndarray | None = None
    # Telemetry. Read by the metrics logger, never differentiated.
    diag: Dict[str, Any] = struct.field(default_factory=dict)


class CausalAttention(nnx.Module):
    """Multi-head self-attention, RoPE, causal mask folded into an additive bias."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32):
        assert dim % num_heads == 0, "dim must divide num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.k = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.v = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.o = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        cos, sin = rope_tables(max_pos, self.head_dim)
        self.cos, self.sin = cos, sin

    def __call__(self, x, pad_bias=None):
        b, s, d = x.shape
        q = self.q_norm(self.q(x).reshape(b, s, self.num_heads, self.head_dim))
        k = self.k_norm(self.k(x).reshape(b, s, self.num_heads, self.head_dim))
        v = self.v(x).reshape(b, s, self.num_heads, self.head_dim)

        cos = self.cos[:s, None, :]
        sin = self.sin[:s, None, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # q_norm/k_norm run in f32 for stability, so q/k come out f32 while v is in
        # the compute dtype. Cast q/k back so all three match (dot_product_attention
        # requires it) and attention takes the tensor-core path. No-op in f32 (CPU /
        # toy harness); the real-scale f16 run needs it.
        q = q.astype(x.dtype)
        k = k.astype(x.dtype)

        pos = jnp.arange(s)
        causal = pos[:, None] >= pos[None, :]                   # [s, s], True = allowed
        bias = jnp.where(causal, 0.0, -1e9)[None, None, :, :]   # [1, 1, s, s]
        if pad_bias is not None:
            bias = bias + pad_bias                              # pad_bias [b, 1, 1, s]
        # The bias must stay f32: cast to f16 turns -1e9 into -inf (f16 max
        # ~65504), and a fully-masked row would softmax to NaN (#84).
        # dot_product_attention adds the bias to its f32 logits, so f16 q/k/v
        # keep the tensor-core path.
        out = jax.nn.dot_product_attention(q, k, v, bias=bias)
        return self.o(out.reshape(b, s, d))


class Block(nnx.Module):
    """Pre-norm transformer block: causal attention + SwiGLU MLP, zero-init residual."""

    def __init__(self, dim, num_heads, max_pos, rngs, dtype=jnp.float32,
                 post_norm=False):
        self.attn = CausalAttention(dim, num_heads, max_pos, rngs, dtype)
        self.norm1 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        # Post-norm on each residual branch (#235). norm1/norm2 bound the branch
        # INPUT; these bound its OUTPUT, which is the quantity that overflows —
        # a SwiGLU product multiplies two projections, so an input that merely
        # aligns with high-gain directions in both comes out quadratically large.
        self.post_norm = post_norm
        if post_norm:
            self.attn_out_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
            self.mlp_out_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        # SwiGLU's 8/3 width, rounded up to a multiple of 64.
        hidden = ((int(8 * dim / 3) + 63) // 64) * 64
        self.gate_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.up_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.down_proj = nnx.Linear(hidden, dim, kernel_init=jax.nn.initializers.zeros, rngs=rngs, dtype=dtype)

    def __call__(self, x, pad_bias=None, probe=False):
        """The block's output; with `probe`, also the peak |output| of each branch
        (attention, MLP), read in the compute dtype before the add (#536): with an f32
        stream (#357) these f16 matmul outputs are what can still overflow."""
        attn_out = self.attn(self.norm1(x), pad_bias)
        x = x + (self.attn_out_norm(attn_out) if self.post_norm else attn_out)
        h = self.norm2(x)
        mlp_out = self.down_proj(jax.nn.silu(self.gate_proj(h)) * self.up_proj(h))
        x = x + (self.mlp_out_norm(mlp_out) if self.post_norm else mlp_out)
        if probe:
            return x, jnp.stack([jnp.max(jnp.abs(attn_out)), jnp.max(jnp.abs(mlp_out))]).astype(jnp.float32)
        return x


class PlainTransformer(nnx.Module):
    """N distinct causal blocks, a tied LM head, and nothing else.

    Shaped by `config` (NUM_HEADS, PLAIN_LAYERS, MAX_SEQ_LEN, PAD_TOKEN_ID, POST_NORM,
    RESIDUAL_DTYPE). Each is overridable by keyword, so a test can build a tiny
    instance of any config. `pad_token_id` is the id masked out of attention (and,
    by the loss and the probe, out of the targets); `max_seq_len` is the window
    generation pads to.
    """

    def __init__(self, latent_dim, rngs, config, *, vocab_size=VOCAB_SIZE, num_heads=None,
                 num_layers=None, max_seq_len=None, pad_token_id=None, dtype=COMPUTE_DTYPE,
                 post_norm=None, residual_dtype=None):
        num_heads = config.NUM_HEADS if num_heads is None else num_heads
        num_layers = config.PLAIN_LAYERS if num_layers is None else num_layers
        max_seq_len = config.MAX_SEQ_LEN if max_seq_len is None else max_seq_len
        pad_token_id = config.PAD_TOKEN_ID if pad_token_id is None else pad_token_id
        post_norm = config.POST_NORM if post_norm is None else post_norm
        residual_dtype = config.RESIDUAL_DTYPE if residual_dtype is None else residual_dtype
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.latent_dim = latent_dim
        self.dtype = dtype
        # The embedding's output dtype is the residual stream's (#357), never narrower
        # than compute. Nothing else has to change: a block's `x + branch` promotes to
        # the wider of the two, and its norms read the stream at that width and hand
        # the matmuls `dtype`.
        self.embed = nnx.Embed(vocab_size, latent_dim, rngs=rngs,
                               dtype=jnp.promote_types(dtype, residual_dtype))
        self.blocks = nnx.List([
            Block(latent_dim, num_heads, max_seq_len, rngs, dtype,
                  post_norm=post_norm)
            for _ in range(num_layers)
        ])
        self.out_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

    def _stream(self, tokens, keep_states=False):
        """The residual stream through the stack: (final z, act_max per state,
        residual RMS per state, branch output peaks per block [N, 2], states).

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
        branch_peaks = []
        for blk in self.blocks:
            z, peaks = blk(z, pad_bias, probe=True)
            branch_peaks.append(peaks)
            measure(z)
            if states is not None:
                states.append(z)
        return z, jnp.stack(maxes), jnp.stack(rmses), jnp.stack(branch_peaks), states

    def __call__(self, tokens, training=False, logits_at=None):
        """Score `tokens`: pre-head states when `training`, else logits.

        `logits_at` is the generation seam (#206): when it names a position, fill
        `logits` for that position alone, shaped [b, 1, vocab].
        """
        z, act_maxes, act_rmses, branch_peaks, _ = self._stream(tokens)
        z = self.out_norm(z)
        diag = {
            # The scalar every reader since #235 reads, and the #368 alarm watches.
            "act_max": jax.lax.stop_gradient(jnp.max(act_maxes)),
            "act_max_blocks": jax.lax.stop_gradient(act_maxes),
            "act_rms_blocks": jax.lax.stop_gradient(act_rmses),
            # Each branch's f16 output before the add (#536): what the margin alarm
            # watches now that the stream itself is f32.
            "branch_max": jax.lax.stop_gradient(jnp.max(branch_peaks)),
            "branch_max_blocks": jax.lax.stop_gradient(branch_peaks),
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

    def capture_trajectory(self, tokens):
        """The residual stream after every block, for instruments (#391).

        `[N+1, b, s, dim]` in f32: index 0 is the embedding output, index k the
        stream after block k, so the last entry is the state the out-norm and the
        tied head read. Not part of `__call__`, so the jitted training call never
        carries a flag only instruments read.
        """
        *_, states = self._stream(tokens, keep_states=True)
        assert states is not None
        return jnp.stack([state.astype(jnp.float32) for state in states])
