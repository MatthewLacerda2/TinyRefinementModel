"""Production adapter for Plan A (CausalRefiner).

`trm.model.refiner.CausalRefiner` is kept deliberately pure and config-free so the same
architecture runs at toy scale in the ablation harness and at real scale here. This
module is the thin seam that wires it to production config and to the trainer's
neutral model contract (`trm.model.contract.LanguageModel`).

Plan A is a plain causal LM as far as the training loop is concerned: each window
is scored independently, there is no state carried between them, and there are no
auxiliary objectives. So beyond `__call__` it overrides only `training_depth` (the
sampled loop depth, #316) and `capture_trajectory` (for instruments) — the contract's
other defaults (no carried state, no graded extras) are already the truth here. Before
#105 this same fact had to be expressed as *impersonation*: a zero forget-cost and
a zero diversity loss for schedules to multiply, a never-read hunch buffer for the
trainer's bookkeeping to write into, and placeholder zeros for the reasoner's
diagnostics. All of it is gone.

`depth` maps to the refinement depth (sampled per step in training, fixed at
inference) — the one dial Plan A actually uses.
"""

import jax.numpy as jnp

from trm.config import VOCAB_SIZE, MAX_STEPS_LIMIT, COMPUTE_DTYPE
from trm.model.contract import LMOutput, LanguageModel
from trm.model.refiner import CausalRefiner
from trm.train.schedules import sample_reasoning_depth


class RefinerForTraining(LanguageModel):
    """CausalRefiner behind the trainer's neutral contract.

    Shaped by `config` (NUM_HEADS, REFINER_ENCODER_LAYERS, MAX_SEQ_LEN, PAD_TOKEN_ID,
    TIME_SIGNAL, POST_NORM); every architecture knob is overridable so the integration
    test can build a tiny instance of any config. A call that names no depth runs
    config.INFERENCE_DEPTH loops, and the training draws replay from config.DATA_SEED.
    """

    def __init__(self, latent_dim, rngs, config, *, vocab_size=VOCAB_SIZE, num_heads=None,
                 encoder_layers=None, max_depth=MAX_STEPS_LIMIT, max_seq_len=None, pad_token_id=None,
                 dtype=COMPUTE_DTYPE, time_signal=None, post_norm=None):
        num_heads = config.NUM_HEADS if num_heads is None else num_heads
        encoder_layers = config.REFINER_ENCODER_LAYERS if encoder_layers is None else encoder_layers
        max_seq_len = config.MAX_SEQ_LEN if max_seq_len is None else max_seq_len
        pad_token_id = config.PAD_TOKEN_ID if pad_token_id is None else pad_token_id
        time_signal = config.TIME_SIGNAL if time_signal is None else time_signal
        post_norm = config.POST_NORM if post_norm is None else post_norm
        self.max_seq_len = max_seq_len
        self.inference_depth = config.INFERENCE_DEPTH
        self.depth_seed = config.DATA_SEED
        self.pad_token_id = pad_token_id
        self.latent_dim = latent_dim
        self.refiner = CausalRefiner(
            dim=latent_dim, vocab_size=vocab_size, num_heads=num_heads,
            num_encoder_layers=encoder_layers, max_depth=max_depth,
            max_seq_len=max_seq_len, dtype=dtype, rngs=rngs,
            time_signal=time_signal,
            post_norm=post_norm,
        )

    def training_depth(self, micro_step):
        """Loops of the shared block for this micro-step: uniform in
        [1, MAX_STEPS_LIMIT], replayed exactly on resume (#316)."""
        return sample_reasoning_depth(micro_step, self.depth_seed)

    def __call__(self, tokens, depth=None, training=False, new_document=True,
                 logits_at=None):
        # training selects pre-head states vs logits. new_document is part of the
        # contract for models that carry state across windows; Plan A carries none,
        # so each window is a standalone causal LM prediction either way.
        # The default depth is the serving knee (INFERENCE_DEPTH, the 2026-06-19
        # dense-sweep plateau), NOT MAX_STEPS_LIMIT: callers that don't say a depth
        # get the cheapest setting the evidence says is equivalent. Training always
        # passes its sampled depth explicitly.
        depth = self.inference_depth if depth is None else depth
        pad_mask = tokens != self.pad_token_id
        if training:
            # Return pre-head states; the loss does the chunked LM-head projection
            # (#19), avoiding the full [b, s, vocab] f32 logit peak.
            hidden = self.refiner(tokens, depth=depth, pad_mask=pad_mask, return_hidden=True)
            return LMOutput(hidden=hidden)
        return LMOutput(logits=self.refiner(tokens, depth=depth, pad_mask=pad_mask,
                                            logits_at=logits_at))

    def capture_trajectory(self, tokens, depth=None):
        """The refinement trajectory at production scale (#225).

        `[depth+1, b, s, dim]`, index 0 being the encoder output before any
        refinement. 8.8 MB at the live config (depth 8, seq 512, dim 960, f16),
        which is why this has been affordable the whole time — what was expensive
        was the per-pass *logits* it used to be bundled with.

        The final state is the same array the ordinary forward pass produces, so
        an instrument reading this is measuring the computation the model
        actually runs, not a parallel one. No depth: the serving depth.
        """
        depth = self.inference_depth if depth is None else depth
        pad_mask = tokens != self.pad_token_id
        return self.refiner(tokens, depth=depth, pad_mask=pad_mask,
                            return_trajectory=True)

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
