"""What the training loop requires of a model — and nothing more.

The loop speaks plain language-model: *here are tokens and a depth; give me
predictions (or pre-head states for the chunked loss), plus any auxiliary terms
you want graded.* Every architecture implements this; none of them has to wear
another architecture's costume.

The four hooks below all default to the stateless, no-extra-objective case, so a
plain LM implements `__call__` and stops. An architecture that carries state
between windows, or that has its own regularizers, overrides exactly the hooks
it needs — and the shared loop stays free of its bookkeeping (#105).
"""

from contextlib import contextmanager
from typing import Any, Dict

import jax.numpy as jnp
from flax import nnx, struct


@struct.dataclass
class LMOutput:
    """One forward pass. Exactly one of `logits` / `hidden` is filled.

    `hidden` (pre-head states, [b, s, d]) is what training returns: the loss
    projects the LM head chunk-by-chunk (#19) so the full [b, s, vocab] f32
    logit tensor is never materialized. Inference fills `logits` instead.
    """

    logits: jnp.ndarray = None
    hidden: jnp.ndarray = None
    # Unweighted auxiliary objectives, by name. The trainer never interprets
    # these — it hands them back to grade_aux() and adds whatever comes out.
    # A model with no extra objectives leaves this empty; it does NOT report
    # zeros to keep a schedule company.
    aux: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    # Telemetry. Read by the metrics logger, never differentiated.
    diag: Dict[str, Any] = struct.field(default_factory=dict)


class LanguageModel(nnx.Module):
    """Base class carrying the neutral defaults. Subclass and override as needed."""

    def __call__(self, tokens, depth, training=False, new_document=True,
                 logits_at=None) -> LMOutput:
        """Score `tokens` using `depth` units of compute.

        `new_document` says whether this window starts a fresh document — a fact
        about the data stream, which a model carrying state across windows needs
        and a stateless one can ignore.

        `logits_at` is the generation seam (#206): when it names a position, fill
        `logits` for that position ALONE, shaped [b, 1, vocab]. Generation emits
        one token per forward and reads one row, so projecting the tied head over
        every position is ~a fifth of per-token compute spent on rows that are
        thrown away — and the index is traced, so XLA cannot remove them. Unlike
        the other three arguments this one is not optional to honour: a model that
        accepted it and ignored it would return full logits and the caller would
        read row 0. Implement it or do not accept it — an arch that omits it from
        its signature fails loudly at the call, which is the safe direction.
        """
        raise NotImplementedError

    def grade_aux(self, window_aux, opt_step):
        """Weight this step's auxiliary terms into named scalars for the loss.

        `window_aux` is the list of per-window `LMOutput.aux` dicts, in window
        order. Returns {name: weighted scalar}; the trainer adds the values to
        the cross-entropy in iteration order. Any schedule the weights follow
        belongs here, in the architecture that owns the objective.
        """
        return {}

    def reset_state(self):
        """Drop whatever is carried between windows. Default: nothing is carried."""

    def end_step(self, new_document):
        """Settle carried state at the end of a training step. Default: no-op."""

    @contextmanager
    def isolated_state(self):
        """Forward passes inside leave the carried training state untouched, so
        an eval probe can never perturb the run measuring it. Default: there is
        nothing to protect."""
        yield

    def legacy_checkpoint_variables(self):
        """Variables that *past* versions of this model saved and this one no
        longer defines, as {name: zeros_like_leaf}.

        Restoring is a strict structural match, so a checkpoint written before a
        buffer was removed cannot be read by a model that lacks it. Declaring
        the buffer here lets checkpoint_utils match the on-disk shape and then
        discard the value — without weakening the check that catches a
        checkpoint genuinely missing weights the model needs.
        """
        return {}
