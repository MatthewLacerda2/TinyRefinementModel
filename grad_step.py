import jax
from flax import nnx
import jax.numpy as jnp

from config import (
    MAX_SEQ_LEN,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
)
from schedules import (
    forget_lambda_schedule,
    diversity_lambda_schedule,
)
from losses import chunked_cross_entropy

def zero_fraction_by_group(grad_tree):
    """Fraction of exactly-zero entries in `grad_tree`, one value per top-level
    param group (the model's top-level attributes: 'embed', 'encoder_stack',
    'reasoning_stack', ...).

    Instrument for #82: f16 gradient underflow rounds entries to exactly zero
    without ever going non-finite, so it's invisible to the NaN-streak abort
    and to the global grad norm. Reported per group, not as one number,
    because the embedding's zero-fraction is legitimately huge (only tokens
    present in the micro-batch get a gradient row) — the signal to watch
    lives in the dense-kernel groups (attention projections, MLP, norms),
    which should read ~0 in healthy f16 training.
    """
    zero_counts = {}
    sizes = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grad_tree)[0]:
        group = str(getattr(path[0], 'key', path[0]))
        zero_counts[group] = zero_counts.get(group, 0) + jnp.sum(leaf == 0)
        sizes[group] = sizes.get(group, 0) + leaf.size
    return {group: zero_counts[group] / sizes[group] for group in zero_counts}


def compute_total_loss(ce1, ce2, out1, out2, f_lambda, d_lambda):
    """Assembles the full training loss from both segments' outputs.

    Standalone (not buried in the jitted grad step) so the loss-wiring test can
    assert every cost component actually contributes — a forget-cost term was
    once computed and logged but never added here, and the model spent ~3900
    opt steps not learning to forget.

    Plain CE on both segments plus the two regularizers. The old refinement,
    anchor, and ponder terms are gone: with the reasoning depth randomly sampled
    per training step, "use the extra steps well" is enforced structurally
    instead of through patch losses (which made copying step 1 the optimum).
    """
    return (ce1 + ce2) \
           + d_lambda * (out1.diversity_loss + out2.diversity_loss) \
           + f_lambda * (out1.forget_cost + out2.forget_cost)


@nnx.jit(static_argnames=['max_steps'])
def compute_grad_step(model, batch_tokens, step, max_steps, doc_boundary=False):
    # A document boundary means the hunch cache holds state from an unrelated
    # document — the model should start its slots fresh.
    should_refresh = jnp.any(doc_boundary).squeeze()

    def loss_fn(model):
        # Training models return pre-head states (out.hidden), not full logits, so the
        # CE is scored chunk-by-chunk through the tied embedding (chunked_cross_entropy,
        # #19) — this is what keeps the [b, s, vocab] f32 logit peak off the card.
        embedding = model.embed.embedding[...]

        def ce_of(out, targets):
            return chunked_cross_entropy(out.hidden, embedding, targets, PAD_TOKEN_ID)

        seq1_in, seq1_out = batch_tokens[:, :MAX_SEQ_LEN], batch_tokens[:, 1:MAX_SEQ_LEN+1]
        seq2_in, seq2_out = batch_tokens[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN], batch_tokens[:, MAX_SEQ_LEN+1:2*MAX_SEQ_LEN+1]

        out1 = model(seq1_in, max_steps=max_steps, training=True, should_refresh=should_refresh)
        ce1 = ce_of(out1, seq1_out)

        out2 = model(seq2_in, max_steps=max_steps, training=True, should_refresh=False)
        ce2 = ce_of(out2, seq2_out)

        opt_step = step // ACCUMULATION_STEPS
        f_lambda = forget_lambda_schedule(opt_step)
        d_lambda = diversity_lambda_schedule(opt_step)

        total_loss = compute_total_loss(ce1, ce2, out1, out2, f_lambda, d_lambda)

        # No NaN masking here: a non-finite loss must surface in the train loop
        # (which skips the update and aborts on a streak), not be silently zeroed.
        new_diag = {
            **out2.diag,
            'seg1_ce': jax.lax.stop_gradient(ce1),
            'token_loss': jax.lax.stop_gradient(ce2),
        }
        out2 = out2.replace(logits=None, hidden=None, diag=new_diag)
        return total_loss, out2

    (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    current_hunch = model.hunch_cache[...]
    cleared_hunch = jnp.zeros_like(current_hunch)
    
    # After the step, carry the hunch forward UNLESS a document boundary was hit
    carried_hunch = jax.lax.cond(
        should_refresh,
        lambda: jax.lax.stop_gradient(cleared_hunch),
        lambda: jax.lax.stop_gradient(current_hunch)
    )
    
    model.hunch_cache[...] = carried_hunch

    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))

    # Carried on the aux (not the return tuple) so this instrument doesn't
    # force every one of the ~15 call sites that unpack this function's
    # 4-tuple to change.
    out = out.replace(diag={**out.diag, 'grad_zero_frac': zero_fraction_by_group(grads)})

    return loss, out, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)
