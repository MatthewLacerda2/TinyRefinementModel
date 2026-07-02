import jax
import optax
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
        def compute_ce(logits, targets):
            mask = targets != PAD_TOKEN_ID
            return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets) * mask) / jnp.sum(mask).clip(min=1)

        seq1_in, seq1_out = batch_tokens[:, :MAX_SEQ_LEN], batch_tokens[:, 1:MAX_SEQ_LEN+1]
        seq2_in, seq2_out = batch_tokens[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN], batch_tokens[:, MAX_SEQ_LEN+1:2*MAX_SEQ_LEN+1]

        out1 = model(seq1_in, max_steps=max_steps, training=True, should_refresh=should_refresh)
        ce1 = compute_ce(out1.logits, seq1_out)
        
        out2 = model(seq2_in, max_steps=max_steps, training=True, should_refresh=False)
        ce2 = compute_ce(out2.logits, seq2_out)

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
        out2 = out2.replace(logits=None, diag=new_diag)
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
    
    return loss, out, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)
