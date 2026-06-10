import jax
import optax
from flax import nnx
import jax.numpy as jnp

from config import (
    MAX_SEQ_LEN,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
)

# Light regularizer on expected computation depth.
# We will anneal this using ponder_lambda_schedule.

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

        # stop_gradient on ce1: the refinement term should only push ce2 down toward
        # ce1 as a target. Without this, the gradient through ce1 perversely incentivizes
        # making seq1 worse to reduce the ce2-ce1 gap.
        refinement_regression = jnp.maximum(0.0, ce2 - jax.lax.stop_gradient(ce1))
        refinement_loss = refinement_regression * 0.08
        
        early_penalty = ce1 * 0.03

        from schedules import forget_lambda_schedule, diversity_lambda_schedule, ponder_lambda_schedule
        opt_step = step // ACCUMULATION_STEPS
        f_lambda = forget_lambda_schedule(opt_step)
        d_lambda = diversity_lambda_schedule(opt_step)
        p_lambda = ponder_lambda_schedule(opt_step)

        total_loss = (ce1 + ce2) \
                     + d_lambda * (out1.diversity_loss + out2.diversity_loss) \
                     + f_lambda * (out1.forget_cost + out2.forget_cost) \
                     + refinement_loss \
                     + early_penalty \
                     + p_lambda * (out1.ponder_cost + out2.ponder_cost)

        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
        
        new_halt_diag = {
            **out2.halt_diag,
            'ce1': jax.lax.stop_gradient(ce1),
            'token_loss': jax.lax.stop_gradient(ce2),
            'ponder_cost': jax.lax.stop_gradient(out1.ponder_cost + out2.ponder_cost),
        }
        out2 = out2.replace(logits=None, halt_diag=new_halt_diag)
        return total_loss, out2

    (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    current_hunch = model.hunch_cache.value
    cleared_hunch = jnp.zeros_like(current_hunch)
    
    # After the step, carry the hunch forward UNLESS a document boundary was hit
    carried_hunch = jax.lax.cond(
        should_refresh,
        lambda: jax.lax.stop_gradient(cleared_hunch),
        lambda: jax.lax.stop_gradient(current_hunch)
    )
    
    model.hunch_cache.value = carried_hunch

    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))
    
    return loss, out, grads, grad_norm


@nnx.jit
def apply_grads(opt, grads, model):
    opt.update(model, grads)
