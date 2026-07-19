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
from losses import chunked_cross_entropy_rows

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

        seq1_in, seq1_out = batch_tokens[:, :MAX_SEQ_LEN], batch_tokens[:, 1:MAX_SEQ_LEN+1]
        seq2_in, seq2_out = batch_tokens[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN], batch_tokens[:, MAX_SEQ_LEN+1:2*MAX_SEQ_LEN+1]

        out1 = model(seq1_in, max_steps=max_steps, training=True, should_refresh=should_refresh)
        out2 = model(seq2_in, max_steps=max_steps, training=True, should_refresh=False)

        # Both windows are scored in ONE chunked-CE scan, stacked on the batch axis:
        # two separate calls duplicated the [vocab, dim] f32 gradient plumbing across
        # custom_vjp boundaries XLA cannot fuse — the ~1.3 GiB temp-arena OOM at
        # dim960 (#128). Per-row sums/counts keep ce1/ce2 numerically identical to
        # the two-call version; only the shared embedding-grad summation order moved.
        b = seq1_in.shape[0]
        hidden = jnp.concatenate([out1.hidden, out2.hidden], axis=0)
        targets = jnp.concatenate([seq1_out, seq2_out], axis=0)
        loss_sums, counts, row_stats = chunked_cross_entropy_rows(
            hidden, embedding, targets, PAD_TOKEN_ID)
        counts = jax.lax.stop_gradient(counts).clip(min=1.0)
        ce1 = loss_sums[:b].sum() / counts[:b].sum()
        ce2 = loss_sums[b:].sum() / counts[b:].sum()
        # Window-2 telemetry, as before (weighted by row counts when b > 1).
        logit_stats = {
            'out_entropy': jnp.sum(row_stats['out_entropy'][b:] * counts[b:]) / counts[b:].sum(),
            'logz_mean': jnp.sum(row_stats['logz_mean'][b:] * counts[b:]) / counts[b:].sum(),
            'max_abs_logit': jnp.max(row_stats['max_abs_logit'][b:]),
        }

        opt_step = step // ACCUMULATION_STEPS
        f_lambda = forget_lambda_schedule(opt_step)
        d_lambda = diversity_lambda_schedule(opt_step)

        total_loss = compute_total_loss(ce1, ce2, out1, out2, f_lambda, d_lambda)

        # No NaN masking here: a non-finite loss must surface in the train loop
        # (which skips the update and aborts on a streak), not be silently zeroed.
        # Logit-scale thermometer (#80): the CE scan's telemetry over window 2 —
        # same segment 'token_loss' reads — so entropy/log-Z drift (collapse or
        # blur) is visible in the metrics stream instead of surfacing as loss
        # weirdness. Measurement only; the CE backward ignores its cotangent.
        new_diag = {
            **out2.diag,
            **jax.lax.stop_gradient(logit_stats),
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

    return loss, out, grads, grad_norm


def _path_key_name(key):
    # jax path entries come in several flavors (DictKey.key, GetAttrKey.name,
    # SequenceKey.idx); normalize them all to a plain string.
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def grad_zero_fractions(grads):
    """Fraction of exactly-zero entries per top-level param group (#82).

    f16 gradient underflow is silent: entries round to exactly zero, the loss
    plateaus, and the NaN-streak abort never fires because underflow isn't NaN.
    The global grad norm can't show a tail of layers that quietly froze, so the
    no-loss-scaling dtype policy (config.py) gets measured instead of assumed.

    Grouping strips wrapper levels holding a single child (the refiner
    adapter's params all live under 'refiner'), so both arches report their
    real top-level groups (embed, encoder, refine_block, ...). Reading the
    numbers — #82's caveat, amended by what the unit test showed:
      - time_embed rows for unsampled depths are legitimately zero; the tied
        token embedding, by contrast, gets gradient on EVERY row through the
        CE head projection, but rare-token magnitudes are small enough to
        round to zero benignly in f16 — embedding-style groups stay excluded
        from the decision scalar (dense_zero_frac_max) either way.
      - zero-init down_proj kernels block all gradient to gate/up_proj, so
        block groups carry a large *structural* zero fraction until the first
        optimizer updates land — attribute early readings to that, not
        underflow. The dense signal, once training is moving, is ~0 healthy.
    """
    leaves = jax.tree_util.tree_flatten_with_path(grads)[0]
    paths = [tuple(_path_key_name(k) for k in path) for path, _ in leaves]

    level = 0
    while len({p[min(level, len(p) - 1)] for p in paths}) == 1 \
            and any(len(p) > level + 1 for p in paths):
        level += 1

    zeros, sizes = {}, {}
    for p, (_, leaf) in zip(paths, leaves):
        group = p[min(level, len(p) - 1)]
        zeros[group] = zeros.get(group, 0) + jnp.sum(leaf == 0)
        sizes[group] = sizes.get(group, 0) + leaf.size
    return {g: zeros[g] / sizes[g] for g in zeros}


def dense_zero_frac_max(zero_fracs):
    """Worst zero-fraction among the dense groups — the #82 decision-rule scalar.

    Embedding-style groups are excluded: their zeros are unused rows, not
    underflow. Accepts the grad_zero_fractions dict (jax or python scalars).
    """
    dense = [v for k, v in zero_fracs.items() if "embed" not in k]
    return max(dense) if dense else float("nan")


# Donation (#128): without it, this step holds input AND output copies of the
# whole optimizer state (MultiSteps f32 accumulator, mu, nu), the params, and
# the grads at once — a 4.65GiB buffer assignment that, not compute_grad_step,
# was the true dim960 OOM. Donating aliases old state to new in place (~2.2GiB
# saved). The caller must not touch `grads` after this call — the trainer
# samples its zero-frac telemetry BEFORE applying, for exactly this reason.
@nnx.jit(donate_argnums=(0, 1, 2))
def apply_grads(opt, grads, model):
    opt.update(model, grads)
