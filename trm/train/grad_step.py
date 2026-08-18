import jax
from flax import nnx
import jax.numpy as jnp

from trm.config import (
    MAX_SEQ_LEN,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
)
from trm.train.losses import chunked_cross_entropy_rows

def compute_total_loss(ce1, ce2, graded_aux):
    """Assembles the full training loss: CE on both windows, plus whatever
    auxiliary terms the architecture asked to have graded.

    Standalone (not buried in the jitted grad step) so the loss-wiring test can
    assert every cost the model reports actually contributes — a forget-cost term
    was once computed and logged but never added here, and the model spent ~3900
    opt steps not learning to forget.

    This function knows no architecture's cost names: `graded_aux` arrives
    already weighted from the model's own `grade_aux` (#105). Terms are added one
    at a time in iteration order rather than summed as a group, so the arithmetic
    an architecture's recorded trajectory was produced under is reproducible.
    """
    total = ce1 + ce2
    for term in graded_aux.values():
        total = total + term
    return total


@nnx.jit(static_argnames=['depth'])
def compute_grad_step(model, batch_tokens, step, depth, doc_boundary=False, loss_scale=1.0):
    # Whether this batch opens a new document is a fact about the data stream;
    # what a model does with it (start fresh, or continue from carried state) is
    # the model's business.
    new_document = jnp.any(doc_boundary).squeeze()

    def loss_fn(model):
        # Training models return pre-head states (out.hidden), not full logits, so the
        # CE is scored chunk-by-chunk through the tied embedding (chunked_cross_entropy,
        # #19) — this is what keeps the [b, s, vocab] f32 logit peak off the card.
        embedding = model.embed.embedding[...]

        seq1_in, seq1_out = batch_tokens[:, :MAX_SEQ_LEN], batch_tokens[:, 1:MAX_SEQ_LEN+1]
        seq2_in, seq2_out = batch_tokens[:, MAX_SEQ_LEN:2*MAX_SEQ_LEN], batch_tokens[:, MAX_SEQ_LEN+1:2*MAX_SEQ_LEN+1]

        out1 = model(seq1_in, depth=depth, training=True, new_document=new_document)
        out2 = model(seq2_in, depth=depth, training=True, new_document=False)

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
        graded_aux = model.grade_aux([out1.aux, out2.aux], opt_step)

        total_loss = compute_total_loss(ce1, ce2, graded_aux)

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
        # Scaled for the backward pass only (#199): every intermediate gradient is
        # loss_scale times larger while it is held in f16, so the small ones clear
        # the subnormal floor instead of rounding to exactly zero. Undone below,
        # before anything reads a gradient — in exact arithmetic this is a no-op.
        return total_loss * loss_scale, out2

    (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Unscale before ANYTHING reads these (#199): the optimizer must see the true
    # gradient, and grad_norm has to stay comparable with every step this run
    # already recorded. An overflow to inf survives the division as inf, so the
    # trainer's non-finite branch still catches it — that is what tells the scaler
    # it went too high.
    loss = loss / loss_scale
    grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)

    # Let the model settle whatever it carries into the next step. A stateless
    # architecture does nothing here.
    model.end_step(new_document)

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
