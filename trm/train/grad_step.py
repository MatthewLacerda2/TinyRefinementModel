import functools

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from trm.train.losses import chunked_cross_entropy_rows


@nnx.jit(static_argnames=['z_loss_weight'])
def compute_grad_step(model, batch_tokens, loss_scale=1.0, clip_norm=jnp.inf, z_loss_weight=0.0):
    # A row is two windows and the target after them, so the window length is the
    # batch's own; the pad is the one the model masks (#475). `z_loss_weight` is
    # Config.Z_LOSS_WEIGHT (#369), static (a Python float), 0 off: the trainer hands
    # it down through HotPath.
    def loss_fn(model):
        # Training models return pre-head states (out.hidden), not full logits, so the
        # CE is scored chunk-by-chunk through the tied embedding (chunked_cross_entropy,
        # #19) — this is what keeps the [b, s, vocab] f32 logit peak off the card.
        embedding = model.embed.embedding[...]

        window = (batch_tokens.shape[1] - 1) // 2
        seq1_in, seq1_out = batch_tokens[:, :window], batch_tokens[:, 1:window+1]
        seq2_in, seq2_out = batch_tokens[:, window:2*window], batch_tokens[:, window+1:2*window+1]

        out1 = model(seq1_in, training=True)
        out2 = model(seq2_in, training=True)

        # Both windows are scored in ONE chunked-CE scan, stacked on the batch axis:
        # two separate calls duplicated the [vocab, dim] f32 gradient plumbing across
        # custom_vjp boundaries XLA cannot fuse — the ~1.3 GiB temp-arena OOM at
        # dim960 (#128). Per-row sums/counts keep ce1/ce2 numerically identical to
        # the two-call version; only the shared embedding-grad summation order moved.
        b = seq1_in.shape[0]
        hidden = jnp.concatenate([out1.hidden, out2.hidden], axis=0)
        targets = jnp.concatenate([seq1_out, seq2_out], axis=0)
        loss_sums, counts, row_stats = chunked_cross_entropy_rows(  # pyright: ignore[reportGeneralTypeIssues] -- jax leaves custom_vjp's return untyped
            hidden, embedding, targets, model.pad_token_id, z_weight=z_loss_weight)
        counts = jax.lax.stop_gradient(counts).clip(min=1.0)
        ce1 = loss_sums[:b].sum() / counts[:b].sum()
        ce2 = loss_sums[b:].sum() / counts[b:].sum()
        # Window-2 telemetry, as before (weighted by row counts when b > 1).
        logit_stats = {
            'out_entropy': jnp.sum(row_stats['out_entropy'][b:] * counts[b:]) / counts[b:].sum(),
            'logz_mean': jnp.sum(row_stats['logz_mean'][b:] * counts[b:]) / counts[b:].sum(),
            'max_abs_logit': jnp.max(row_stats['max_abs_logit'][b:]),
        }

        total_loss = ce1 + ce2

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

    sq_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)
    grad_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq_norms)))

    # Clip THIS micro-step, not the accumulated mean (#201). MultiSteps wraps the
    # optimizer chain, so optax's clip_by_global_norm only ever sees the average of
    # ACCUMULATION_STEPS of these — and with a p99/p50 ratio above 50, one outlier
    # decides the direction of that average while the clip merely resizes it. The
    # ceiling is a multiple of the run's own typical norm, computed by the caller,
    # so it tracks the model instead of a constant that expires.
    #
    # grad_norm is returned UNCLIPPED: it is the run's oldest continuous telemetry
    # ("raw per-micro-step, BEFORE clip" — grad_zero_fractions' sibling in the CSV),
    # and rewriting its meaning would silently break every historical comparison.
    # A non-finite norm leaves the gradients untouched: the trainer skips the whole
    # micro-step anyway, and 0/inf would turn a skip into a NaN.
    safe_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, 1.0)
    clip_factor = jnp.minimum(1.0, clip_norm / jnp.maximum(safe_norm, 1e-12))
    grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)

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

    Grouping strips wrapper levels holding a single child, so the groups are the
    model's real top-level ones (embed, blocks, out_norm). Reading the numbers —
    #82's caveat, amended by what the unit test showed:
      - the tied token embedding gets gradient on EVERY row through the CE head
        projection, but rare-token magnitudes are small enough to round to zero
        benignly in f16 — embedding-style groups stay excluded from the decision
        scalar (dense_zero_frac_max).
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
    for p, (_, leaf) in zip(paths, leaves, strict=True):
        group = p[min(level, len(p) - 1)]
        zeros[group] = zeros.get(group, 0) + jnp.sum(leaf == 0)
        sizes[group] = sizes.get(group, 0) + leaf.size
    return {g: zeros[g] / sizes[g] for g in zeros}


@jax.jit
def _applied_stats(acc, seen, grads):
    applied = jax.tree_util.tree_map(lambda a, g: a + (g - a) / (seen + 1), acc, grads)
    return grad_zero_fractions(applied), optax.global_norm(applied)


def applied_gradient_stats(opt, grads):
    """(zero fraction per group, global norm) of the gradient the optimizer applies —
    the two numbers the telemetry logs (#191, #180) — WITHOUT materializing it.

    Building the applied gradient whole would cost a full tree; read eagerly at every logging step it
    is a 564 MiB f32 temporary at dim 960 that neither the fit gate nor the
    headroom smoke ever sees (they never reach a logging step). It put the 9-layer
    AdamW run 565 MiB above the smoke's peak and OOM'd every Muon arm (#26).
    Under jit, XLA fuses the fold into the per-leaf reductions and nothing
    tree-sized is allocated.
    """
    state = opt.opt_state
    seen = jax.tree_util.tree_leaves(state.mini_step)[0]
    fracs, norm = _applied_stats(nnx.to_pure_dict(state.acc_grads), seen, nnx.to_pure_dict(grads))
    return fracs, norm


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
def _apply_body(opt, model, grads):
    opt.update(model, grads)


def _accumulate_body(opt, grads):
    """Fold this micro-step's gradient into the optimizer's running mean, in place."""
    state = nnx.pure(opt.opt_state)
    new_state = opt.tx.accumulate(nnx.pure(nnx.state(grads, opt.wrt)), state)
    nnx.update(opt.opt_state, nnx.state(new_state))


_apply = nnx.jit(_apply_body, donate_argnums=(0, 1, 2))
_accumulate = nnx.jit(_accumulate_body, donate_argnums=(0, 1))


def apply_grads(opt, grads, model):
    """One micro-step's worth of optimizer work.

    With an accumulating optimizer (trm/train/accumulate.py) that is either a
    fold into the running mean or, on the window's last micro-step, the real
    update — decided here from the optimizer's own counter, on the host, so no
    branch is traced into the program and both paths donate their buffers. A
    plain optimizer (the smokes' own optax chains) updates every call, as before.
    """
    tx = opt.tx
    if hasattr(tx, "emits_next") and not tx.emits_next(nnx.pure(opt.opt_state)):
        _accumulate(opt, grads)
    else:
        _apply(opt, model, grads)


# The grad step reads the parameters and returns only what it can change: the model's
# non-parameter state (the RoPE tables, unchanged). Nothing is donated or handed back unchanged, so nothing is
# copied. An earlier version passed the whole (model, optimizer) state through a
# donated grad step, and on the card the gradient output reused a donated buffer that
# jit had forwarded unchanged — the accumulator was silently overwritten (#474; the
# CPU ignores donation, so only the GPU identity check showed it).
@functools.partial(jax.jit, static_argnames=("graphdef", "z_loss_weight"))
def _grad_step_pure(graphdef, params, rest, batch_tokens, loss_scale, clip_norm, z_loss_weight):
    model = nnx.merge(graphdef, params, rest)
    loss, out, grads, grad_norm = compute_grad_step(
        model, batch_tokens, loss_scale=loss_scale, clip_norm=clip_norm, z_loss_weight=z_loss_weight)
    return nnx.state(model, nnx.Not(nnx.Param)), loss, out, grads, grad_norm


# Accumulation (63 of every 64 micro-steps) runs over the split optimizer state and
# donates what the nnx.jit version donated (#128). The window's update, once per
# optimizer step, goes through the module objects and the nnx.jit `_apply`. A jax.jit
# over the split state must never donate the params together with the optimizer
# state: on the card, for the full model, that pair gives a different update (an XLA
# aliasing defect; any other donation set is bit-identical — #533's table).
@functools.partial(jax.jit, static_argnames=("opt_graphdef",), donate_argnames=("opt_state", "grads"))
def _accumulate_pure(opt_graphdef, opt_state, grads):
    opt = nnx.merge(opt_graphdef, opt_state)
    _accumulate_body(opt, grads)
    return nnx.state(opt)


class HotPath:
    """compute_grad_step and apply_grads for the training loop, with the NNX graph
    walked once instead of on every call (#474).

    `nnx.jit` splits its module arguments into graph + state and merges them back on
    every call; on this model that host work held most of the device's idle time
    (#473: 43-58% idle per micro-step). Flax's guidance for a hot loop is to split
    once and `jax.jit` a pure function over the state. That is what the grad step
    and the accumulation do (127 of 128 calls per optimizer step): the model and the
    optimizer are split here and each call passes only arrays to a `jax.jit` that
    merges and runs the same nnx code as before. The window's update, once per
    optimizer step, goes through the module objects and the old nnx.jit `_apply`
    (#533 says why). (`nnx.cached_partial` would be smaller, but cannot cache a
    module holding raw arrays, and the RoPE tables are raw arrays whose checkpoint
    paths must not move.)

    The accumulation window's counter is kept on the host too: `apply_grads` reads it
    from the device (`int(mini_step)`, a sync) after walking the optimizer state, on
    every micro-step. It is read once here and advanced exactly as LazyMultiSteps
    advances it; `check_counter` compares it with the device when the caller wants.

    The live state is this object's while the loop runs. `model` / `optimizer` hand
    back the module objects brought up to date, and anything done through them (a
    checkpoint, validation) is picked up by the next call.
    Bit-identical to compute_grad_step + apply_grads, on the CPU and on the card
    (tests/core/test_hot_path.py; on the card, tests/expensive/test_gpu_identity.py, #540).
    """

    def __init__(self, model, opt, *, z_loss_weight):
        # The run's Config.Z_LOSS_WEIGHT (#369), no default: what the loop trains with
        # is the caller's to say, never this class's.
        self._model, self._opt = model, opt
        self._z_loss_weight = float(z_loss_weight)
        self._graphdef, _, _ = nnx.split(model, nnx.Param, ...)
        self._opt_graphdef = nnx.graphdef(opt)
        self._read_objects()
        self._objects_behind = self._state_behind = False
        self._tx = opt.tx if hasattr(opt.tx, "emits_next") else None
        if self._tx is not None:
            state = nnx.pure(opt.opt_state)
            self._mini, self._gstep = int(state.mini_step), int(state.gradient_step)

    def _read_objects(self):
        _, self._params, self._rest = nnx.split(self._model, nnx.Param, ...)
        self._opt_state = nnx.state(self._opt)

    @property
    def model(self):
        return self.sync()[0]

    @property
    def optimizer(self):
        return self.sync()[1]

    def sync(self):
        """The module objects, up to date; whatever is done through them is read back
        by the next hot call."""
        if self._objects_behind:
            nnx.update(self._model, self._params, self._rest)
            nnx.update(self._opt, self._opt_state)
            self._objects_behind = False
        self._state_behind = True
        return self._model, self._opt

    def _live(self):
        if self._state_behind:
            self._read_objects()
            self._state_behind = False
        self._objects_behind = True

    def grad_step(self, batch_tokens, loss_scale: jax.typing.ArrayLike = 1.0,
                  clip_norm: jax.typing.ArrayLike = jnp.inf):
        """compute_grad_step(model, ...), on the live state."""
        self._live()
        self._rest, loss, out, grads, grad_norm = _grad_step_pure(
            self._graphdef, self._params, self._rest, batch_tokens, loss_scale, clip_norm,
            self._z_loss_weight)
        return loss, out, grads, grad_norm

    def apply(self, grads):
        """apply_grads(opt, grads, model), without the per-call graph walk or sync."""
        self._live()
        if self._tx is not None:
            k = int(self._tx._every_k_schedule(self._gstep))
            if self._mini != k - 1:
                self._opt_state = _accumulate_pure(self._opt_graphdef, self._opt_state, grads)
                self._mini = (self._mini + 1) % k
                return
            self._mini, self._gstep = (self._mini + 1) % k, self._gstep + 1
        model, opt = self.sync()
        _apply(opt, model, grads)

    def check_counter(self):
        """Fail loudly if the host's window counter ever drifts from the device's."""
        if self._tx is None:
            return
        state = nnx.pure(self.optimizer.opt_state)
        device = (int(state.mini_step), int(state.gradient_step))
        if device != (self._mini, self._gstep):
            raise RuntimeError(f"accumulation counter drift: host {(self._mini, self._gstep)}, "
                               f"device {device} (#474)")
