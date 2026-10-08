import functools
from typing import cast

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


def _emit_body(opt, model):
    """nnx.Optimizer.update, with the window's last gradient already in the mean (#293)."""
    params = nnx.pure(nnx.state(model, opt.wrt))
    updates, new_opt_state = opt.tx.emit(nnx.pure(opt.opt_state), params)
    nnx.update(model, optax.apply_updates(cast(optax.Params, params), updates))
    nnx.update(opt.opt_state, nnx.state(new_opt_state))
    opt.step[...] += 1


_emit = nnx.jit(_emit_body, donate_argnums=(0, 1))


# Every micro-step folds its gradient into the window's mean in the program that
# computes it (#293), so no gradient tree crosses a program boundary: one held for a
# second program was ~590 MiB at the arena's peak. A non-finite step folds nothing,
# as the loop's skip always did (#199, #355): a per-element select, not a cond, which
# could not alias the donated state and would copy it whole (trm/train/accumulate.py).
# The grad step reads the parameters and donates the optimizer state only, never the
# two together (#533). With `sample`, it also returns the log row's gradient
# telemetry (#82, #180, #191), read here because the gradient never leaves.
@functools.partial(jax.jit, static_argnames=("graphdef", "opt_graphdef", "z_loss_weight", "sample"),
                   donate_argnames=("opt_state",))
def _step_pure(graphdef, params, rest, opt_graphdef, opt_state, batch_tokens, loss_scale, clip_norm,
               z_loss_weight, sample):
    model = nnx.merge(graphdef, params, rest)
    loss, out, grads, grad_norm = compute_grad_step(
        model, batch_tokens, loss_scale=loss_scale, clip_norm=clip_norm, z_loss_weight=z_loss_weight)
    opt = nnx.merge(opt_graphdef, opt_state)
    state = nnx.pure(opt.opt_state)
    finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
    folded = opt.tx.accumulate(nnx.pure(nnx.state(grads, opt.wrt)), state)
    new_state = jax.tree_util.tree_map(lambda new, old: jnp.where(finite, new, old), folded, state)
    nnx.update(opt.opt_state, nnx.state(new_state))
    sampled = None
    if sample:
        sampled = (grad_zero_fractions(grads), grad_zero_fractions(new_state.acc_grads),
                   optax.global_norm(new_state.acc_grads))
    return nnx.state(model, nnx.Not(nnx.Param)), loss, out, grad_norm, nnx.state(opt), sampled


class HotPath:
    """compute_grad_step and apply_grads for the training loop, with the NNX graph
    walked once instead of on every call (#474), and the gradient folded into the
    window's mean in the program that computes it (#293).

    `nnx.jit` splits its module arguments into graph + state and merges them back on
    every call; on this model that host work held most of the device's idle time
    (#473: 43-58% idle per micro-step). Flax's guidance for a hot loop is to split
    once and `jax.jit` a pure function over the state. That is what `step` does on every
    micro-step: the model and the optimizer are split here and each call passes only
    arrays to a `jax.jit` that merges and runs the same nnx code as before. The
    window's update, once per optimizer step, is `_emit` through the module objects
    and nnx.jit (#533 says why). (`nnx.cached_partial` would be smaller, but cannot cache a
    module holding raw arrays, and the RoPE tables are raw arrays whose checkpoint
    paths must not move.)

    The accumulation window's counter is kept on the host too: `apply_grads` reads it
    from the device (`int(mini_step)`, a sync) after walking the optimizer state, on
    every micro-step. It is read once here and advanced by `commit` exactly as
    LazyMultiSteps advances it; `check_counter` compares it with the device.

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
        if not hasattr(opt.tx, "emit"):
            raise TypeError("HotPath folds every micro-step into an accumulating optimizer "
                            "(trm/train/accumulate.py); this one is not")
        self._tx = opt.tx
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

    def step(self, batch_tokens, loss_scale: jax.typing.ArrayLike = 1.0,
             clip_norm: jax.typing.ArrayLike = jnp.inf, *, sample: bool = False):
        """compute_grad_step on the live state, its gradient folded into the window's mean
        on the device when finite. (loss, out, grad_norm, sampled): `sampled` is the log
        row's (micro-step zero fractions, applied zero fractions, applied norm) when asked.
        Call `commit()` after a finite step; a non-finite one folded nothing."""
        self._live()
        self._rest, loss, out, grad_norm, self._opt_state, sampled = _step_pure(
            self._graphdef, self._params, self._rest, self._opt_graphdef, self._opt_state, batch_tokens,
            loss_scale, clip_norm, self._z_loss_weight, sample)
        return loss, out, grad_norm, sampled

    def commit(self):
        """Advance the window past a finite step; on its last micro-step, run the update."""
        k = int(self._tx._every_k_schedule(self._gstep))
        if self._mini != k - 1:
            self._mini += 1
            return
        self._mini, self._gstep = 0, self._gstep + 1
        model, opt = self.sync()
        _emit(opt, model)

    def check_counter(self):
        """Fail loudly if the host's window counter ever drifts from the device's."""
        state = nnx.pure(self.optimizer.opt_state)
        device = (int(state.mini_step), int(state.gradient_step))
        if device != (self._mini, self._gstep):
            raise RuntimeError(f"accumulation counter drift: host {(self._mini, self._gstep)}, "
                               f"device {device} (#474)")
