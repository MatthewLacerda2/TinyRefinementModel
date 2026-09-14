"""Gradient accumulation that runs the optimizer once per window, not once per micro-step.

optax.MultiSteps is jit-friendly in a way that costs the whole optimizer step
every micro-step: its `_do_update` calls the inner transformation on every call
and selects the result with `jnp.where(emit, ...)` (optax/transforms/
_accumulation.py). For AdamW that is a wasted elementwise pass over 137M
parameters 127 times per optimizer step — the flat ~69 ms per micro-step #24
measured, a quarter of a batch-1 step. For Muon it is Newton–Schulz on every
weight matrix 128 times per update, plus its temporaries, which is what pushed
every Muon arm of #26 out of memory while the smoke's single apply fit fine.

This subclass keeps MultiSteps' state and its numbers exactly — the same Welford
mean, the same inner update on the same accumulated gradient at the same step —
and swaps the `where` for a `lax.cond`, so the inner transformation is traced
into the program but executed only on the emitting micro-step. The non-emitting
branch touches nothing but `acc_grads` and the counter.
"""

from __future__ import annotations

import jax
import optax
from optax import tree as otree
from optax._src import numerics
from optax.transforms._accumulation import MultiStepsState


class LazyMultiSteps(optax.MultiSteps):
    def update(self, updates, state, params=None, **extra_args):
        k_steps = self._every_k_schedule(state.gradient_step)
        should_skip_update, skip_state = self._should_skip_update_fn(
            updates, state.gradient_step, params)

        def accumulate(updates, state, params):
            """Fold this micro-step's gradient into the running mean; no optimizer."""
            acc_grads = jax.tree_util.tree_map(
                lambda upd, acc: self._acc_update(upd, acc, n_acc=state.mini_step),
                updates, state.acc_grads)
            new_state = MultiStepsState(
                mini_step=numerics.safe_increment(state.mini_step) % k_steps,
                gradient_step=state.gradient_step,
                inner_opt_state=state.inner_opt_state,
                acc_grads=acc_grads,
                skip_state=skip_state)
            return otree.zeros_like(updates), new_state

        def emit(updates, state, params):
            """The window's last micro-step: the optimizer runs on the mean, once."""
            acc_grads = jax.tree_util.tree_map(
                lambda upd, acc: self._acc_update(upd, acc, n_acc=state.mini_step),
                updates, state.acc_grads)
            final_updates, new_inner_state = self._opt.update(
                acc_grads, state.inner_opt_state, params=params, **extra_args)
            new_state = MultiStepsState(
                mini_step=numerics.safe_increment(state.mini_step) % k_steps,
                gradient_step=numerics.safe_increment(state.gradient_step),
                inner_opt_state=new_inner_state,
                acc_grads=otree.zeros_like(acc_grads),
                skip_state=skip_state)
            return final_updates, new_state

        def do_update(updates, state, params):
            is_last = state.mini_step == (k_steps - 1)
            return jax.lax.cond(is_last, emit, accumulate, updates, state, params)

        def skip_update(updates, state, params):
            return otree.zeros_like(updates), MultiStepsState(
                mini_step=state.mini_step, gradient_step=state.gradient_step,
                inner_opt_state=state.inner_opt_state, acc_grads=state.acc_grads,
                skip_state=skip_state)

        return jax.lax.cond(should_skip_update, skip_update, do_update, updates, state, params)


def multi_steps(inner, every_k_schedule, use_grad_mean=True):
    return LazyMultiSteps(inner, every_k_schedule=every_k_schedule, use_grad_mean=use_grad_mean)
