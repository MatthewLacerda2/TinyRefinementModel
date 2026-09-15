"""Gradient accumulation that runs the optimizer once per window, not once per micro-step.

optax.MultiSteps is jit-friendly in a way that costs the whole optimizer step
every micro-step: its `_do_update` calls the inner transformation on every call
and selects the result with `jnp.where(emit, ...)` (optax/transforms/
_accumulation.py). For AdamW that is a wasted pass over every parameter 127
times per optimizer step; for Muon it is Newton–Schulz on every weight matrix
128 times per update, with its temporaries — the transient that OOM'd every
Muon arm of #26 while the smoke's single apply fit fine.

The first fix here wrapped the two branches in `lax.cond`. That removed the
compute and added a worse transient: a cond cannot alias its operands to its
outputs, so every micro-step copied the whole optimizer state plus the updates
— 1.85 GiB at dim 960 — and the arms died again, ten to thirty opt steps in.

So there is no branch inside the program at all. The trainer (through
`grad_step.apply_grads`) knows from the optimizer's own counter whether this
micro-step ends the window, and calls one of two jitted functions: `accumulate`,
which folds the gradient into the running mean in place, or `update`, which folds
it, runs the inner optimizer once, and resets. Both donate their buffers. The
state is optax's MultiStepsState unchanged — same Welford mean, same inner update
on the same accumulated gradient at the same step, so checkpoints restore either
way and the numbers are bit-identical to optax.MultiSteps.
"""

from __future__ import annotations

import jax
import optax
from optax import tree as otree
from optax._src import numerics
from optax.transforms._accumulation import MultiStepsState


class LazyMultiSteps(optax.MultiSteps):
    def _fold(self, updates, state):
        return jax.tree_util.tree_map(
            lambda upd, acc: self._acc_update(upd, acc, n_acc=state.mini_step),
            updates, state.acc_grads)

    def emits_next(self, state) -> bool:
        """Whether the next call ends the window. A host-side read of the counter,
        so the caller can pick `accumulate` or `update` without a traced branch."""
        k_steps = self._every_k_schedule(state.gradient_step)
        return int(state.mini_step) == int(k_steps) - 1

    def accumulate(self, updates, state):
        """A non-emitting micro-step: fold the gradient into the running mean."""
        k_steps = self._every_k_schedule(state.gradient_step)
        return MultiStepsState(
            mini_step=numerics.safe_increment(state.mini_step) % k_steps,
            gradient_step=state.gradient_step,
            inner_opt_state=state.inner_opt_state,
            acc_grads=self._fold(updates, state),
            skip_state=state.skip_state)

    def update(self, updates, state, params=None, **extra_args):
        """The window's last micro-step: the optimizer runs on the mean, once.

        Only correct on an emitting step — `apply_grads` guarantees that by
        checking `emits_next` first. Called on any other step it would apply the
        inner optimizer to a partial mean."""
        k_steps = self._every_k_schedule(state.gradient_step)
        acc_grads = self._fold(updates, state)
        final_updates, new_inner_state = self._opt.update(
            acc_grads, state.inner_opt_state, params=params, **extra_args)
        new_state = MultiStepsState(
            mini_step=numerics.safe_increment(state.mini_step) % k_steps,
            gradient_step=numerics.safe_increment(state.gradient_step),
            inner_opt_state=new_inner_state,
            acc_grads=otree.zeros_like(acc_grads),
            skip_state=state.skip_state)
        return final_updates, new_state


def multi_steps(inner, every_k_schedule, use_grad_mean=True):
    return LazyMultiSteps(inner, every_k_schedule=every_k_schedule, use_grad_mean=use_grad_mean)
