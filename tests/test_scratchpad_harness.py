"""Wiring guards for the serial-scratchpad proof harness (#38).

These pin the properties the ablation's interpretation rests on: the task's
recurrence is what the design doc says, serial and parallel arms differ by
exactly one variable (data flow, not parameters), and the slot grades are a
live gradient path — not a bonus the optimizer can ignore."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scratchpad_harness import ScratchpadNet, affine_chain_task, n_params, train_one_arm

K, M, DIM = 3, 7, 32


def test_affine_chain_matches_reference():
    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(0), 16)
    a, b = np.asarray(tokens[:, 0::2]), np.asarray(tokens[:, 1::2])
    r = np.zeros(16, dtype=np.int64)
    for k in range(K):
        r = (r * a[:, k] + b[:, k]) % M
        np.testing.assert_array_equal(np.asarray(subs[:, k]), r)
    assert (a >= 1).all(), "a tokens must be nonzero (invertible affine maps)"


def test_serial_and_parallel_share_one_param_tree():
    serial = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                           serial=True, rngs=nnx.Rngs(0))
    parallel = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                             serial=False, rngs=nnx.Rngs(0))
    assert n_params(serial) == n_params(parallel)
    s_leaves = jax.tree_util.tree_leaves(nnx.state(serial, nnx.Param))
    p_leaves = jax.tree_util.tree_leaves(nnx.state(parallel, nnx.Param))
    for sl, pl in zip(s_leaves, p_leaves):
        np.testing.assert_array_equal(np.asarray(sl), np.asarray(pl))


def test_slot_order_flow_is_the_one_variable():
    """Slot k+1 must read slot k in the serial arm and must NOT in the parallel
    arm — measured as gradient flow from slot 0's index embedding into slot 1's
    grade logits."""
    tokens, _ = affine_chain_task(K, M)(jax.random.PRNGKey(1), 4)

    def slot_grad(serial, of_slot, wrt_row):
        """grad of sum(slot_logits[:, of_slot]) w.r.t. slot_index embedding row."""
        model = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                              serial=serial, rngs=nnx.Rngs(0))
        graph, state = nnx.split(model)

        def f(emb_row):
            st = jax.tree_util.tree_map(lambda x: x, state)  # shallow copy
            emb = st["slot_index"]["embedding"][...]
            st["slot_index"]["embedding"][...] = emb.at[wrt_row].set(emb_row)
            _, slot_logits = nnx.merge(graph, st)(tokens)
            return jnp.sum(slot_logits[:, of_slot])

        row = nnx.state(model)["slot_index"]["embedding"][...][wrt_row]
        return float(jnp.abs(jax.grad(f)(row)).max())

    assert slot_grad(serial=True, of_slot=1, wrt_row=0) > 0.0, \
        "serial: slot 1 must depend on slot 0"
    assert slot_grad(serial=False, of_slot=1, wrt_row=0) == 0.0, \
        "parallel: slots must be independent"
    # The reverse direction must NEVER flow, in either mode — an earlier slot
    # depending on a later one would be a future-slot leak (guards refactors of
    # the write loop; currently impossible by construction).
    assert slot_grad(serial=True, of_slot=0, wrt_row=1) == 0.0, \
        "serial: slot 0 must not depend on slot 1"


def test_all_arms_train_and_beat_nothing_burns():
    """Full train/eval path runs for every arm at toy size, losses finite, and
    the graded arms report per-slot accuracies (the collapse detector)."""
    for arm in ("serial", "parallel", "depthonly"):
        acc, slot_acc, params = train_one_arm(arm, K=K, m=M, dim=DIM, steps=40,
                                              batch=64, n_pool=512, n_test=128, seed=0)
        assert np.isfinite(acc) and 0.0 <= acc <= 1.0, f"{arm}: bad final acc {acc}"
        assert params > 0
        if arm != "depthonly":
            assert len(slot_acc) == K and all(np.isfinite(a) for a in slot_acc)
