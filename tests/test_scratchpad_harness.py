"""Wiring guards for the serial-scratchpad proof harness (#38).

These pin the properties the ablation's interpretation rests on: the task's
recurrence is what the design doc says, serial and parallel arms differ by
exactly one variable (data flow, not parameters), and the slot grades are a
live gradient path — not a bonus the optimizer can ignore."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def test_slotsonly_readout_cannot_see_token_states():
    """#62 wiring guard, same style as the order-flow guard: the slots-only
    readout must have exactly zero gradient from the token states it was
    blinded to — otherwise the ablation isn't measuring what it claims —
    while the tokens+slots readout must have a live token path."""
    tokens, _ = affine_chain_task(K, M)(jax.random.PRNGKey(2), 4)

    def token_grad_into_answer(read_tokens):
        model = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                              serial=True, read_tokens=read_tokens, rngs=nnx.Rngs(0))
        h = model.embed(tokens)
        for blk in model.encoder:
            h = blk(h)
        queries = model.slot_index(jnp.arange(K))[None].repeat(4, axis=0)
        slots = []
        for k in range(K):
            ctx = jnp.concatenate([h] + slots, axis=1) if slots else h
            slots.append(model.write_block(queries[:, k:k + 1], ctx))
        slots = jnp.concatenate(slots, axis=1)
        # Differentiate the answer w.r.t. the token states AT THE READOUT ONLY
        # (slots held fixed): any nonzero grad is a direct token->answer path.
        g = jax.grad(lambda hh: jnp.sum(model.readout(hh, slots)))(h)
        return float(jnp.abs(g).max())

    assert token_grad_into_answer(read_tokens=False) == 0.0, \
        "slotsonly: the readout must be blind to token states"
    assert token_grad_into_answer(read_tokens=True) > 0.0, \
        "serial: the tokens+slots readout must actually read the tokens"


def test_slotsonly_shares_the_serial_param_tree():
    serial = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                           serial=True, read_tokens=True, rngs=nnx.Rngs(0))
    blind = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                          serial=True, read_tokens=False, rngs=nnx.Rngs(0))
    for sl, bl in zip(jax.tree_util.tree_leaves(nnx.state(serial, nnx.Param)),
                      jax.tree_util.tree_leaves(nnx.state(blind, nnx.Param))):
        np.testing.assert_array_equal(np.asarray(sl), np.asarray(bl))


def test_finalonly_grade_is_a_detached_probe():
    """finalonly (#67): the slot CE must reach ONLY the readout head — every
    parameter upstream of the slots (write block, encoder, slot indices) gets
    zero gradient from the grade, so final-answer CE is the model's sole
    teacher. The graded serial arm must keep the grade live, unchanged."""
    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(2), 8)

    def slot_ce_grads(probe_only):
        model = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                              serial=True, probe_only=probe_only, rngs=nnx.Rngs(0))

        def slot_ce(mdl):
            _, slot_logits = mdl(tokens)
            return optax.softmax_cross_entropy_with_integer_labels(slot_logits, subs).mean()

        return nnx.to_flat_state(nnx.grad(slot_ce)(model))

    for path, leaf in slot_ce_grads(probe_only=True):
        mx = float(jnp.abs(leaf[...]).max())
        if path[0] == "slot_readout":
            assert mx > 0.0, "probe head itself must still train"
        else:
            assert mx == 0.0, f"grade leaked upstream into {'/'.join(map(str, path))}"
    assert any(path[0] == "write_block" and float(jnp.abs(leaf[...]).max()) > 0.0
               for path, leaf in slot_ce_grads(probe_only=False)), \
        "graded arm: slot CE must still teach the write block"


def test_all_arms_train_and_beat_nothing_burns():
    """Full train/eval path runs for every arm at toy size, losses finite, and
    the graded arms report per-slot accuracies (the collapse detector)."""
    for arm in ("serial", "parallel", "depthonly", "slotsonly", "finalonly"):
        acc, slot_acc, params = train_one_arm(arm, K=K, m=M, dim=DIM, steps=40,
                                              batch=64, n_pool=512, n_test=128, seed=0)
        assert np.isfinite(acc) and 0.0 <= acc <= 1.0, f"{arm}: bad final acc {acc}"
        assert params > 0
        if arm != "depthonly":
            assert len(slot_acc) == K and all(np.isfinite(a) for a in slot_acc)
