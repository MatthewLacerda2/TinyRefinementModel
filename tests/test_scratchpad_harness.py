"""Wiring guards for the serial-scratchpad proof harness (#38).

These pin the properties the ablation's interpretation rests on: the task's
recurrence is what the design doc says, serial and parallel arms differ by
exactly one variable (data flow, not parameters), and the slot grades are a
live gradient path — not a bonus the optimizer can ignore."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scratchpad_harness import (ScratchpadNet, affine_chain_task,
                                affine_chain_varlen_task, arm_losses,
                                grade_lambda, halt_indices, n_params,
                                parse_arm_spec, slot_cosines, train_one_arm)

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


def test_varlen_task_identity_tail():
    """#39 split: the trailing (a=1, b=0) pairs must leave the sub-results
    frozen (r_k = r_L past the effective length), the live prefix must follow
    the same recurrence as the base task, and every effective length 1..K must
    actually occur in a reasonable sample."""
    tokens, subs = affine_chain_varlen_task(K, M)(jax.random.PRNGKey(0), 512)
    a, b = np.asarray(tokens[:, 0::2]), np.asarray(tokens[:, 1::2])
    assert (a >= 1).all()
    r = np.zeros(512, dtype=np.int64)
    for k in range(K):
        r = (r * a[:, k] + b[:, k]) % M
        np.testing.assert_array_equal(np.asarray(subs[:, k]), r)
    # identity tail: once a step is (1, 0) every later step is too, and the
    # sub-result stops moving there
    identity = (a == 1) & (b == 0)
    # tail[k] True iff steps k..K-1 are ALL identity (the trailing identity block)
    tail = np.maximum.accumulate(~identity[:, ::-1], axis=1)[:, ::-1] == 0
    sub_np = np.asarray(subs)
    frozen = sub_np[:, 1:] == sub_np[:, :-1]
    assert frozen[tail[:, 1:]].all(), "sub-results must be frozen on the identity tail"
    lengths = K - tail.sum(axis=1)     # live prefix; coincidental (1,0) live steps can shorten it
    assert lengths.min() <= 1 and lengths.max() == K and len(np.unique(lengths)) >= K


def test_write_slots_truncation_is_early_stopping():
    """The property cosine halting stands on: because slots are write-once and
    causal, the first n slots of a full run must be bit-identical to a run
    stopped after n writes."""
    tokens, _ = affine_chain_task(K, M)(jax.random.PRNGKey(3), 8)
    model = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                          serial=True, rngs=nnx.Rngs(0))
    h = model.encode(tokens)
    full = model.write_slots(h)
    for n in range(1, K + 1):
        stopped = model.write_slots(h, upto=n)
        np.testing.assert_array_equal(np.asarray(stopped), np.asarray(full[:, :n]))


def test_halt_rule_on_handbuilt_signal():
    """halt_indices must pick the FIRST write whose cosine clears tau, else K;
    slot_cosines must be an exact cosine on hand-built vectors."""
    # cos rows: [c_2, c_3, c_4] (signal after writing slots 2, 3, 4), K=4
    cos = jnp.array([[0.99, 0.10, 0.10],    # halts at write 2
                     [0.10, 0.99, 0.995],   # halts at write 3 (first crossing)
                     [0.10, 0.20, 0.30]])   # never halts -> K
    np.testing.assert_array_equal(np.asarray(halt_indices(cos, 0.9, 4)), [2, 3, 4])
    np.testing.assert_array_equal(np.asarray(halt_indices(cos, 0.0, 4)), [2, 2, 2])
    np.testing.assert_array_equal(np.asarray(halt_indices(cos, 1.0, 4)), [4, 4, 4])
    slots = jnp.array([[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])   # parallel, then orthogonal
    np.testing.assert_allclose(np.asarray(slot_cosines(slots))[0], [1.0, 0.0], atol=1e-6)


def test_densedepth_grade_teaches_the_trunk():
    """#79 wiring guard: the intermediate per-step grades (passes k < K, graded
    at the answer position against r_k through the dedicated head) must be a
    live gradient path into the shared refine block and the encoder — dense
    supervision has to teach the trunk, or the arm would just be depthonly
    with a decorative loss term."""
    from scratchpad_harness import DenseDepthNet

    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(3), 8)
    model = DenseDepthNet(dim=DIM, vocab=M, K=K, rngs=nnx.Rngs(0))

    def intermediate_ce(mdl):
        _, step_logits = mdl(tokens)                        # [K, B, m]
        return optax.softmax_cross_entropy_with_integer_labels(
            step_logits[:K - 1], subs.T[:K - 1]).mean()     # intermediate passes only

    grads = nnx.to_flat_state(nnx.grad(intermediate_ce)(model))
    live = {path[:2] for path, leaf in grads if float(jnp.abs(leaf[...]).max()) > 0.0}
    assert ("refiner", "refine_block") in live, "per-step grade must teach the shared refine block"
    assert ("refiner", "encoder") in live, "per-step grade must reach the encoder"
    assert any(p[0] == "step_readout" for p in live), "the grade head itself must train"


def test_grade_lambda_matches_the_preregistered_schedule():
    """#73 pre-registration at 2500 steps: grade fully on for 0–1000, linear
    decay 1000–1500, exactly zero for 1500–2500. The schedule is written in
    fractions of the budget so the toy-sized test runs exercise it too."""
    S = 2500
    assert grade_lambda(0, S) == 1.0
    assert grade_lambda(999, S) == 1.0
    assert grade_lambda(1000, S) == 1.0   # the on/decay boundary itself
    assert grade_lambda(1250, S) == 0.5
    assert grade_lambda(1500, S) == 0.0
    assert grade_lambda(2499, S) == 0.0
    lams = [grade_lambda(i, S) for i in range(S)]
    assert all(b <= a for a, b in zip(lams, lams[1:])), "λ must never rise"


def test_grade_lambda_onset_and_floor_parameterization():
    """#95 sweeps the schedule: onset moves the decay start (window stays 20%
    of the budget), floor makes the decay land on a residual grade instead of
    zero and hold it there."""
    S = 2500
    # onset 0.2: on 0–500, decay 500–1000, off 1000+
    assert grade_lambda(499, S, onset=0.2) == 1.0
    assert grade_lambda(750, S, onset=0.2) == 0.5
    assert grade_lambda(1000, S, onset=0.2) == 0.0
    # floor 0.1: decays 1.0 → 0.1 across #73's window, then holds
    assert grade_lambda(999, S, floor=0.1) == 1.0
    assert grade_lambda(1250, S, floor=0.1) == 0.55
    assert grade_lambda(1500, S, floor=0.1) == 0.1
    assert grade_lambda(2499, S, floor=0.1) == 0.1


def test_parse_arm_spec_round_trips():
    assert parse_arm_spec("serial") == ("serial", {})
    assert parse_arm_spec("annealed") == ("annealed", {})
    assert parse_arm_spec("annealed@0.2") == ("annealed", {"anneal_onset": 0.2})
    assert parse_arm_spec("annealed@0.4f0.1") == \
        ("annealed", {"anneal_onset": 0.4, "anneal_floor": 0.1})
    assert parse_arm_spec("annealedf0.1") == ("annealed", {"anneal_floor": 0.1})


def test_near_miss_arm_specs_refuse_to_run():
    """A typo'd anneal spec must raise, never silently train some other arm
    under an annealed-looking label — that would fabricate a sweep row."""
    for bad in ("annealed@", "annealed@0.2f", "annealed@1.0", "annealed@abc"):
        with pytest.raises(ValueError):
            parse_arm_spec(bad)
    with pytest.raises(AssertionError):
        train_one_arm("annealed@0.2", K=K, m=M, dim=DIM, steps=2,
                      batch=8, n_pool=16, n_test=8, seed=0)


def test_annealed_lambda_zero_kills_the_grade_gradient():
    """annealed (#73): once λ_slot hits zero the slot grade must teach nothing
    — the grade head gets exactly zero gradient (its only loss path is the
    slot CE) — while final-answer CE keeps teaching the write block. That is
    what makes the post-anneal stretch a true final-only regime (#67's regime,
    warm-started from a formed chain)."""
    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(3), 8)
    model = ScratchpadNet(dim=DIM, vocab=M, num_slots=K, max_seq_len=2 * K,
                          serial=True, rngs=nnx.Rngs(0))

    def total_loss(mdl, lam):
        loss, _ = arm_losses("annealed", mdl, tokens, subs, lam, K)
        return loss

    grads = nnx.to_flat_state(nnx.grad(lambda mm: total_loss(mm, 0.0))(model))
    by_top = {}
    for path, leaf in grads:
        mx = float(jnp.abs(leaf[...]).max())
        by_top[path[0]] = max(by_top.get(path[0], 0.0), mx)
    assert by_top["slot_readout"] == 0.0, "λ=0: the grade head must get zero gradient"
    assert by_top["write_block"] > 0.0, "λ=0: final CE must still teach the write block"


def test_all_arms_train_and_beat_nothing_burns():
    """Full train/eval path runs for every arm at toy size, losses finite, and
    the graded arms report per-slot accuracies (the collapse detector)."""
    for arm in ("serial", "parallel", "depthonly", "slotsonly", "finalonly", "annealed",
                "densedepth", "densedepth_tied"):
        r = train_one_arm(arm, K=K, m=M, dim=DIM, steps=40,
                          batch=64, n_pool=512, n_test=128, seed=0)
        acc = r["final_acc"]
        assert np.isfinite(acc) and 0.0 <= acc <= 1.0, f"{arm}: bad final acc {acc}"
        assert np.isfinite(r["cut_final_acc"]), f"{arm}: missing grade-off checkpoint"
        assert r["params"] > 0
        if arm != "depthonly":
            for accs in (r["slot_acc"], r["cut_slot_acc"]):
                assert len(accs) == K and all(np.isfinite(a) for a in accs)
