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

from scratchpad_harness import (BudgetScratchpadNet, ScratchpadNet, affine_chain_task,
                                arm_losses, encode_links, final_target, grade_lambda,
                                n_params, parse_arm_spec, train_one_arm)

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


def test_budget_one_addr_is_forced_to_one():
    """#63: softmax over a size-1 axis is identically 1 regardless of the
    learned addr_head's weights -- num_slots=1 has literally no addressing
    degree of freedom, so every write is a full overwrite by construction,
    not by anything the optimizer had to discover."""
    model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=1,
                                max_seq_len=2 * K, rngs=nnx.Rngs(0))
    addr_in = jax.random.normal(jax.random.PRNGKey(1), (5, 2 * DIM)) * 50.0  # extreme logits
    addr = jax.nn.softmax(model.addr_head(addr_in), axis=-1)
    np.testing.assert_allclose(np.asarray(addr), np.ones((5, 1)), atol=1e-6)


def test_budget_two_addr_is_a_real_distribution_over_slots():
    """num_slots=2: the address is a genuine 2-way softmax (sums to 1, neither
    entry pinned) -- the addressing wiring that phase 2 needs a policy to
    emerge over, as opposed to the num_slots=1 degenerate case above."""
    model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=2,
                                max_seq_len=2 * K, rngs=nnx.Rngs(0))
    addr_in = jax.random.normal(jax.random.PRNGKey(2), (5, 2 * DIM))
    addr = jax.nn.softmax(model.addr_head(addr_in), axis=-1)
    np.testing.assert_allclose(np.asarray(addr.sum(axis=-1)), np.ones(5), atol=1e-6)
    assert np.asarray(addr).std() > 0.0, "addressing must vary with input, not collapse to a constant"


def test_budget_scratchpad_grade_is_live_on_write_block():
    """The per-step grade on v_k must still reach write_block -- the same
    non-bypassable-supervision property #38's finalonly test guards for
    ScratchpadNet, checked here for the new memory-addressed architecture."""
    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(3), 8)
    model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=2,
                                max_seq_len=2 * K, rngs=nnx.Rngs(0))

    def slot_ce(mdl):
        _, slot_logits = mdl(tokens)
        return optax.softmax_cross_entropy_with_integer_labels(slot_logits, subs).mean()

    grads = nnx.to_flat_state(nnx.grad(slot_ce)(model))
    assert any(path[0] == "write_block" and float(jnp.abs(leaf[...]).max()) > 0.0
               for path, leaf in grads), "slot grade must teach the write block"


def test_budget_recall_readout_cannot_see_token_states():
    """The recall-task arms (#63 phase 2) force read_tokens=False so a correct
    answer can only come from memory, not from re-deriving r_1 out of the
    (invertible, since m is prime) affine chain directly from tokens -- the
    same #62 guard, applied to BudgetScratchpadNet's readout."""
    tokens, _ = affine_chain_task(K, M)(jax.random.PRNGKey(4), 4)

    def token_grad_into_answer(read_tokens):
        model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=2,
                                    max_seq_len=2 * K, read_tokens=read_tokens,
                                    rngs=nnx.Rngs(0))
        h = model.embed(tokens)
        for blk in model.encoder:
            h = blk(h)
        bsz = tokens.shape[0]
        memory = jnp.broadcast_to(model.mem_init[...].astype(h.dtype),
                                  (bsz, model.num_slots, h.shape[-1]))
        for k in range(K):
            q_k = jnp.broadcast_to(model.step_index(jnp.array([k]))[None, :, :],
                                   (bsz, 1, h.shape[-1]))
            v_k = model.write_block(q_k, jnp.concatenate([h, memory], axis=1))
            addr_in = jnp.concatenate([q_k[:, 0], memory.mean(axis=1)], axis=-1)
            addr = jax.nn.softmax(model.addr_head(addr_in), axis=-1)
            memory = (1 - addr[:, :, None]) * memory + addr[:, :, None] * v_k
        g = jax.grad(lambda hh: jnp.sum(model.readout(hh, memory)))(h)
        return float(jnp.abs(g).max())

    assert token_grad_into_answer(read_tokens=False) == 0.0, \
        "recall arms: the readout must be blind to token states"
    assert token_grad_into_answer(read_tokens=True) > 0.0, \
        "sanity: a tokens+memory readout must actually read the tokens"


def test_recall_arms_scored_against_their_trained_target():
    """#63 eval-target regression guard. The original phase-2 run trained the
    recall arms on (r_1+r_K) mod m but scored them against r_K — under that
    bug a SOLVED recall task also reads as chance, so the run's table was
    void. Pin both halves: final_target selects the trained answer per arm,
    and arm_losses gives ~zero final CE to an oracle that outputs exactly the
    recall target — while the same outputs scored as a plain-chain arm must
    not look solved."""
    tokens, subs = affine_chain_task(K, M)(jax.random.PRNGKey(5), 64)
    recall = (subs[:, 0] + subs[:, -1]) % M
    np.testing.assert_array_equal(np.asarray(final_target("serial", subs, M)),
                                  np.asarray(subs[:, -1]))
    np.testing.assert_array_equal(np.asarray(final_target("budget2", subs, M)),
                                  np.asarray(recall))

    class Oracle:
        def __call__(self, tok):
            return 20.0 * jax.nn.one_hot(recall, M), 20.0 * jax.nn.one_hot(subs, M)

    loss_recall, _ = arm_losses("budget2", Oracle(), tokens, subs, 1.0, K)
    loss_chain, _ = arm_losses("serial", Oracle(), tokens, subs, 1.0, K)
    assert float(loss_recall) < 0.01, "oracle recall outputs must score as solved"
    assert float(loss_chain) > 0.5, "the same outputs must NOT solve the plain chain task"


def test_budget_arms_train_and_beat_nothing_burns():
    """#63/#116: full train/eval path for the budget and local-writer arms."""
    for arm in ("overwrite", "budget1", "budget2", "unlimited",
                "budget1_local", "budget2_local", "unlimited_local"):
        r = train_one_arm(arm, K=K, m=M, dim=DIM, steps=40,
                          batch=64, n_pool=512, n_test=128, seed=0)
        acc = r["final_acc"]
        assert np.isfinite(acc) and 0.0 <= acc <= 1.0, f"{arm}: bad final acc {acc}"
        assert r["params"] > 0
        for accs in (r["slot_acc"], r["cut_slot_acc"]):
            assert len(accs) == K and all(np.isfinite(a) for a in accs)


def test_local_writer_encoding_isolates_links():
    """#116 leak guard, half 1: with per-link encoding, link k's encoded states
    must be bit-identical no matter what the OTHER links contain — there is no
    attention path between links. (Half 2 — that write k consumes only link
    k's states — is a one-line concat in the model, guarded by the gradient
    test below.) The #114 leak was exactly this: a full-sequence causal
    encoder let position 2K-1 carry link 1, so 'local context' by position
    masking alone would still smuggle."""
    task = affine_chain_task(K, M)
    tok_a, _ = task(jax.random.PRNGKey(0), 16)
    tok_b, _ = task(jax.random.PRNGKey(1), 16)
    link = 1
    tok_b = tok_b.at[:, 2 * link:2 * link + 2].set(tok_a[:, 2 * link:2 * link + 2])

    model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=1,
                                max_seq_len=2 * K, read_tokens=False,
                                local_writer=True, rngs=nnx.Rngs(0))
    h_a = encode_links(model.embed, model.encoder, tok_a, per_link=True)
    h_b = encode_links(model.embed, model.encoder, tok_b, per_link=True)
    np.testing.assert_array_equal(np.asarray(h_a[:, link]), np.asarray(h_b[:, link]))
    assert not np.array_equal(np.asarray(h_a), np.asarray(h_b)), \
        "sanity: the other links do differ"


def test_local_writer_first_link_reaches_answer_only_through_memory():
    """#116 leak guard, half 2: the gradient of the final answer w.r.t. link
    1's ENCODED states must flow only through the memory chain. Cut the chain
    — stop-gradient the memory after write 1 — and link 1's gradient must be
    exactly zero; leave it intact and it must be nonzero. Under the #114
    full-sequence writer this gradient survives the cut (the last write reads
    link 1 directly), which is the leak."""
    model = BudgetScratchpadNet(dim=DIM, vocab=M, num_steps=K, num_slots=1,
                                max_seq_len=2 * K, read_tokens=False,
                                local_writer=True, rngs=nnx.Rngs(0))
    tokens, _ = affine_chain_task(K, M)(jax.random.PRNGKey(2), 8)

    def answer_sum(h, cut_after_first):
        bsz = tokens.shape[0]
        memory = jnp.broadcast_to(model.mem_init[...].astype(h.dtype),
                                  (bsz, model.num_slots, h.shape[-1]))
        for k in range(K):
            q_k = jnp.broadcast_to(model.step_index(jnp.array([k]))[None, :, :],
                                   (bsz, 1, h.shape[-1]))
            v_k = model.write_block(q_k, jnp.concatenate([h[:, k], memory], axis=1))
            addr_in = jnp.concatenate([q_k[:, 0], memory.mean(axis=1)], axis=-1)
            addr = jax.nn.softmax(model.addr_head(addr_in), axis=-1)
            memory = (1 - addr[:, :, None]) * memory + addr[:, :, None] * v_k
            if k == 0 and cut_after_first:
                memory = jax.lax.stop_gradient(memory)
        return jnp.sum(model.readout(h[:, -1], memory))

    h = encode_links(model.embed, model.encoder, tokens, per_link=True)
    g_cut = jax.grad(answer_sum)(h, cut_after_first=True)
    g_open = jax.grad(answer_sum)(h, cut_after_first=False)
    assert float(jnp.abs(g_cut[:, 0]).max()) == 0.0, \
        "link 1 must be unreachable except through memory"
    assert float(jnp.abs(g_open[:, 0]).max()) > 0.0, \
        "sanity: through memory, link 1 does reach the answer"


def test_local_arms_share_target_with_their_full_context_twins():
    """#116: the _local arms train and score on the same recall target as
    their #63 twins — the writer context is the ONLY variable."""
    _, subs = affine_chain_task(K, M)(jax.random.PRNGKey(3), 32)
    recall = np.asarray((subs[:, 0] + subs[:, -1]) % M)
    for arm in ("budget1_local", "budget2_local", "unlimited_local"):
        np.testing.assert_array_equal(np.asarray(final_target(arm, subs, M)), recall)


def test_variable_chain_task_stationary_after_k_eff():
    """#123: pad links are (0,0), sub-targets freeze at r_{k_eff} through the
    pads, and the final sub IS r_{k_eff} — 'done' must be detectable as the
    state no longer changing, or halting has nothing to track."""
    from scratchpad_harness import variable_chain_task
    tokens, subs, k_eff = variable_chain_task(6, M)(jax.random.PRNGKey(7), 256)
    a = np.asarray(tokens)[:, 0::2]
    subs, k_eff = np.asarray(subs), np.asarray(k_eff)
    for i in range(256):
        assert (a[i, :k_eff[i]] >= 1).all(), "real links must have a >= 1"
        assert (a[i, k_eff[i]:] == 0).all(), "pad links must have a == 0"
        assert (subs[i, k_eff[i] - 1:] == subs[i, k_eff[i] - 1]).all(), \
            "sub-targets must be stationary after the chain ends"
    assert (subs[:, -1] == subs[np.arange(256), k_eff - 1]).all()
    assert set(np.unique(k_eff)) == set(range(1, 7)), "k_eff should span 1..K"


def test_halt_arms_share_param_tree_and_smoke_train():
    """#123: trajectory vs current halt context must be the SAME parameters
    (one-variable ablation — only the halt head's context width differs), and
    all three arms must train end-to-end at toy size with finite metrics."""
    from scratchpad_harness import HaltingScratchpadNet, train_one_halt_arm

    def shapes(net):
        return jax.tree_util.tree_map(lambda x: x.shape, nnx.state(net, nnx.Param))

    traj = HaltingScratchpadNet(dim=DIM, vocab=M, num_slots=4,
                                halt_context="trajectory", rngs=nnx.Rngs(0))
    curr = HaltingScratchpadNet(dim=DIM, vocab=M, num_slots=4,
                                halt_context="current", rngs=nnx.Rngs(0))
    assert shapes(traj) == shapes(curr), "halt arms must share one param tree"

    for arm in ("halt_traj", "halt_state", "halt_off"):
        r = train_one_halt_arm(arm, K=4, m=M, dim=DIM, steps=40,
                               batch=64, n_pool=512, n_test=128, seed=0)
        for key in ("full_acc", "halted_acc", "corr", "p1_mass", "mean_halt"):
            assert np.isfinite(r[key]), f"{arm}: non-finite {key}"
        assert 0.0 <= r["full_acc"] <= 1.0 and 0.0 <= r["halted_acc"] <= 1.0
        assert 1.0 <= r["mean_halt"] <= 4.0, f"{arm}: halt step outside 1..K"

def test_grade_halt_steps_first_crossing_on_hand_built_signal():
    """#39: the detector must halt at the FIRST slot whose grade-logit cosine
    against the previous slot clears tau, and run all K slots when none does.
    Hand-built one-hot grades make every cosine exactly 0 or 1, so the
    expected halt step is known by construction — no model in the loop."""
    from scratchpad_harness import grade_halt_steps
    e = np.eye(M, dtype=np.float32)
    slot_logits = jnp.asarray(np.stack([
        [e[0], e[0], e[1], e[2]],   # cos [1,0,0] -> halt step 1 (2 writes)
        [e[0], e[1], e[1], e[2]],   # cos [0,1,0] -> halt step 2 (3 writes)
        [e[0], e[1], e[2], e[2]],   # cos [0,0,1] -> halt step 3 (4 writes)
        [e[0], e[1], e[2], e[3]],   # cos [0,0,0] -> no crossing, full depth
    ]))
    np.testing.assert_array_equal(
        np.asarray(grade_halt_steps(slot_logits, 0.5)), [1, 2, 3, 3])

    # tau-monotone: a mid-angle transition (cos ~= 0.707) crosses tau = 0.5
    # but not tau = 0.9, and the halt can only move later as tau rises.
    mid = ((e[0] + e[1]) / np.sqrt(2)).astype(np.float32)
    graded = jnp.asarray(np.stack([[e[0], mid, e[2], e[2]]]))  # cos [.707, 0, 1]
    assert int(grade_halt_steps(graded, 0.5)[0]) == 1
    assert int(grade_halt_steps(graded, 0.9)[0]) == 3


def test_grade_halting_ladder_readout_composition():
    """#39: the halt-step -> answer-column composition. #96's ladder had
    off-by-one-prone index math and NO test on it — an off-by-one there shifts
    every halted accuracy while leaving mean writes untouched, i.e. a
    plausible-looking wrong verdict. Answers are built so the target is
    argmax ONLY at the expected halt column; any index slip reads a wrong
    class and accuracy craters from 1.0 to 0.0."""
    from scratchpad_harness import grade_halting_ladder
    e = np.eye(M, dtype=np.float32)
    slot_logits = np.stack([
        [e[0], e[0], e[1], e[2]],   # halt step 1
        [e[0], e[1], e[1], e[2]],   # halt step 2
        [e[0], e[1], e[2], e[2]],   # halt step 3
        [e[0], e[1], e[2], e[3]],   # full depth: step 3
    ])
    expected_step = np.array([1, 2, 3, 3])
    target = np.array([0, 1, 2, 3])
    answers = np.zeros((4, 4, M), np.float32)
    for b in range(4):
        for k in range(4):
            answers[b, k, target[b] if k == expected_step[b] else (target[b] + 1) % M] = 1.0
    te_sub = np.zeros((4, 4), np.int64)
    te_sub[:, -1] = target
    def stub(tok):
        return jnp.asarray(answers), jnp.asarray(slot_logits), None
    [row] = grade_halting_ladder(stub, jnp.zeros((4, 8), jnp.int32),
                                 jnp.asarray(te_sub), jnp.asarray(expected_step + 1),
                                 taus=(0.5,))
    assert row["halted_acc"] == 1.0, "halt-index arithmetic is off"
    assert row["mean_writes"] == pytest.approx((2 + 3 + 4 + 4) / 4)
    assert np.isfinite(row["corr"])


def test_converged_transition_labels_are_exact():
    """#39: convergence labels come from k_eff, never from repeated residues
    (#96 mislabelled 11.4% of steps that way). Transition j feeds slot
    k = j+2 (1-indexed); it is converged iff k > k_eff. Cross-check on real
    task draws: a converged transition's sub-target must not have moved."""
    from scratchpad_harness import converged_transition_labels, variable_chain_task
    labels = np.asarray(converged_transition_labels(jnp.asarray([1, 2, 3, 4]), 4))
    np.testing.assert_array_equal(labels, [[True, True, True],
                                           [False, True, True],
                                           [False, False, True],
                                           [False, False, False]])

    _, subs, k_eff = variable_chain_task(4, M)(jax.random.PRNGKey(11), 512)
    subs = np.asarray(subs)
    lab = np.asarray(converged_transition_labels(k_eff, 4))
    for j in range(3):
        conv = lab[:, j]
        assert (subs[conv, j + 1] == subs[conv, j]).all(), \
            "a converged transition changed the sub-target"


def test_grade_gate_stats_separation_call():
    """#39: the gate must fire 'separated' only when the converged and
    computing cosine means sit more than one pooled sigma apart."""
    from scratchpad_harness import grade_gate_stats
    e = np.eye(M, dtype=np.float32)
    apart = jnp.asarray(np.stack([
        [e[0], e[0], e[0], e[0]],   # k_eff=1: all transitions converged, cos 1
        [e[0], e[1], e[2], e[3]],   # k_eff=4: all computing, cos 0
    ]))
    g = grade_gate_stats(apart, jnp.asarray([1, 4]))
    assert g["separated"] and g["converged_mean"] > 0.9 > 0.1 > g["computing_mean"]

    overlap = jnp.asarray(np.stack([
        [e[0], e[1], e[2], e[3]],
        [e[0], e[1], e[2], e[3]],
    ]))
    g = grade_gate_stats(overlap, jnp.asarray([1, 4]))
    assert not g["separated"], "identical distributions must not pass the gate"
