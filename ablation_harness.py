"""Tiny-config ablation harness for Plan A — the proof instrument.

Trains the CausalRefiner at fixed depths on depth-sensitive toy tasks and reports
val accuracy / CE per depth. The question Plan A has to answer: does looping the
shared block more times let it solve a task a shallow model can't? Fineweb
perplexity can't show this (recurrent-depth wins live on algorithmic tasks), so we
test where depth provably has to do work — cumulative scans, which require
aggregating all prior tokens.

    venv/bin/python ablation_harness.py --task parity --depths 1,2,4,8
"""

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from plan_a_model import CausalRefiner


def cumulative_task(key, batch, seq, mod):
    """Predict the running sum mod `mod` of the inputs so far — a sequential scan.
    Position t's target needs every input <= t, so it rewards aggregation depth.
    mod=2 is parity. Input and target share the vocab {0..mod-1}."""
    x = jax.random.randint(key, (batch, seq), 0, mod)
    targets = jnp.cumsum(x, axis=1) % mod
    mask = jnp.ones((batch, seq), dtype=jnp.float32)
    return x.astype(jnp.int32), targets.astype(jnp.int32), mask


def state_tracking_task(key, batch, seq, n_states=5, n_gen=4):
    """Non-commutative state tracking: each input token picks one of `n_gen` fixed
    permutations of `n_states`; the target at position t is the state reached by
    composing those permutations over the prefix, starting from state 0. Because
    composition is non-abelian, there is no sum/count shortcut — getting position t
    right requires sequentially tracking the state, so it genuinely rewards depth
    (the Liu et al. "Transformers Learn Shortcuts to Automata" regime)."""
    gen_key = jax.random.PRNGKey(12345)  # fixed generators across all calls
    perms = jnp.stack([
        jax.random.permutation(jax.random.fold_in(gen_key, i), n_states) for i in range(n_gen)
    ])  # [n_gen, n_states]

    toks = jax.random.randint(key, (batch, seq), 0, n_gen)

    def track(tok_row):
        def step(state, t):
            new = perms[t][state]
            return new, new
        _, states = jax.lax.scan(step, jnp.int32(0), tok_row)
        return states

    targets = jax.vmap(track)(toks).astype(jnp.int32)
    mask = jnp.ones((batch, seq), dtype=jnp.float32)
    return toks.astype(jnp.int32), targets, mask


SEQ = 24
TASKS = {
    "parity": lambda key, batch: cumulative_task(key, batch, SEQ, 2),
    "cumsum5": lambda key, batch: cumulative_task(key, batch, SEQ, 5),
    "statetrack": lambda key, batch: state_tracking_task(key, batch, SEQ, 5, 4),
}
# Vocab must cover both input tokens and targets. statetrack: inputs in [0,n_gen),
# targets (states) in [0,n_states) -> vocab = max(n_gen, n_states).
VOCAB = {"parity": 2, "cumsum5": 5, "statetrack": 5}


def _masked_acc_ce(logits, tgt, mask):
    pred = logits.argmax(-1)
    acc = jnp.sum((pred == tgt) * mask) / jnp.sum(mask)
    ce = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt) * mask) / jnp.sum(mask)
    return acc, ce


def train_one(task_fn, vocab, depth, *, dim=96, heads=4, enc=2, steps=2500,
              batch=256, lr=2e-3, wd=0.01, seed=0, n_pool=32768, n_test=4096):
    key = jax.random.PRNGKey(seed)
    # Generate the data pool ONCE — the task's perms/scan are expensive, and doing
    # them per step starved the tiny model's GPU (host-bound at ~10% util). Train on
    # minibatches sampled on-device inside the jitted step; eval on a held-out slice
    # so the depth comparison also reflects generalization, not just fit.
    key, dk = jax.random.split(key)
    inp_all, tgt_all, mask_all = task_fn(dk, n_pool + n_test)
    tr_inp, tr_tgt, tr_mask = inp_all[:n_pool], tgt_all[:n_pool], mask_all[:n_pool]
    te_inp, te_tgt, te_mask = inp_all[n_pool:], tgt_all[n_pool:], mask_all[n_pool:]

    model = CausalRefiner(dim=dim, vocab_size=vocab, num_heads=heads,
                          num_encoder_layers=enc, max_depth=max(depth, 1),
                          max_seq_len=SEQ, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    @nnx.jit(static_argnames=["depth"])
    def step(model, opt, key, inp_all, tgt_all, mask_all, depth):
        idx = jax.random.randint(key, (batch,), 0, inp_all.shape[0])
        inp, tgt, mask = inp_all[idx], tgt_all[idx], mask_all[idx]
        def loss_fn(m):
            logits = m(inp, depth=depth)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
            return jnp.sum(ce * mask) / jnp.sum(mask)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit(static_argnames=["depth"])
    def eval_step(model, inp, tgt, mask, depth):
        return _masked_acc_ce(model(inp, depth=depth), tgt, mask)

    for _ in range(steps):
        key, k = jax.random.split(key)
        step(model, opt, k, tr_inp, tr_tgt, tr_mask, depth)

    acc, ce = eval_step(model, te_inp, te_tgt, te_mask, depth)
    return float(acc), float(ce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="parity", choices=list(TASKS))
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task_fn = TASKS[args.task]
    vocab = VOCAB[args.task]
    depths = [int(d) for d in args.depths.split(",")]

    print(f"== Plan A depth ablation: task={args.task} (seq={SEQ}, vocab={vocab}) steps={args.steps} seed={args.seed} ==")
    print(f"{'depth':>6} {'val_acc':>9} {'val_ce':>9} {'sec':>7}")
    results = {}
    for d in depths:
        t0 = time.time()
        acc, ce = train_one(task_fn, vocab, d, steps=args.steps, seed=args.seed)
        results[d] = (acc, ce)
        print(f"{d:>6} {acc:>9.4f} {ce:>9.4f} {time.time() - t0:>7.1f}")

    base_acc = results[depths[0]][0]
    best_d = max(results, key=lambda d: results[d][0])
    print(f"\ndepth {depths[0]}: acc {base_acc:.4f}  ->  best depth {best_d}: acc {results[best_d][0]:.4f}  |  gain {results[best_d][0] - base_acc:+.4f}")


if __name__ == "__main__":
    main()
