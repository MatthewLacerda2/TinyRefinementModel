"""Tiny-config ablation harness for Plan A — the proof instrument.

Trains the CausalRefiner at fixed depths on depth-sensitive toy tasks and reports
val accuracy / CE per depth. The question Plan A has to answer: does looping the
shared block more times let it solve a task a shallow model can't? Fineweb
perplexity can't show this (recurrent-depth wins live on algorithmic tasks), so we
test where depth provably has to do work — cumulative scans, which require
aggregating all prior tokens.

Two arms (#34), selected with --arch:
  refiner  — CausalRefiner: ONE shared block looped `depth` times (the live bet).
  vanilla  — VanillaTransformer: `depth` DISTINCT blocks, no weight sharing, no
             time embedding. Approximately matched per-token FLOPs at equal depth
             (the refiner spends a little extra on its update gate and time
             embedding), ~depth× the block parameters — the arm that can afford
             to memorize.

Two probe families:
  generalization (parity / cumsum5 / statetrack) — a rule exists; held-out
             accuracy measures whether the model learned the procedure.
  memorization (memorize, --mem-pairs N) — NO rule exists (fixed random
             key→value dictionary), so the only way to be right is storage.
             Recall over the trained pairs measures capacity; sweep N to read it.

    venv/bin/python ablation_harness.py --task parity --depths 1,2,4,8
    venv/bin/python ablation_harness.py --task memorize --mem-pairs 4096 --arch vanilla
"""

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from plan_a_model import Block, CausalRefiner


class VanillaTransformer(nnx.Module):
    """Matched-compute control arm: same embed / encoder / tied head as the
    CausalRefiner, but the refinement loop is `num_blocks` DISTINCT Blocks — no
    weight sharing, no time embedding, no update gate. At num_blocks == depth the
    per-token FLOPs approximately match the refiner (which additionally spends
    gate + time-embedding FLOPs per iteration); parameter count is ~depth× larger."""

    def __init__(self, *, dim, vocab_size, num_heads=4, num_encoder_layers=2,
                 num_blocks=8, max_seq_len=512, rngs, dtype=jnp.float32):
        self.dtype = dtype
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs, dtype=dtype)
        self.encoder = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype) for _ in range(num_encoder_layers)
        ])
        self.blocks = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype) for _ in range(num_blocks)
        ])
        self.out_norm = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

    def __call__(self, tokens, depth=None, pad_mask=None):
        # `depth` is accepted for interface parity with CausalRefiner and ignored:
        # this arm's depth is its (fixed) number of distinct blocks.
        pad_bias = None
        if pad_mask is not None:
            pad_bias = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
            pad_bias = pad_bias[:, None, None, :]

        z = self.embed(tokens)
        for blk in self.encoder:
            z = blk(z, pad_bias)
        for blk in self.blocks:
            z = blk(z, pad_bias)

        z = self.out_norm(z)
        embed_t = self.embed.embedding[...].astype(self.dtype).T
        return jnp.matmul(z.astype(self.dtype), embed_t, preferred_element_type=jnp.float32)


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


def memorize_task(n_pairs):
    """Pure-memorization probe: a FIXED dictionary of n_pairs random key→value pairs
    (values drawn once from a constant seed, same dictionary every call). No rule
    exists — the values are random — so the only way to be right is to store the
    pairs. The target at position t is dict[input[t]] (per-position lookup, causal-
    safe). Recall over the trained dictionary is the capacity metric; sweep n_pairs
    to find where it saturates."""
    values = jax.random.randint(jax.random.PRNGKey(424242), (n_pairs,), 0, n_pairs)

    def fn(key, batch, seq):
        x = jax.random.randint(key, (batch, seq), 0, n_pairs)
        mask = jnp.ones((batch, seq), dtype=jnp.float32)
        return x.astype(jnp.int32), values[x].astype(jnp.int32), mask
    return fn


SEQ = 24  # default train length
TASKS = {
    "parity": lambda key, batch, seq: cumulative_task(key, batch, seq, 2),
    "cumsum5": lambda key, batch, seq: cumulative_task(key, batch, seq, 5),
    "statetrack": lambda key, batch, seq: state_tracking_task(key, batch, seq, 5, 4),
}
# Vocab must cover both input tokens and targets. statetrack: inputs in [0,n_gen),
# targets (states) in [0,n_states) -> vocab = max(n_gen, n_states).
VOCAB = {"parity": 2, "cumsum5": 5, "statetrack": 5}
# `memorize` is built per-run (vocab = --mem-pairs), see main().


def train_one(task_fn, vocab, depth, *, arch="refiner", dim=96, heads=4, enc=2, steps=2500,
              batch=256, lr=2e-3, wd=0.01, seed=0, gate_bias=0.0, grad_last=None,
              per_pass_loss=False, islands=False, readouts=False,
              n_pool=32768, n_test=4096, eval_batch=256, train_seq=SEQ, test_seq=None):
    # When test_seq > train_seq this is a length-generalization probe: the model
    # trains on length train_seq and is evaluated on longer sequences, so it must
    # use RoPE positions it never saw in training. test_seq=None -> same length.
    test_seq = train_seq if test_seq is None else test_seq
    key = jax.random.PRNGKey(seed)
    # Generate each pool ONCE — the task's perms/scan are expensive, and doing them
    # per step starved the tiny model's GPU (host-bound at ~10% util). Train on
    # minibatches sampled on-device inside the jitted step; eval on a separately
    # drawn held-out pool so the comparison reflects generalization, not just fit.
    # (For `memorize` the dictionary is fixed, so the "held-out" pool is new
    # sequences over the SAME trained pairs — i.e. recall, which is the point.)
    key, dk_tr, dk_te = jax.random.split(key, 3)
    tr_inp, tr_tgt, tr_mask = task_fn(dk_tr, n_pool, train_seq)
    te_inp, te_tgt, te_mask = task_fn(dk_te, n_test, test_seq)

    if arch == "vanilla":
        model = VanillaTransformer(dim=dim, vocab_size=vocab, num_heads=heads,
                                   num_encoder_layers=enc, num_blocks=max(depth, 1),
                                   max_seq_len=max(train_seq, test_seq), rngs=nnx.Rngs(seed))
    else:
        model = CausalRefiner(dim=dim, vocab_size=vocab, num_heads=heads,
                              num_encoder_layers=enc, max_depth=max(depth, 1),
                              max_seq_len=max(train_seq, test_seq), gate_bias=gate_bias, rngs=nnx.Rngs(seed))
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    # Truncated backprop (#64) / gradient islands + per-pass supervision (#75):
    # refiner-arm knobs only — vanilla has no recurrence to cut or grade.
    # Training-only; eval has no backward pass, so its final-pass metric is the
    # same instrument for every arm. All are closed over, so static at trace time.
    model_kwargs = {} if arch == "vanilla" else {"grad_last": grad_last, "islands": islands}

    @nnx.jit(static_argnames=["depth"])
    def step(model, opt, key, inp_all, tgt_all, mask_all, depth):
        idx = jax.random.randint(key, (batch,), 0, inp_all.shape[0])
        inp, tgt, mask = inp_all[idx], tgt_all[idx], mask_all[idx]
        def loss_fn(m):
            if per_pass_loss:
                # Deep supervision (#75): grade EVERY pass's draft against the
                # target, uniform mean over passes. Broadcast targets/mask over
                # the leading pass axis of [depth, b, s, vocab] logits.
                logits_all, _ = m(inp, depth=depth, return_all_iters=True, **model_kwargs)
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits_all, labels=jnp.broadcast_to(tgt, (depth,) + tgt.shape))
                return jnp.sum(ce * mask) / (jnp.sum(mask) * depth)
            logits = m(inp, depth=depth, **model_kwargs)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
            return jnp.sum(ce * mask) / jnp.sum(mask)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit(static_argnames=["depth"])
    def eval_chunk_sums(model, inp, tgt, mask, depth):
        logits = model(inp, depth=depth)
        pred = logits.argmax(-1)
        acc_sum = jnp.sum((pred == tgt) * mask)
        ce_sum = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt) * mask)
        return acc_sum, ce_sum, jnp.sum(mask)

    for _ in range(steps):
        key, k = jax.random.split(key)
        step(model, opt, k, tr_inp, tr_tgt, tr_mask, depth)

    # Eval in fixed-size chunks: a full-pool eval materializes [n_test, seq, vocab]
    # f32 logits, which OOMs at large --mem-pairs (6.4 GB at N=16384 — already past
    # the 6 GB card; 51 GB at N=65536). Keep n_test a multiple of eval_batch or
    # the ragged last chunk pays one extra compile.
    acc_sum = ce_sum = count = 0.0
    for i in range(0, te_inp.shape[0], eval_batch):
        s = slice(i, i + eval_batch)
        a, c, m = eval_chunk_sums(model, te_inp[s], te_tgt[s], te_mask[s], depth)
        acc_sum, ce_sum, count = acc_sum + float(a), ce_sum + float(c), count + float(m)

    if not readouts:
        return acc_sum / count, ce_sum / count, n_params

    # #75 readouts (refiner only) — how each part of the model reacts:
    #   per-pass accuracy: is every pass improving the draft, or does the work
    #     happen late? gate openness: do early passes make real updates?
    #   depth transfer: trained at `depth`, evaled at 1..12 — past max_depth the
    #     time embedding saturates (jnp.take clips the row index), so passes
    #     beyond training depth reuse the last time signal.
    @nnx.jit(static_argnames=["depth"])
    def eval_passes(model, inp, tgt, mask, depth):
        logits_all, gates = model(inp, depth=depth, return_all_iters=True)
        hit = (logits_all.argmax(-1) == tgt[None]) * mask[None]
        return jnp.sum(hit, axis=(1, 2)) / jnp.sum(mask), gates

    sub = slice(0, min(1024, te_inp.shape[0]))
    pass_acc, gate_open = eval_passes(model, te_inp[sub], te_tgt[sub], te_mask[sub], depth)
    transfer = {}
    for d in range(1, 13):
        a, _, m = eval_chunk_sums(model, te_inp[sub], te_tgt[sub], te_mask[sub], d)
        transfer[d] = float(a) / float(m)
    extras = {
        "pass_acc": [float(x) for x in pass_acc],
        "gate_open": None if gate_open is None else [float(x) for x in gate_open],
        "transfer": transfer,
    }
    return acc_sum / count, ce_sum / count, n_params, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="parity", choices=list(TASKS) + ["memorize"])
    ap.add_argument("--arch", default="refiner", choices=["refiner", "vanilla"],
                    help="refiner = one shared block looped `depth` times; vanilla = `depth` distinct blocks (matched FLOPs, ~depth× block params)")
    ap.add_argument("--mem-pairs", type=int, default=1024,
                    help="memorize task: size N of the fixed random key→value dictionary (vocab = N); sweep N to read capacity")
    ap.add_argument("--dim", type=int, default=96,
                    help="model width (must divide heads=4 evenly); shrink it to make capacity bind sooner on the memorize probe")
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-bias", type=float, default=0.0,
                    help="init bias of the update gate; negative = retention-biased (stabilizes deep recurrence). refiner-only.")
    ap.add_argument("--grad-last", type=int, default=None,
                    help="backprop through only the last J refinement iterations (truncated backprop, #64); default = full backprop. refiner-only.")
    ap.add_argument("--per-pass-loss", action="store_true",
                    help="deep supervision (#75): grade every refinement pass against the target (uniform mean over passes). refiner-only.")
    ap.add_argument("--islands", action="store_true",
                    help="cut the gradient chain at every pass boundary (#75) — pair with --per-pass-loss. refiner-only.")
    ap.add_argument("--readouts", action="store_true",
                    help="print #75 readouts: per-pass accuracy, gate openness per pass, depth-transfer curve (eval at depths 1..12). refiner-only.")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-seq", type=int, default=SEQ)
    ap.add_argument("--test-seq", type=int, default=None,
                    help="eval length; > train-seq makes it a length-generalization probe (default: = train-seq)")
    args = ap.parse_args()

    if args.task == "memorize":
        task_fn = memorize_task(args.mem_pairs)
        vocab = args.mem_pairs
        task_desc = f"memorize[N={args.mem_pairs}]"
    else:
        task_fn = TASKS[args.task]
        vocab = VOCAB[args.task]
        task_desc = args.task
    depths = [int(d) for d in args.depths.split(",")]
    test_seq = args.train_seq if args.test_seq is None else args.test_seq

    print(f"== depth ablation: arch={args.arch} dim={args.dim} task={task_desc} (train_seq={args.train_seq}, test_seq={test_seq}, vocab={vocab}) steps={args.steps} seed={args.seed} gate_bias={args.gate_bias} grad_last={args.grad_last} per_pass={args.per_pass_loss} islands={args.islands} lr={args.lr} ==")
    print(f"{'depth':>6} {'params':>9} {'val_acc':>9} {'val_ce':>9} {'sec':>7}")
    results = {}
    for d in depths:
        t0 = time.time()
        out = train_one(task_fn, vocab, d, arch=args.arch, dim=args.dim,
                        steps=args.steps, seed=args.seed, gate_bias=args.gate_bias,
                        grad_last=args.grad_last, per_pass_loss=args.per_pass_loss,
                        islands=args.islands, readouts=args.readouts, lr=args.lr,
                        train_seq=args.train_seq, test_seq=test_seq)
        acc, ce, n_params = out[:3]
        results[d] = (acc, ce)
        print(f"{d:>6} {n_params / 1e6:>8.2f}M {acc:>9.4f} {ce:>9.4f} {time.time() - t0:>7.1f}")
        if args.readouts:
            extras = out[3]
            print("  pass_acc: " + " ".join(f"{a:.4f}" for a in extras["pass_acc"]))
            if extras["gate_open"] is not None:
                print("  gate_open: " + " ".join(f"{g:.4f}" for g in extras["gate_open"]))
            print("  transfer: " + " ".join(f"d{k}={v:.4f}" for k, v in extras["transfer"].items()))

    base_acc = results[depths[0]][0]
    best_d = max(results, key=lambda d: results[d][0])
    print(f"\ndepth {depths[0]}: acc {base_acc:.4f}  ->  best depth {best_d}: acc {results[best_d][0]:.4f}  |  gain {results[best_d][0] - base_acc:+.4f}")


if __name__ == "__main__":
    main()
