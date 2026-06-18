"""Stage 2 transfer probe (Option B): did LM pretraining preserve depth-usefulness?

Stage 1 showed refinement depth is invisible to language-modeling CE on every
domain (docs/findings/2026-06-16-plan-a-depth-ablation.md + the Stage-1 run). Two
explanations: (benign) LM CE merely dilutes the rare depth-requiring tokens, or
(sobering) LM pretraining collapsed the model onto a depth-invariant solution that
ignores the recurrence. Perplexity cannot separate them; only a task that
*structurally* requires multi-step inference can.

This fine-tunes the FULL-scale CausalRefiner on non-commutative state tracking (the
toy task depth provably needs — Liu et al. automata regime) at each fixed depth,
from two inits: random (scratch) and the pretrained checkpoint. The contrast of the
two depth->accuracy staircases is the read-out:
  - scratch staircase AND pretrained staircase -> depth survives pretraining (good)
  - scratch staircase, pretrained flat          -> pretraining killed depth (sobering)
  - both flat                                    -> task too easy at this scale; retune
A from-scratch control at the SAME (full) architecture is essential: a dim=512 model
might solve the task at depth-1 with no headroom (as cumsum5 did at tiny scale), so a
flat pretrained curve is only meaningful against what scratch does at the same size.

The model is built at the checkpoint's max_seq_len (512) and fed short sequences, so
the pretrained state restores with matching shapes; difficulty is tuned via --seq /
--n-states so the scratch model is not saturated at depth 1.

    DATA_ROOT unused. GPU only (the full model + optimizer needs the whole card, so
    pause any live training first). Example:
    PYTHONPATH=. python tools/eval_refiner_depth_finetune.py \
        --init both --depths 1,2,4,8 --steps 1500 --seq 48 --n-states 8
"""

import os
import argparse
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from config import LATENT_DIM, MAX_SEQ_LEN
from plan_a_trainer import RefinerForTraining
from ablation_harness import state_tracking_task
from checkpoint_utils import discover_latest_checkpoint_run

# Map the task's symbols onto distinct, arbitrary-but-fixed real token ids, so the
# pretrained embedding rows and tied head are genuinely exercised (not the unused
# tail of the 100352-vocab). Generator-choice tokens and state tokens stay disjoint.
INPUT_BASE = 1000
STATE_BASE = 2000


def build_model(init, checkpoint_path):
    """Full-scale refiner at the checkpoint's architecture. init='scratch' -> random;
    init='pretrained' -> restore the saved model state."""
    model = RefinerForTraining(LATENT_DIM, nnx.Rngs(0))  # max_seq_len defaults to 512
    if init == "pretrained":
        if checkpoint_path is None:
            checkpoint_path, run_id = discover_latest_checkpoint_run()
            if checkpoint_path is None:
                raise SystemExit("No checkpointed run found under runs/.")
            print(f"🔎 pretrained init from latest run: {run_id}")
        checkpoint_path = os.path.abspath(checkpoint_path)
        mngr = ocp.CheckpointManager(
            checkpoint_path,
            item_names=("model", "optimizer", "monitor_state", "step"),
        )
        latest = mngr.latest_step()
        if latest is None:
            raise SystemExit(f"No checkpoint under {checkpoint_path}")
        restored = mngr.restore(
            latest, args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))))
        nnx.update(model, restored["model"])
        print(f"📖 restored pretrained weights from step {latest}")
    return model


def make_task(seed, n_pool, n_test, seq, n_states, n_gen):
    """State-tracking pools mapped to real token ids; held-out test pool drawn
    separately so the score reflects generalization, not memorization."""
    key = jax.random.PRNGKey(seed)
    k_tr, k_te = jax.random.split(key)
    tr_in, tr_tgt, tr_mask = state_tracking_task(k_tr, n_pool, seq, n_states, n_gen)
    te_in, te_tgt, te_mask = state_tracking_task(k_te, n_test, seq, n_states, n_gen)
    # Shift symbols into the real-token id ranges.
    return (
        (tr_in + INPUT_BASE, tr_tgt + STATE_BASE, tr_mask),
        (te_in + INPUT_BASE, te_tgt + STATE_BASE, te_mask),
    )


def train_eval(model, train, test, depth, steps, batch, lr, seed):
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=0.01), wrt=nnx.Param)
    tr_in, tr_tgt, tr_mask = train

    @nnx.jit(static_argnames=["depth"])
    def step(model, opt, key, depth):
        idx = jax.random.randint(key, (batch,), 0, tr_in.shape[0])
        inp, tgt, mask = tr_in[idx], tr_tgt[idx], tr_mask[idx]
        def loss_fn(m):
            logits = m(inp, max_steps=depth, training=True).logits
            ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
            return jnp.sum(ce * mask) / jnp.sum(mask)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit(static_argnames=["depth"])
    def eval_chunk(model, inp, tgt, mask, depth):
        # Chunked: the full test pool's logits ([n_test*seq, 100352]) would be tens of
        # GB; score batch-sized chunks and accumulate masked sums instead.
        logits = model(inp, max_steps=depth, training=False).logits
        correct = jnp.sum((logits.argmax(-1) == tgt) * mask)
        ce = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt) * mask)
        return correct, ce, jnp.sum(mask)

    key = jax.random.PRNGKey(seed)
    for _ in range(steps):
        key, k = jax.random.split(key)
        step(model, opt, k, depth)

    te_in, te_tgt, te_mask = test
    correct = ce = count = 0.0
    for i in range(0, te_in.shape[0], batch):
        c, e, m = eval_chunk(model, te_in[i:i + batch], te_tgt[i:i + batch], te_mask[i:i + batch], depth)
        correct += float(c)
        ce += float(e)
        count += float(m)
    return correct / count, ce / count


def main():
    ap = argparse.ArgumentParser(description="Stage 2: full-scale depth ablation, scratch vs pretrained")
    ap.add_argument("--init", default="both", choices=["scratch", "pretrained", "both"])
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16, help="small: full-vocab logits are the memory hog")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--n-states", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=4)
    ap.add_argument("--n-pool", type=int, default=8192)
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seq > MAX_SEQ_LEN:
        raise SystemExit(f"--seq {args.seq} exceeds model max_seq_len {MAX_SEQ_LEN}")
    depths = [int(d) for d in args.depths.split(",")]
    inits = ["scratch", "pretrained"] if args.init == "both" else [args.init]

    train, test = make_task(args.seed, args.n_pool, args.n_test, args.seq, args.n_states, args.n_gen)
    chance = 1.0 / args.n_states
    print(f"== Stage 2 depth fine-tune | seq={args.seq} n_states={args.n_states} n_gen={args.n_gen} "
          f"| steps={args.steps} batch={args.batch} lr={args.lr} | chance acc={chance:.3f} ==")
    print(f"{'init':>10} {'depth':>6} {'val_acc':>9} {'val_ce':>9} {'sec':>7}")
    for init in inits:
        for d in depths:
            model = build_model(init, args.checkpoint_path)
            t0 = time.time()
            acc, ce = train_eval(model, train, test, d, args.steps, args.batch, args.lr, args.seed)
            print(f"{init:>10} {d:>6} {acc:>9.4f} {ce:>9.4f} {time.time() - t0:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
