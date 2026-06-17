"""Stage 1 transfer probe: does refinement depth lower held-out CE, by domain?

Plan A's depth recurrence is proven to do real sequential work on toy
state-tracking (docs/findings/2026-06-16-plan-a-depth-ablation.md). This asks the
real-model question on the 77.65M fineweb checkpoint: does looping the shared block
deeper lower next-token CE — and does it help MORE on compositional domains (code,
math) than on associative web text? A depth gradient on code/math but flat on web
would be the first sign the toy-task win transfers to language. A flat curve
everywhere is also informative: it says the depth win does not (yet) reach LM CE,
or the base is too undertrained to exhibit it.

Same checkpoint, same held-out data, depth as the only swept variable — the direct
language analog of the ablation's within-model depth sweep.

Run against the latest (or a given) checkpoint:
    DATA_ROOT=runs/data PYTHONPATH=. python tools/eval_refiner_depth_transfer.py \
        [--depths 1,2,4,8] [--batches 32] [--skip 3000000]
On CPU (no GPU contention with a live run) prepend FORCE_F32_COMPUTE=1 — CPU XLA
cannot lower the f16-with-f32-accumulation matmuls. On GPU, drop the mem fraction
(XLA_PYTHON_CLIENT_MEM_FRACTION) if a training run already holds the card.
"""

import os
import argparse

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp

from config import LATENT_DIM, MAX_SEQ_LEN, PAD_TOKEN_ID, BATCH_SIZE, resolve_root
from plan_a_trainer import RefinerForTraining
from data_loaders import TextDataGenerator
from checkpoint_utils import discover_latest_checkpoint_run

# Curriculum domains, associative -> compositional. Depth should pay off most where
# the prediction needs multi-step aggregation (math/code), least on web text.
# Each domain carries its own held-out skip: the run consumes <5% of any corpus, so
# these sit deep past the trained range while leaving a held-out tail. A single skip
# can't serve all three — 3M overshoots the smaller code/math corpora entirely.
#   corpus sizes (stride-windows): web ~4.34M, code ~1.89M, math ~1.12M
DOMAINS = {
    "web": ("pretrain/fineweb-edu", 3_000_000),
    "code": ("pretrain/codeparrot", 1_500_000),
    "math": ("pretrain/finemath", 900_000),
}


def restore_refiner(checkpoint_path=None):
    """Model-only restore into the production refiner wrapper (RefinerForTraining),
    so the checkpoint's saved 'model' state loads with matching structure."""
    if checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is None:
            raise SystemExit("No checkpointed run found under runs/.")
        print(f"🔎 Using latest checkpointed run: {run_id}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    model = RefinerForTraining(LATENT_DIM, nnx.Rngs(42))
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "optimizer", "monitor_state", "step"),
    )
    latest = mngr.latest_step()
    if latest is None:
        raise SystemExit(f"No checkpoint found under {checkpoint_path}")
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
    )
    nnx.update(model, restored["model"])
    print(f"📖 Restored refiner from step {latest} ({checkpoint_path})")
    return model, latest


def load_domain_batches(source, num_batches, skip):
    """Held-out batches for one domain, skipping past the trained range."""
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise SystemExit("DATA_ROOT is not set (try DATA_ROOT=runs/data).")
    gen = TextDataGenerator(f"{resolve_root(data_root)}/{source}")
    gen.skip_count = skip
    batches = []
    while len(batches) < num_batches:
        batch, _ = gen.get_batch(BATCH_SIZE)
        if batch is None:
            break
        batches.append(batch)
    return batches


@nnx.jit(static_argnames=["depth"])
def _ce_sums(model, batch, depth):
    """Masked next-token CE sum + token count over both windows at a fixed depth —
    mirrors trainer._val_ce_sums, but depth is the swept argument here."""
    seq1_in, seq1_out = batch[:, :MAX_SEQ_LEN], batch[:, 1:MAX_SEQ_LEN + 1]
    seq2_in, seq2_out = batch[:, MAX_SEQ_LEN:2 * MAX_SEQ_LEN], batch[:, MAX_SEQ_LEN + 1:2 * MAX_SEQ_LEN + 1]
    out1 = model(seq1_in, max_steps=depth, training=False)
    out2 = model(seq2_in, max_steps=depth, training=False)
    total, count = jnp.array(0.0), jnp.array(0)
    for logits, targets in ((out1.logits, seq1_out), (out2.logits, seq2_out)):
        mask = targets != PAD_TOKEN_ID
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        total += jnp.sum(ce * mask)
        count += jnp.sum(mask)
    return total, count


def main():
    ap = argparse.ArgumentParser(description="depth-swept held-out CE by domain")
    ap.add_argument("--checkpoint-path", default=None, help="Orbax dir (default: latest run)")
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--batches", type=int, default=32, help="held-out batches per domain")
    ap.add_argument("--skip", type=int, default=None,
                    help="override the per-domain held-out skip (applies to all domains)")
    args = ap.parse_args()
    depths = [int(d) for d in args.depths.split(",")]

    model, step = restore_refiner(args.checkpoint_path)
    print(f"== Refiner depth-transfer probe: step {step} | {args.batches} batches/domain | depths {depths} ==")
    print(f"{'domain':>6} " + " ".join(f"{'d' + str(d):>9}" for d in depths) + "   d1->best")
    for name, (source, domain_skip) in DOMAINS.items():
        skip = args.skip if args.skip is not None else domain_skip
        batches = load_domain_batches(source, args.batches, skip)
        if not batches:
            print(f"{name:>6}   (no held-out data in {source})")
            continue
        ces = {}
        for d in depths:
            tot, cnt = jnp.array(0.0), jnp.array(0)
            for b in batches:
                t, c = _ce_sums(model, b, d)
                tot += t
                cnt += c
            ces[d] = float(tot / cnt)
        best = min(ces.values())
        row = " ".join(f"{ces[d]:9.4f}" for d in depths)
        print(f"{name:>6} {row}   {ces[depths[0]] - best:+.4f}")


if __name__ == "__main__":
    main()
