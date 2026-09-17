"""How much of the card does training take? Measured on the path we ship (#161).

    python -m instruments.vram_headroom_smoke                      # the config a launch would run
    python -m instruments.vram_headroom_smoke --layers 10          # a candidate plain stack
    for L in 8 9 10; do python -m instruments.vram_headroom_smoke --layers $L; done

The question is always the same: will a launch at this config survive? The old
version answered something adjacent, precisely and confidently, and was cited as
clearance anyway. It ran the `platform` allocator, which cannot fragment, while
fragmentation killed every base run; it polled nvidia-smi every 150ms and
reported the sampled maximum as a peak (batch 2 once read *lower* than batch 1);
it built its own optimizer with an f32 first moment; and it never crossed an
optimizer apply or ran the validation probe the trainer also runs.

Now, in one fresh process per config:
  * production's allocator and memory fraction, from the same keys trm/train/start.py
    sets (tests/apparatus/test_instrument_environment.py holds them together);
  * production's optimizer chain (bf16 first moment, masked weight decay, MultiSteps);
  * ACCUMULATION_STEPS + 1 micro-steps, so the optimizer apply is inside the
    measurement, cycling every sampled depth so each depth's program is compiled
    (the plain stack too: the trainer passes it the sampled depth as a static jit
    argument, #316);
  * then one validation probe, as the trainer runs on its cadence.

Two numbers, kept apart because they fail differently:
  * arena peak against the arena's limit — `memory_stats()` `peak_bytes_in_use`
    and `bytes_limit`, both exact. Under cuda_async at MEM_FRACTION 0.85 the pool is
    reserved up front, so nvidia-smi reads ~the limit whatever the model uses;
    the headroom that decides an OOM is limit minus peak, and only the allocator
    knows it.
  * outside the arena — nvidia-smi's used minus that limit: the CUDA context and
    the driver's compiled-graph buffers (#160's OOM was there). A poll, so a
    sample; it is labelled as one.

Still not covered: the checkpoint managers' host-side staging and a real data
pipeline. Synthetic tokens are fine for memory — VRAM depends on shapes, not
values. The only complete answer remains a real launch through its first
optimizer apply and checkpoint.
"""

import os

# Production's environment, not a convenient one: the same setdefault lines as
# trm/train/start.py, so an override from the shell still reaches both.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import argparse
import threading
import time

import jax
import jax.numpy as jnp
from flax import nnx

from instruments import results
from instruments._common import gpu_memory_used_mib, param_count
from instruments.arch import add_arch_argument, build
from trm.config import (ACCUMULATION_STEPS, BATCH_SIZE, LATENT_DIM, MAX_SEQ_LEN, MAX_STEPS_LIMIT,
                        NUM_HEADS, PLAIN_LAYERS, REFINER_ENCODER_LAYERS, VOCAB_SIZE)
from trm.train.grad_step import apply_grads, compute_grad_step
from trm.train.optimizers import optimizer_chain
from trm.train.validation import _val_ce_sums

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "arena peak / headroom": ("measured", "memory_stats() peak_bytes_in_use against bytes_limit under production's allocator"),
    "outside arena": ("sampled", "nvidia-smi poll minus the arena limit; a transient can be missed"),
}

CARD_MIB = 6144


class CardSampler:
    """nvidia-smi's memory.used, polled: the reserved pool plus everything outside
    it. A poll, so reported as a sample and never as a peak."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = gpu_memory_used_mib()  # None when the card cannot be read: the sample is skipped
            if used is not None:
                self.peak_mib = max(self.peak_mib, used)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)


def depth_schedule(micro_steps, max_depth=MAX_STEPS_LIMIT):
    """Depths to run, cycling 1..max so every depth program is compiled — the
    trainer samples them all, and each compiled graph costs driver memory.

    For every arch, the plain stack included. It ignores depth, but the trainer
    still samples one per micro-step and passes it to the grad step as a static jit
    argument, so a plain run holds one compiled program per depth (#316). This used
    to compile one program for plain and so measured less than a launch holds. When
    #316 makes depth an arch property, this follows the trainer."""
    return [(i % max_depth) + 1 for i in range(micro_steps)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arch_argument(ap)
    ap.add_argument("--dim", type=int, default=LATENT_DIM, help="LATENT_DIM (must be divisible by --heads)")
    ap.add_argument("--heads", type=int, default=NUM_HEADS)
    ap.add_argument("--layers", type=int, default=PLAIN_LAYERS, help="block count for --arch plain")
    ap.add_argument("--encoder-layers", type=int, default=REFINER_ENCODER_LAYERS, help="for --arch refiner")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="micro-batch (per accumulation step)")
    ap.add_argument("--depth", type=int, default=MAX_STEPS_LIMIT,
                    help="deepest sampled depth; the run compiles one program per depth 1..DEPTH, "
                         "following depth_schedule. Until #316 the trainer compiled one program per "
                         "sampled depth for plain too, so there --depth 1 measures one program where "
                         "such a launch holds MAX_STEPS_LIMIT of them")
    ap.add_argument("--micro-steps", type=int, default=ACCUMULATION_STEPS + 1,
                    help="default crosses one optimizer apply (ACCUMULATION_STEPS + 1)")
    args = ap.parse_args(argv)
    if args.dim % args.heads:
        raise SystemExit(f"--dim {args.dim} not divisible by --heads {args.heads}")

    overrides = {"plain": {"num_heads": args.heads, "num_layers": args.layers},
                 "refiner": {"num_heads": args.heads, "encoder_layers": args.encoder_layers},
                 "reasoner": {}}[args.arch]
    model = build(args.arch, dim=args.dim, **overrides)
    shape = {"plain": f"{args.layers} layers", "refiner": f"{args.encoder_layers} encoder + loop to depth {args.depth}",
             "reasoner": "reasoner"}[args.arch]
    print(f"sizing {args.arch} ({shape}) at dim {args.dim}, {args.heads} heads, batch {args.batch}, "
          f"{param_count(model) / 1e6:.1f}M params, allocator {os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']}")

    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)
    batch = jax.random.randint(jax.random.PRNGKey(0), (args.batch, 2 * MAX_SEQ_LEN + 1), 0, VOCAB_SIZE,
                               dtype=jnp.int32)
    doc_boundary = jnp.zeros((args.batch,), dtype=bool)
    device = jax.local_devices()[0]

    with CardSampler() as card:
        for step, depth in enumerate(depth_schedule(args.micro_steps, args.depth)):
            loss, _out, grads, _gn = compute_grad_step(model, batch, step, depth, doc_boundary)
            apply_grads(optimizer, grads, model)
        float(loss)
        with model.isolated_state():
            model.reset_state()
            float(_val_ce_sums(model, batch[:1])[0])
        time.sleep(0.5)

    stats = device.memory_stats()
    arena_mib, limit_mib = stats["peak_bytes_in_use"] / 2**20, stats["bytes_limit"] / 2**20
    outside_mib = max(card.peak_mib - limit_mib, 0.0)
    print(f"arena peak (exact):      {arena_mib:6.0f} MiB of a {limit_mib:.0f} MiB limit "
          f"({arena_mib / limit_mib:.1%}) — headroom {limit_mib - arena_mib:.0f} MiB")
    print(f"outside arena (sampled): {outside_mib:6.0f} MiB (context + driver graph buffers; "
          f"{CARD_MIB - card.peak_mib:.0f} MiB of the card never touched)")
    print(f"crossed {args.micro_steps // ACCUMULATION_STEPS} optimizer apply(s) and one validation probe")
    if args.arch == "plain":
        print(f"note: plain compiled {min(args.depth, args.micro_steps)} depth program(s) here; until #316 "
              f"the trainer compiled one program per sampled depth for plain too. The plain peaks "
              f"recorded in trm/config.py and model_stats.MEASURED_PEAKS were taken with ONE program, "
              f"so compare this reading with them only at the program count the trainer uses.")
    results.emit(f"{args.arch}-{shape.split()[0]}", arena_peak_mib=arena_mib, arena_limit_mib=limit_mib,
                 headroom_mib=limit_mib - arena_mib, outside_arena_sampled_mib=outside_mib)


if __name__ == "__main__":
    main()
