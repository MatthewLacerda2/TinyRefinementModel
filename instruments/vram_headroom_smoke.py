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
    measurement, through the trainer's HotPath (#316, #474);
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
from trm.config import VOCAB_SIZE
from trm.model import build_model
from trm.settings import CONFIG
from trm.train.grad_step import HotPath
from trm.train.optimizers import optimizer_chain
from trm.train.schedules import Schedules
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dim", type=int, default=CONFIG.LATENT_DIM, help="LATENT_DIM (must be divisible by --heads)")
    ap.add_argument("--heads", type=int, default=CONFIG.NUM_HEADS)
    ap.add_argument("--layers", type=int, default=CONFIG.PLAIN_LAYERS, help="block count (PLAIN_LAYERS)")
    ap.add_argument("--batch", type=int, default=CONFIG.BATCH_SIZE, help="micro-batch (per accumulation step)")
    ap.add_argument("--micro-steps", type=int, default=CONFIG.ACCUMULATION_STEPS + 1,
                    help="default crosses one optimizer apply (ACCUMULATION_STEPS + 1)")
    args = ap.parse_args(argv)
    if args.dim % args.heads:
        raise SystemExit(f"--dim {args.dim} not divisible by --heads {args.heads}")

    model = build_model(CONFIG, nnx.Rngs(0), dim=args.dim, num_heads=args.heads, num_layers=args.layers)
    print(f"sizing {args.layers} layers at dim {args.dim}, {args.heads} heads, batch {args.batch}, "
          f"{param_count(model) / 1e6:.1f}M params, allocator {os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']}")

    optimizer = nnx.Optimizer(model, optimizer_chain(CONFIG, Schedules.of(CONFIG).learning_rate), wrt=nnx.Param)
    batch = jax.random.randint(jax.random.PRNGKey(0), (args.batch, 2 * CONFIG.MAX_SEQ_LEN + 1), 0, VOCAB_SIZE,
                               dtype=jnp.int32)
    device = jax.local_devices()[0]

    # The trainer's own step (#474): split state, one graph walk, donated buffers (#548).
    hot = HotPath(model, optimizer, z_loss_weight=CONFIG.Z_LOSS_WEIGHT)
    with CardSampler() as card:
        for _ in range(args.micro_steps):
            loss, _out, _gn, _ = hot.step(batch)
            hot.commit()
        float(loss)
        float(_val_ce_sums(hot.model, batch[:1])[0])
        time.sleep(0.5)

    stats = device.memory_stats()
    arena_mib, limit_mib = stats["peak_bytes_in_use"] / 2**20, stats["bytes_limit"] / 2**20
    outside_mib = max(card.peak_mib - limit_mib, 0.0)
    print(f"arena peak (exact):      {arena_mib:6.0f} MiB of a {limit_mib:.0f} MiB limit "
          f"({arena_mib / limit_mib:.1%}) — headroom {limit_mib - arena_mib:.0f} MiB")
    print(f"outside arena (sampled): {outside_mib:6.0f} MiB (context + driver graph buffers; "
          f"{CARD_MIB - card.peak_mib:.0f} MiB of the card never touched)")
    print(f"crossed {args.micro_steps // CONFIG.ACCUMULATION_STEPS} optimizer apply(s) and one validation probe")
    results.emit(f"plain-{args.layers}", arena_peak_mib=arena_mib, arena_limit_mib=limit_mib,
                 headroom_mib=limit_mib - arena_mib, outside_arena_sampled_mib=outside_mib)


if __name__ == "__main__":
    main()
