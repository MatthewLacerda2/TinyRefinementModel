"""Where does a grad step's VRAM go? — a memory profiler for the recurring OOM problem.

OOMs (the dim960 base run, the SFT phase switch, past dead runs) have been debugged by
launch-and-watch-it-die roulette. This replaces that with a measurement: build the real
model + optimizer at the *current config*, compile the real grad step, and report

  1. the static memory breakdown — arguments (params + inputs), output (grads), and the
     peak **temp/scratch** where transient activations live (this is what OOMs a run);
  2. the **largest tensor shapes in the compiled HLO**, aggregated — which names the
     biggest buffers (e.g. a ``f32[1,512,50304]`` logit slab) so a mystery transient
     becomes a shape you can point at.

Static analysis (1) and the HLO scan (2) work even when actually running would OOM, so
this is usable precisely when you need it most. ``--run`` additionally executes one step
and reads the driver's peak / largest-allocation stats (skip it if the config OOMs).

Set the architecture you want in config.py (LATENT_DIM, NUM_HEADS) first — both arches
read it — then:

    PYTHONPATH=. python tools/mem_profile.py --arch refiner --depth 8
    PYTHONPATH=. python tools/mem_profile.py --arch reasoner --depth 8 --top 15
"""

import os
# Measure the real f16 GPU footprint, on demand (no 75% land-grab that hides the peak).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import re
from collections import defaultdict

import jax
import jax.numpy as jnp
from flax import nnx

from config import VOCAB_SIZE, MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS

_DT_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "s32": 4, "s8": 1, "u32": 4, "pred": 1, "f64": 8}


def _gib(n):
    return f"{n / 1024**3:7.3f} GiB"


def _build(arch):
    rngs = nnx.Rngs(0)
    if arch == "refiner":
        from plan_a_trainer import RefinerForTraining
        return RefinerForTraining(LATENT_DIM, rngs)
    from model import UniversalReasoner
    return UniversalReasoner(LATENT_DIM, rngs, batch_size=1)


def _top_hlo_shapes(hlo_text, top):
    """Aggregate every typed array shape in the compiled HLO by total bytes, so the
    biggest buffers (the transient suspects) surface with their shape and count."""
    sizes = defaultdict(lambda: [0, 0])  # shape_str -> [count, total_bytes]
    for dt, dims in re.findall(r"\b(f32|f16|bf16|s32|s8|u32|pred|f64)\[([\d,]+)\]", hlo_text):
        n = 1
        for d in dims.split(","):
            n *= int(d)
        b = n * _DT_BYTES[dt]
        key = f"{dt}[{dims}]"
        sizes[key][0] += 1
        sizes[key][1] += b
    ranked = sorted(sizes.items(), key=lambda kv: kv[1][1], reverse=True)
    return ranked[:top]


def main():
    ap = argparse.ArgumentParser(description="grad-step VRAM profiler")
    ap.add_argument("--arch", choices=["reasoner", "refiner"], default="reasoner")
    ap.add_argument("--depth", type=int, default=8, help="reasoning depth (8 = deepest, peak)")
    ap.add_argument("--top", type=int, default=12, help="how many largest HLO shapes to list")
    ap.add_argument("--run", action="store_true", help="also execute one step (skip if it OOMs)")
    args = ap.parse_args()

    from grad_step import compute_grad_step

    model = _build(args.arch)
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"arch={args.arch}  dim={LATENT_DIM}  heads={NUM_HEADS}  "
          f"params={n_params / 1e6:.1f}M  depth={args.depth}  seq={MAX_SEQ_LEN}  vocab={VOCAB_SIZE}")

    batch = jax.random.randint(jax.random.PRNGKey(0), (1, 2 * MAX_SEQ_LEN + 1), 0, VOCAB_SIZE, dtype=jnp.int32)
    doc_boundary = jnp.zeros((1,), dtype=bool)
    step = jnp.array(0)

    compiled = compute_grad_step.lower(model, batch, step, args.depth, doc_boundary=doc_boundary).compile()

    ma = compiled.memory_analysis()
    arg = getattr(ma, "argument_size_in_bytes", 0)
    out = getattr(ma, "output_size_in_bytes", 0)
    tmp = getattr(ma, "temp_size_in_bytes", 0)
    print("\n=== grad-step memory_analysis (static, compile-time) ===")
    print(f"  arguments (params + inputs):                 {_gib(arg)}")
    print(f"  output (gradients + aux):                    {_gib(out)}")
    print(f"  temp / scratch  (PEAK transient activations): {_gib(tmp)}  <-- what OOMs a run")
    print(f"  ~grad-step footprint (arg+out+temp):         {_gib(arg + out + tmp)}")

    print(f"\n=== largest {args.top} tensor shapes in the compiled HLO (total bytes) ===")
    for shape, (count, total) in _top_hlo_shapes(compiled.as_text(), args.top):
        print(f"  {_gib(total)}  x{count:<4}  {shape}")

    if args.run:
        print("\n=== executing one step (driver stats) ===")
        try:
            loss, *_ = compute_grad_step(model, batch, step, args.depth, doc_boundary)
            loss.block_until_ready()
            ms = jax.devices()[0].memory_stats()
            print(f"  peak_bytes_in_use: {_gib(ms.get('peak_bytes_in_use', 0))}")
            print(f"  largest_alloc:     {_gib(ms.get('largest_alloc_size', 0))}")
        except Exception as e:
            print(f"  step raised (likely OOM): {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
