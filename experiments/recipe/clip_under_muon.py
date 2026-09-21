"""Does the global-norm clip reach the weights through a scale-invariant optimizer? (#447)

`trm/train/optimizers.py` runs `clip_by_global_norm(CLIP_NORM)` in front of Muon on
the matrices and AdamW on the rest. Both are invariant to a *uniform* rescale of the
gradient — Muon divides its momentum by the Frobenius norm before Newton-Schulz,
AdamW's m/sqrt(v) cancels a common factor — so the clip cannot cap a step's size.
What it can do is change how much each step *weighs* inside the moment estimates:
a step whose gradient was large counts for less. Whether that is a real effect or a
rounding error depends on how fast the gradient norm moves against how long each
optimizer remembers, and that is what this measures.

**A replay, not a training run.** One gradient sequence is fed identically to two
optimizer states — the clip in front, and nothing in front — and the updates each
emits are compared, per partition, at every step. Same parameters, same gradients:
the only difference between the two updates is the clip's effect on the history.
The parameters follow the unclipped arm, so the gradients come from a trajectory
the shipped optimizer would actually take.

The directions are real — a small plain model on real FineWeb-Edu tokens, folded
into a small vocabulary — and every step's gradient is rescaled to the **base run's
recorded pre-clip norm** at that step (`applied_grad_norm`, logged every five opt
steps; interpolated in log space between them, with the logged step-to-step scatter
added back as noise, since smoothing it away would understate exactly the
variation the clip reweights).

    python -m experiments.recipe.clip_under_muon \\
        --metrics runs/run_20260920_191351/metrics.csv --steps 1300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np

from instruments.results import emit

# Step bands the readout is summarised over: the base run's clip was active 100% of
# the time in the first three, 28% in the fourth, and not once after step 1,000.
# Past the last logged step the norm is held at its last logged trend value.
BANDS = ((0, 100), (100, 250), (250, 512), (512, 1000), (1000, 1500), (1500, 2000),
         (2000, 3000), (3000, 10**9))


def norm_profile(metrics_csv: pathlib.Path, steps: int, seed: int, scatter_scale: float = 1.0) -> np.ndarray:
    """The base run's pre-clip global norm at every opt step in [0, steps).

    Logged every few steps, so the gaps are filled by interpolating log(norm) and
    adding back noise with the scatter the logged points show around that trend,
    times `scatter_scale`. The true step-to-step variation is not logged, and the
    clip's effect on a momentum depends on it, so 0 (the smooth trend alone) is the
    other bound worth reading.
    """
    # Only the logged points inside the replayed horizon: the source is usually a live
    # run's metrics.csv, and rows it appends later must not change a replay already
    # recorded. Past the last logged point, the norm holds its last value.
    points = []
    with metrics_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("applied_grad_norm") and int(row["step"]) < steps:
                points.append((int(row["step"]), float(row["applied_grad_norm"])))
    if not points:
        raise SystemExit(f"{metrics_csv} has no applied_grad_norm column values")
    xs = np.array([p[0] for p in points], dtype=float)
    logs = np.log([p[1] for p in points])
    trend = np.convolve(logs, np.ones(5) / 5, mode="same")
    scatter = float(np.std((logs - trend)[2:-2])) if len(logs) > 8 else 0.0
    wanted = np.arange(steps, dtype=float)
    smooth = np.interp(wanted, xs, logs)
    noise = np.random.default_rng(seed).normal(0.0, scatter * scatter_scale, size=steps)
    logged = np.isin(wanted, xs)
    return np.exp(np.where(logged, smooth, smooth + noise))


def token_batches(data_dir: pathlib.Path, vocab: int, seq: int, batch: int, seed: int):
    shards = sorted(data_dir.glob("chunk_*.npy"))
    if not shards:
        raise SystemExit(f"no chunk_*.npy under {data_dir}")
    tokens = np.load(shards[0], mmap_mode="r")
    rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, len(tokens) - seq - 1, size=batch)
        rows = np.stack([np.asarray(tokens[s:s + seq + 1]) for s in starts])
        yield (rows % vocab).astype(np.int32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=pathlib.Path, required=True,
                    help="a run's metrics.csv — its applied_grad_norm is the norm profile replayed")
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("runs/data/pretrain/fineweb-edu"))
    ap.add_argument("--steps", type=int, default=1300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4,
                    help="moves the parameter trajectory only; both arms share it, so it "
                         "cannot move the comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scatter-scale", type=float, default=1.0,
                    help="multiplies the step-to-step noise put back between logged norms; "
                         "0 replays the smooth trend alone")
    ap.add_argument("--csv-out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from trm.model import build_model
    from trm.train.optimizers import inner_optimizer, muon_partition

    model = build_model("plain", args.dim, nnx.Rngs(args.seed), vocab_size=args.vocab,
                        num_heads=4, num_layers=args.layers, max_seq_len=args.seq)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    labels = jax.tree_util.tree_leaves(muon_partition(params))

    def loss_fn(p, tokens):
        logits = nnx.merge(graphdef, p, rest)(tokens[:, :-1], training=False).logits
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean()

    grad_fn = jax.jit(jax.grad(loss_fn))
    optimizer = inner_optimizer(optax.constant_schedule(args.lr))
    update = jax.jit(optimizer.update)
    clip = optax.clip_by_global_norm(args.clip)

    def flat(tree, which):
        leaves = [np.asarray(leaf, dtype=np.float64).ravel()
                  for leaf, label in zip(jax.tree_util.tree_leaves(tree), labels) if label == which]
        return np.concatenate(leaves)

    def compare(a, b):
        out = {}
        for which in ("muon", "adam"):
            x, y = flat(a, which), flat(b, which)
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            out[which] = (float(x @ y / (nx * ny)), float(nx / ny))
        return out

    def rescale(tree, factor):
        return jax.tree_util.tree_map(lambda g: g * factor, tree)

    profile = norm_profile(args.metrics, args.steps, args.seed, args.scatter_scale)
    batches = token_batches(args.data, args.vocab, args.seq, args.batch, args.seed)

    # ── the two exact checks: what the clip cannot do ──────────────────────────
    probe = [grad_fn(params, jnp.asarray(next(batches))) for _ in range(40)]
    fresh_a, fresh_b = optimizer.init(params), optimizer.init(params)
    worst_uniform = 1.0
    for g in probe:
        big, fresh_a = update(g, fresh_a, params)
        small, fresh_b = update(rescale(g, 0.25), fresh_b, params)
        worst_uniform = min(worst_uniform, *(cos for cos, _ in compare(big, small).values()))
    g0 = rescale(probe[0], float(profile[0]) / float(optax.global_norm(probe[0])))
    first_a, _ = update(clip.update(g0, clip.init(params))[0], optimizer.init(params), params)
    first_b, _ = update(g0, optimizer.init(params), params)
    first = compare(first_a, first_b)
    print(f"uniform x0.25 rescale, 40 steps: worst update cosine {worst_uniform:.9f}")
    print(f"first step from fresh state at norm {profile[0]:.2f}: "
          + "  ".join(f"{k} cos {c:.9f} ratio {r:.6f}" for k, (c, r) in first.items()))

    # ── the replay ─────────────────────────────────────────────────────────────
    state_clipped, state_raw = optimizer.init(params), optimizer.init(params)
    clip_state = clip.init(params)
    rows = []
    for step in range(args.steps):
        g = grad_fn(params, jnp.asarray(next(batches)))
        g = rescale(g, float(profile[step]) / float(optax.global_norm(g)))
        clipped, clip_state = clip.update(g, clip_state)
        upd_clipped, state_clipped = update(clipped, state_clipped, params)
        upd_raw, state_raw = update(g, state_raw, params)
        cmp = compare(upd_clipped, upd_raw)
        rows.append({"step": step, "norm": float(profile[step]),
                     "clip_active": int(profile[step] > args.clip),
                     "muon_cos": cmp["muon"][0], "muon_ratio": cmp["muon"][1],
                     "adam_cos": cmp["adam"][0], "adam_ratio": cmp["adam"][1]})
        params = optax.apply_updates(params, upd_raw)
        if step % 100 == 0:
            print(f"step {step:5d}  norm {profile[step]:6.3f}  "
                  f"muon cos {cmp['muon'][0]:.5f} ratio {cmp['muon'][1]:.4f}  "
                  f"adam cos {cmp['adam'][0]:.5f} ratio {cmp['adam'][1]:.4f}", flush=True)

    print("\nband            clip on   muon cos (min/median)   muon ratio   adam cos (min/median)   adam ratio")
    for lo, hi in BANDS:
        band = [r for r in rows if lo <= r["step"] < hi]
        if not band:
            continue
        mc = [r["muon_cos"] for r in band]
        ac = [r["adam_cos"] for r in band]
        mr = float(np.median([r["muon_ratio"] for r in band]))
        ar = float(np.median([r["adam_ratio"] for r in band]))
        on = sum(r["clip_active"] for r in band) / len(band)
        print(f"{lo:>5}-{min(hi, args.steps):<6}  {on:7.1%}   {min(mc):.5f} / {np.median(mc):.5f}"
              f"      {mr:.4f}     {min(ac):.5f} / {np.median(ac):.5f}      {ar:.4f}")
        # One point per (quantity, band), all under one metric name: a spec judges a
        # single metric_key, and these are four different quantities.
        span = f"{lo}-{min(hi, args.steps)}"
        for quantity, value in (("muon_cos", float(np.median(mc))), ("muon_ratio", mr),
                                ("adam_cos", float(np.median(ac))), ("adam_ratio", ar)):
            emit(f"{quantity}@{span}", value=value)

    if args.csv_out:
        with args.csv_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-step rows: {args.csv_out}")
    print(json.dumps({"uniform_worst_cos": worst_uniform,
                      "first_step": {k: {"cos": c, "ratio": r} for k, (c, r) in first.items()}}))
    return 0 if math.isfinite(worst_uniform) else 1


if __name__ == "__main__":
    raise SystemExit(main())
