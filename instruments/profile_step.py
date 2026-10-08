"""A profiler trace of the real trainer's micro-step: where the time goes, by name (#473).

Everything we knew about training time was inferred from wall-clock deltas (#411:
24.2 s per opt step in the trainer against 18.7 s for `bench_train_step`'s loop mode).
This runs the trainer's own `loop.train_loop` (data pipeline, loss scaler, grad guard,
logging) on a fresh model, lets --warmup micro-steps compile, records the next
--micro-steps under `jax.profiler`, and prints where each one's time went:

  - wall ms/micro-step, measured between consecutive `trm/data_get` span starts;
  - the device's busy and idle time inside that window;
  - host ms per named span (`loop.span`), and how much of it the device sat idle;
  - the Python around the two `nnx.jit` calls, dispatch excluded (the #474 gate).

The trace lands under runs/profile/<run_id>/trace (open perfetto_trace.json.gz in
ui.perfetto.dev), with the summary beside it in summary.txt:

  venv/bin/python -m instruments.profile_step [--warmup N] [--micro-steps N]

Needs DATA_ROOT (the real loader is part of what is measured). The window starts on a
fresh init, so it holds opt-step boundaries but not the rare ones: a log row every
LOG_REAL_STEPS opt steps, the probe and checkpoint every 64; those land in the
`(unnamed)` row when a window reaches them.
"""

import os

# Production's allocator (trm/train/start.py): the trace is of the trainer as launched.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import argparse
import glob
import queue

import jax

from trm.runtime.checkpoints import load_or_create_checkpoint
from trm.runtime.run_tracker import RunTracker
from trm.settings import CONFIG
from trm.train import loop, trainer

REPORTS = {
    "ms/micro-step": ("measured", "between consecutive trm/data_get span starts on the trainer thread, "
                                  "over the traced window; one run, mean with p50 and max"),
    "s/opt step": ("estimated", "mean ms/micro-step x ACCUMULATION_STEPS"),
    "device busy / idle": ("measured", "union of the device plane's events (kernels, copies) inside the window"),
    "host ms per span": ("measured", "each trm/ span's duration, clipped to the window, per micro-step"),
    "nnx.jit wrapper ms": ("measured", "grad_step and apply_grads spans minus the PjitFunction events inside them"),
}

# #411's readings, per opt step at batch 1 x 128 micro-steps: what the trace attributes.
REFERENCE_S_PER_OPT_STEP = {"trainer": 24.2, "bench loop": 18.7, "bench kernel": 15.7}
# The #474 gate: nnx.jit's own host time, both calls together, ms per micro-step.
NNX_JIT_GATE_MS = 5.0
# Device-plane lines the profiler derives from the kernel lines; counting them again
# would read a whole XLA module's span, gaps included, as busy.
DERIVED_LINES = {"XLA Modules", "XLA Ops", "Steps", "Framework Name Scope", "TensorFlow Name Scope",
                 "TensorFlow Ops", "Source code", "Launch Stats"}
DISPATCH = "PjitFunction"
BOUNDARY = "data_get"
JIT_SPANS = ("grad_step", "apply_grads")
# A loader that died leaves the trainer blocked on an empty queue forever.
QUEUE_TIMEOUT_S = 600


class TracedQueue:
    """The trainer's data queue, with the trace started and stopped from inside it.

    Get k feeds micro-step k. The trace starts at get `warmup - 1`, so the next get's
    span is the window's first boundary, and ends the loop at get `warmup +
    micro_steps + 1`, after the last boundary's span has closed."""

    def __init__(self, inner, warmup, micro_steps, trace_dir, settle):
        self.inner, self.trace_dir, self.settle = inner, trace_dir, settle
        self.start_at, self.stop_at = warmup - 1, warmup + micro_steps + 1
        self.served = 0

    def get(self):
        k, self.served = self.served, self.served + 1
        if k == self.start_at:
            self.settle()
            jax.profiler.start_trace(self.trace_dir, create_perfetto_trace=True)
        if k == self.stop_at:
            self.settle()
            jax.profiler.stop_trace()
            return None, None, None
        try:
            return self.inner.get(timeout=QUEUE_TIMEOUT_S)
        except queue.Empty:
            raise SystemExit(f"no batch in {QUEUE_TIMEOUT_S}s: the loader thread died (see above)") from None


def merged(intervals):
    """Sorted, non-overlapping cover of (start, end) intervals."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def covered(cover, lo, hi):
    """How much of [lo, hi) a merged cover holds."""
    return sum(max(0, min(end, hi) - max(start, lo)) for start, end in cover)


def read_trace(path, prefix=loop.SPAN_PREFIX):
    """(spans, dispatch, device, device_lines) from an .xplane.pb, times in ns.

    spans: (name, start, end) of the trainer's named spans, prefix stripped.
    dispatch: (start, end) of the XLA calls on the spans' own thread; another thread's
    (the loader's) would otherwise be subtracted from spans it never ran inside."""
    from jax.profiler import ProfileData
    spans, dispatch, device, device_lines = [], [], [], []
    for plane in ProfileData.from_file(path).planes:
        for line in plane.lines:
            events = [(e.name, e.start_ns, e.end_ns) for e in line.events]
            if plane.name.startswith("/device:"):
                if line.name not in DERIVED_LINES and events:
                    device += [(s, e) for _, s, e in events]
                    device_lines.append(line.name)
            elif any(name.startswith(prefix) for name, _, _ in events):
                spans += [(name[len(prefix):], s, e) for name, s, e in events if name.startswith(prefix)]
                dispatch += [(s, e) for name, s, e in events if name.startswith(DISPATCH)]
    return spans, dispatch, device, device_lines


def summarize(spans, dispatch, device):
    """Per-micro-step accounting of the window between the first and last boundary."""
    starts = sorted(s for name, s, _ in spans if name == BOUNDARY)
    if len(starts) < 2:
        raise SystemExit(f"{len(starts)} trm/{BOUNDARY} spans in the trace: no whole micro-step to read")
    lo, hi, n = starts[0], starts[-1], len(starts) - 1
    walls = sorted(b - a for a, b in zip(starts, starts[1:]))
    busy = merged(device)
    calls = merged(dispatch)
    per_span = {}
    for name, s, e in spans:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        total, idle, wrapper = per_span.get(name, (0, 0, 0))
        per_span[name] = (total + e - s, idle + (e - s) - covered(busy, s, e),
                          wrapper + (e - s) - covered(calls, s, e))
    return {"micro_steps": n, "wall": hi - lo, "p50": walls[n // 2], "max": walls[-1],
            "busy": covered(busy, lo, hi), "spans": per_span}


def format_summary(summary, device_lines, trace_path):
    n, wall = summary["micro_steps"], summary["wall"]
    ms = lambda ns: ns / n / 1e6  # noqa: E731 - ns over the window -> ms per micro-step
    mean = ms(wall)
    reference = ", ".join(f"{k} {v}" for k, v in REFERENCE_S_PER_OPT_STEP.items())
    lines = [
        f"profile: {n} micro-steps | batch {CONFIG.BATCH_SIZE} x {CONFIG.ACCUMULATION_STEPS} "
        f"micro-steps per opt step",
        f"trace:   {trace_path}",
        f"wall     {mean:8.1f} ms/micro-step (p50 {summary['p50'] / 1e6:.1f}, max {summary['max'] / 1e6:.1f})"
        f" -> {mean * CONFIG.ACCUMULATION_STEPS / 1000:.2f} s/opt step (#411, batch 1: {reference})",
    ]
    if device_lines:
        busy = ms(summary["busy"])
        lines.append(f"device   busy {busy:.1f} | idle {mean - busy:.1f} ms/micro-step "
                     f"({1 - busy / mean:.0%} idle) over lines: {', '.join(sorted(device_lines))}")
    else:
        lines.append("device   no device events in the trace (a CPU backend, or CUPTI did not load): "
                     "busy/idle not measured, the idle column below reads the whole span")
    lines.append(f"{'host span':22} {'ms/micro-step':>14} {'% wall':>7} {'device idle ms':>15}")
    named = 0
    for name, (total, idle, _) in sorted(summary["spans"].items(), key=lambda kv: -kv[1][0]):
        named += total
        lines.append(f"  {name:20} {ms(total):14.2f} {total / wall:7.1%} {ms(idle):15.2f}")
    lines.append(f"  {'(unnamed)':20} {ms(wall - named):14.2f} {(wall - named) / wall:7.1%}")
    wrapper = {name: summary["spans"].get(name, (0, 0, 0))[2] for name in JIT_SPANS}
    lines.append("nnx.jit wrapper, span minus the XLA call inside it: "
                 + " + ".join(f"{name} {ms(v):.2f}" for name, v in wrapper.items())
                 + f" = {ms(sum(wrapper.values())):.2f} ms/micro-step (#474 gate: {NNX_JIT_GATE_MS:g})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Warmup crosses one opt step, so both apply_grads paths compile before the trace.
    parser.add_argument("--warmup", type=int, default=CONFIG.ACCUMULATION_STEPS + 2,
                        help="untraced micro-steps, compile included (> ACCUMULATION_STEPS)")
    parser.add_argument("--micro-steps", type=int, default=CONFIG.ACCUMULATION_STEPS,
                        help="micro-steps measured; the default window holds one optimizer update")
    parser.add_argument("--out", default="runs/profile", help="where the profile's run folder goes")
    args = parser.parse_args()
    if not trainer.DATA_ROOT:
        raise SystemExit("DATA_ROOT is not set: the real loader is part of what this measures")
    if args.warmup < 2 or args.micro_steps < 1:
        raise SystemExit("need --warmup >= 2 and --micro-steps >= 1")

    run_tracker = RunTracker(CONFIG, runs_root=args.out)
    run_tracker.start_session()
    model, optimizer = trainer.init_model_and_optimizer(CONFIG)
    mngr, best_mngr, monitor, start_step = load_or_create_checkpoint(
        CONFIG, model, optimizer, os.path.abspath(os.path.join(run_tracker.run_dir, "checkpoints")), force_new_run=True)
    trace_dir = os.path.join(run_tracker.run_dir, "trace")
    data_queue = TracedQueue(
        trainer.setup_data_pipeline(CONFIG, start_step), args.warmup, args.micro_steps, trace_dir,
        settle=lambda: jax.block_until_ready(jax.live_arrays()))
    loop.train_loop(CONFIG, model, optimizer, data_queue, mngr, best_mngr, monitor, start_step, run_tracker,
                    trainer.DATA_ROOT)
    if data_queue.served <= data_queue.stop_at:
        raise SystemExit("the loop ended before the traced window closed")

    xplane = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    perfetto = glob.glob(os.path.join(trace_dir, "**", "perfetto_trace.json.gz"), recursive=True)
    spans, dispatch, device, device_lines = read_trace(xplane[0])
    text = format_summary(summarize(spans, dispatch, device), device_lines, (perfetto or xplane)[0])
    with open(os.path.join(run_tracker.run_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
