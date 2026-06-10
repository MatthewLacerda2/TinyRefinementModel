# Performance & VRAM Plan — Algorithmic Optimizations

Date: 2026-06-09. Scope: algorithmic/runtime improvements only — the model architecture
(dims, blocks, losses, curriculum) is explicitly out of scope. Hardware: RTX 2060, 6 GB
VRAM, Turing (no bf16, f16 compute policy), BATCH_SIZE=1 with 128-step accumulation.

Ground truth from the Phase 5 derived stats: 79.6M params; static memory is
~1.2 GB (304 MB f32 weights + 608 MB AdamW moments + 304 MB grads) **plus** the
`optax.MultiSteps` gradient-accumulation buffer (another param-sized f32 tree, ~304 MB).
So ~1.5 GB is static and ~4.5 GB is available for activations and workspace.

## Rule zero: measure before and after every change

None of the hypotheses below get merged on faith. Build a tiny benchmark harness first
(`tools/bench_train_step.py`): synthetic batches, ~20 warmup steps, then N timed
micro-steps with `jax.block_until_ready`, reporting steps/sec and peak VRAM via
`jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`. Each candidate lands as
its own commit with the before/after numbers in the commit message. Changes that should
be math-identical are additionally verified with the bit-identical-loss smoke (the same
one used through Phases 2–5); changes that alter execution order but not semantics get
a short real-data convergence sanity run.

## Measured baseline (2026-06-09, user-reported + metrics.csv analysis)

- **~6.03–6.5 s per optimizer step** (the 128 accumulated micro-steps included) on the
  pre-refactor code — i.e. **~47–51 ms per micro-step**, each processing 2×512 = 1024
  tokens → roughly **20–21k tokens/sec**.
- **~3.8 GB steady VRAM** observed in nvtop. With ~1.5 GB static (weights + AdamW +
  grads + MultiSteps accumulator), dynamic memory is ~2.3 GB. Headroom to the 6 GB
  ceiling: ~2.2 GB. Note: nvtop readings are only meaningful under the current
  `platform` allocator; if P2 switches to BFC (which preallocates), use
  `memory_stats()["peak_bytes_in_use"]` instead.
- **The per-step time was consistent across the whole run** — which spanned reasoning
  depth 1, 2, and 4 phases. Per-step time not jumping at the depth boundaries is strong
  evidence that the reasoning scan is *not* the cost center. That matches the FLOPs:
  each reasoning step runs 8 shared-block iterations over only 32 slot tokens, a few
  percent of the encoder+decoder work over 512 tokens. The cost centers are the
  depth-independent parts: encoder/decoder stacks, the LM head, remat recompute, and
  host-sync overhead.
- **Rough roofline sanity check:** per micro-step ≈ 0.5–0.7 TFLOP (the LM head alone is
  ~half of it: 2·1024·512·100352 ≈ 105 GFLOP forward), so 48 ms implies ~10–14 TFLOPs
  sustained — maybe ~20–25% of the 2060's tensor-core peak. Interpretation: there is
  likely 2–3x headroom, not 10x; the GPU is not pathologically idle, so FLOP-level
  levers (less remat recompute, f16 tensor-core head) matter as much as sync removal.
  The harness will apportion this precisely.

## Candidates, ranked by expected impact

### P1 — Fuse the train step and stop syncing every micro-step (wall clock, big)
**Today:** every micro-step does two jitted dispatches (`compute_grad_step`, then
`apply_grads`) and then immediately calls `float(loss)`, `float(grad_norm)`,
`float(out.halt_diag[...])` — each `float()` blocks the host until the GPU finishes.
With BATCH_SIZE=1 the per-step compute is small, so this host-device ping-pong is
proportionally enormous; the GPU idles between steps. `jnp.array(step)` also makes a
tiny host→device transfer every step.
**Plan:** one jitted `train_step(model, optimizer, batch, step, doc_boundary)` that
computes grads, checks finiteness **on device** (skip the update via `lax.cond`,
increment an on-device non-finite streak counter, clear the hunch cache in-branch),
applies the update, and accumulates the running metric sums (loss, token loss, grad
norm) in on-device state. The Python loop just dispatches; it pulls metrics to the host
once per log interval (every 640 micro-steps) and checks the streak counter there.
**Preserves:** identical loss math, identical skip semantics; the abort-on-streak just
becomes granular to the log interval instead of the exact step.
**Side task:** `t_compute` currently measures "until the float() sync" — replace with
explicit `block_until_ready` timing at the log boundary so the number means something.
**Expectation revised by the baseline:** the roofline estimate suggests the GPU is
~20–25% utilized, not idle — so P1's win is real but probably tens of percent, not
multiples. Measure before assuming it's the headline.

### P2 — Allocator experiment (wall clock, possibly big, one line)
**Today:** `XLA_PYTHON_CLIENT_ALLOCATOR=platform` performs a synchronous cudaMalloc/free
per buffer — it is the slowest allocator JAX offers, usually chosen only to coexist with
a display using the same GPU.
**Plan:** benchmark the default BFC allocator with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
(and optionally `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`) against the current setting. If
BFC is stable alongside the desktop session, keep it. Zero code risk; pure measurement.

### P3 — Tame the LM-head memory (VRAM, biggest single lever)
**Today:** `seq_norm(z_seq_out) @ embed.T` materializes f32 logits of shape
[1, 512, 100352] ≈ **205 MB**, and the CE backward keeps logits-sized residuals — the
head likely dominates peak activation memory by an order of magnitude over everything
else.
**Plan, in order of preference:**
1. Chunked CE: compute logits + cross-entropy over sequence chunks (e.g. 64–128
   positions) inside a `jax.checkpoint`/scan, accumulating the masked CE sum. Peak drops
   from ~205 MB to ~25–50 MB at the cost of recomputing the head matmul in the backward.
2. And/or cast the head matmul to COMPUTE_DTYPE (f16 logits, f32 log-softmax per chunk)
   — halves what remains; verify CE stability in f16 against the f32 reference on a few
   hundred real batches before trusting it.
**Why it matters even at batch 1:** freed VRAM is what pays for P5 (less remat) and any
future batching — it converts directly into speed elsewhere.

### P4 — Buffer donation (VRAM, medium)
**Today:** `apply_grads` allocates new optimizer/param buffers while the old ones are
still alive; the grads tree (304 MB) also stays alive across the two dispatches.
**Plan:** with the fused P1 step, donate the optimizer state and grads
(`donate_argnames` on the jit) so updates happen in place. Needs care with nnx state
threading — investigate what flax nnx's jit wrapper supports before committing to it.

### P5 — Remat tuning: stop paying for double checkpointing (wall clock, medium)
**Today:** the reasoning stack runs per-block remat *inside* the scan-level
`jax.checkpoint(scan_step)` — the backward recomputes recomputations. This was the safe
choice for 6 GB and is what all runs trained with, but it was never measured.
**Plan:** three benchmark configs: (a) current; (b) scan-level checkpoint only — drop
per-block remat in the reasoning stack; (c) after P3 lands, additionally try disabling
per-block remat in the encoder/decoder stacks. Adopt the fastest config that stays under
~5 GB peak. This is the experiment the old stale comment in `layers.py` was gesturing at.
**Priority revised by the baseline:** the depth-independence of the user's per-step
timing shows the reasoning scan is cheap — config (b) will barely matter. Config (c),
encoder/decoder remat, is where the recompute FLOPs actually are (remat makes backward
≈ fwd + recompute + 2·fwd ≈ 4× forward; dropping it where memory allows saves ~25% of
total FLOPs). With ~2.2 GB of measured headroom, (c) may fit even before P3.

### P6 — Attention backend probe (wall clock, small-to-medium, cheap to test)
**Today:** `jax.nn.dot_product_attention` with default implementation; on Turing the
fused cudnn flash path is generally unsupported (it targets Ampere+), so XLA likely
materializes [1, 16, 512, 544] score tensors (~9 MB each, f16) — fine for memory,
suboptimal for speed.
**Plan:** try `implementation="cudnn"` and measure; expect it to be rejected on SM75,
in which case document that and move on. Ten-minute experiment.

### P7 — True batching across independent streams (throughput, biggest reward, gated)
**Today:** BATCH_SIZE=1 almost certainly leaves the 2060 underutilized at 512×512 dims.
The model *mostly* supports B>1 already (hunch cache is [B, S, D], gates are
per-element), with one semantic blocker: `should_refresh = jnp.any(doc_boundary)` is a
scalar — one stream hitting a document boundary would wipe **all** streams' hunches.
**Plan:** replace the scalar `lax.cond` refresh with per-element `jnp.where` selection
(this is arguably a correctness improvement, not just an enabler), have the data loader
serve N independent document streams, and benchmark B=2/B=4 against the VRAM freed by
P3/P5. Tokens/sec could plausibly 2–3x.
**Gate:** this changes training semantics (effective batch size per optimizer step, and
per-stream hunch continuity), so it lands last, behind a real convergence comparison on
a few thousand opt steps — not just the bench harness.
**Memory math from the baseline:** B=2 roughly doubles the ~2.3 GB dynamic portion →
~6.1 GB total, over the ceiling. P3 (and possibly P5's memory choices) are therefore
*prerequisites* for P7, not just nice-to-haves.

## Explicitly out of scope
- KV-cache / incremental decoding for inference — already logged in PLAN.md as designed
  future work; it does not affect training throughput.
- Anything that changes the loss, the curriculum, or the architecture — that is the
  next stage's discussion, informed by metrics.csv, and must not be entangled with
  performance commits.

## Execution order

1. Benchmark harness + baseline numbers (also record the current run's steps/sec for
   reference).
2. P2 (allocator) — pure measurement, may reset the baseline.
3. P1 (fused step, deferred sync) — the expected headline win.
4. P3 (head memory), then P5 (remat tuning) which spends the freed memory.
5. P4 (donation) and P6 (cudnn probe) opportunistically alongside.
6. P7 (batching) last, behind its convergence gate.

## Results log

**2026-06-10 — baseline correction.** The user-reported 6.03–6.5 s/opt-step was a timing
artifact: the old train loop stopped `t_compute`'s clock before the `float(loss)` sync,
so it measured async dispatch (~9.5 ms × 640 micro-steps ≈ 6.1 s per log window), not GPU
work. Ground truth from `run_metadata.json` durations vs CSV steps: **73.1 h for 4815 opt
steps = 54.7 s/opt-step = 427 ms/micro-step** (~2.4k tok/s, ~3% of tensor-core peak).
The roofline section above, which trusted the 48 ms figure, overestimated utilization by
~8x — real headroom is large. (Phase 3's reordering of the floats incidentally already
fixed the `t_compute` measurement; `calculate_tokens` in plot_history was also
under-reporting trained tokens by the 128x accumulation factor — fixed.)

**2026-06-10 — bench matrix** (tools/bench_train_step.py, depth 8, 60 timed steps,
kernel mode unless noted; RTX 2060, GPU otherwise idle — display runs on the iGPU):

| config | ms/micro-step | opt-step | peak VRAM |
|---|---|---|---|
| platform alloc, full remat (production config) | 388 (loop: 388) | 49.7 s | n/a (no stats) |
| BFC preallocated (MEM_FRACTION=0.85), full remat | 322 | 41.3 s | 3555 MB |
| BFC, no enc/dec remat | 314 | 40.2 s | 3573 MB |
| BFC, no reasoning remat | 271 | 34.7 s | 3555 MB |
| **BFC, no per-block remat at all (adopted)** | **264** | **33.8 s** | **3573 MB** |

Decisions taken:
- **P2 adopted:** `start_training.py` now sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`
  (BFC, preallocated) instead of the platform allocator. Note: BFC *without*
  preallocation fragments and OOMs mid-run — preallocation is required.
- **P5 adopted:** `use_remat=False` for all three stacks. Per-block remat saved no
  measurable VRAM (the scan-level checkpoint already bounds the reasoning loop) and
  cost ~18%. The pre-Phase-2 comment warning about double checkpointing was correct.
- **P1 demoted to cleanup:** loop mode == kernel mode (388.3 vs 389.3 ms) — the loop is
  entirely GPU-bound; per-step host syncs cost nothing today. The fused step remains
  worthwhile only as code simplification, not as a performance item.
- **P6 dead:** cudnn attention rejects the broadcastable [B,1,1,KV] bias shape (and
  Turing flash support is doubtful anyway). Would need bias materialization to even
  probe further; not worth it.
- Net so far: 388 → 264 ms/micro-step (**−32%**, 49.7 s → 33.8 s per opt step) with
  no math changes (bit-identical loss smoke green) and unchanged peak VRAM.

**Next hypothesis (promoted to top of queue): the LM head runs in f32.**
`seq_norm(z) @ embed.T` multiplies f32 × f32 — no tensor cores, and the 2060's f32
throughput is ~6.5 TFLOPs vs ~26-50 f16. The head is ~55% of model FLOPs
(2·1024·512·100352 ≈ 105 GFLOP fwd per micro-step across both segments), so an
f16-input matmul with f32 accumulation (`preferred_element_type`) could plausibly cut
another 25-40% of step time *and* is the P3 memory lever. Numerics-affecting → needs
the CE-delta verification described in P3 before adoption.

## Acceptance criteria
- Every merged change has before/after steps/sec and peak-VRAM numbers in its commit.
- Math-preserving changes keep the bit-identical-loss smoke green; the rest pass a
  short-run convergence sanity check.
- Peak VRAM stays under ~5 GB so the desktop session does not OOM the trainer.
- `pytest` and `ruff check` stay green throughout.
