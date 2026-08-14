# Every base run this project ever started died of BFC allocator fragmentation, not of anything about the model

Status: confirmed (2026-08-14 — the live run cleared opt step 1540; see "Confirmation")
Date: 2026-08-14
Commit: 5455e4d  Run: runs/run_20260813_214725  Measured with: `python -m trm.runtime.supervisor --stop-step 30518 --run-dir runs/run_20260813_214725 --log runs/run_20260813_214725.log --issue 157 -- --checkpoint-path runs/run_20260813_214725/checkpoints`

## Setup

The 138.7M-param `CausalRefiner` at dim960 / 15 heads / depth 8, f16 compute, on the
6GB RTX 2060. Until 2026-08-13 every training launch used JAX's default preallocated
BFC arena at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` (5222MB), set in `trm/train/start.py`.
The finding came out of launching the #157 base run, which OOM'd three times in a row
before the cause was identified.

## Evidence

**The failing allocation is sized by the parameter tree, not the batch.** Two launches
at different batch sizes died asking for nearly the same amount:

| allocator | batch | died at | request |
|---|---|---|---|
| BFC | 2 | opt step 1 | 626.21 MiB |
| BFC | 1 | opt step ~12 | 596.39 MiB |
| cuda_async | 1 | — (step 1255 and running) | — |

138.7M params at f32 is 555MB; with padding that is the ~596MB the optimizer step asks
for contiguously, every step. Batch size moved the *deadline* and not the *wall*, which
is why halving it bought 12 steps instead of a working run.

**The card was not full — the free list was in pieces.** Both crashes printed a
fragmented BFC map, e.g.

```
*********xx******************************************************xx************__________*__________
```

and the decisive case is `runs/base_v1_20260628_004133.log`, which OOM'd trying to
allocate **720.0 KiB** on a 6GB card. A 0.7MB allocation cannot fail for want of
capacity. That is fragmentation, unambiguously.

**Widening the arena does not help; it relocates the failure.** At
`MEM_FRACTION=0.95` (5837MB) the OOM moves outside the arena entirely: the driver
cannot instantiate a CUDA command buffer, reporting **28 alive graphs**. Random-depth
training compiles one program per sampled depth (1..8) times the accumulate/apply
branches, so the driver-side graph cost here is unusually large. The arena starves the
driver or the driver starves the arena; there is no setting of that knob that fits.

**Every previous base run ended the same way.** Every base-run log in `runs/` ends in a
BFC `RESOURCE_EXHAUSTED` at the identical traceback (`trainer.py` → `float(loss)` →
`jax/_src/array.py:_value`):

| run | last opt step | tokens |
|---|---|---|
| run_20260628_184433 | 15 | 2.0M |
| run_20260719_042625 (base v1, refiner) | 1540 | 201.9M |
| run_20260720_012843 (base v1, control) | 1540 | 201.9M |

1540 × `TOKENS_PER_OPT_STEP` (131,072) = 201,850,880 — the "~200M tokens" that CLAUDE.md
records as where past runs died, and which the base-model bar reads as evidence the
model was ~50× undertrained and "never finished school." The undertraining was real;
its cause was never diagnosed, and it was an allocator.

**The fix is free.** `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` replaces the preallocated
arena with CUDA's async mempool, which does not fragment this way. Matched on the idle
card, same config:

| allocator | tok/s | s per opt step |
|---|---|---|
| BFC | 4,583 | 28.6 |
| cuda_async | 4,681 | 28.0 |

BFC's documented 17–22% advantage over the `platform` allocator (PERFORMANCE_PLAN, 2026-06-10)
does not show up in this workload, so nothing is traded away. Landed as the default in
`trm/train/start.py` (#162).

**The current run has cleared the first death point by ~80×.** `run_20260813_214725` is
at opt step 1255 with zero OOMs, through validation and repeated checkpoint saves, where
BFC died at step 12–15.

## Confirmation

`run_20260813_214725` cleared the pre-registered milestone. The kill condition written
into this entry was "dies at opt step 1540 like its predecessors"; it did not.

```
step    ce      val_ce
1530  4.3263
1535  4.3600
1540  4.3115  4.7422    <- run_20260719_042625 and run_20260720_012843 both died here
1545  4.2991
1560  4.2320
```

Zero `RESOURCE_EXHAUSTED` in the log at that point. The `val_ce` on row 1540 is the
tell: validation and the rolling checkpoint both fire every 64 opt steps, and
1536 = 64 × 24, so row 1540 is the first row *after* the heaviest memory event in the
loop — a full param-tree gather for the checkpoint plus a separately compiled
validation program. Both July runs died immediately after that cycle, which is the
churn most likely to push a fragmented arena over the edge. Under cuda_async the same
cycle passes without incident.

## Limitations

- Confirmed for the failure mode, not for the whole budget: the run has cleared 1540 of
  30,518 opt steps. Nothing here claims the remaining ~9 days are guaranteed, only that
  the specific allocator failure that ended every previous attempt no longer occurs.
- Fragmentation is a documented property of arena allocators in general; nothing here
  is novel about BFC. What is novel to us is the *attribution* — that this project's
  entire base-run history, and the premise of its base-model bar, rest on it. That is
  the part worth keeping.
- The 720 KiB failure is from a 2026-06-28 run at a config that is not the current one,
  so it evidences fragmentation but not this exact model's footprint.
- Not tested: whether cuda_async changes numerics or determinism. It should not — it
  allocates the same buffers in a different order — but the golden-run test only covers
  the CPU path, so this is asserted, not measured.
- `instruments/vram_headroom_smoke` did not and could not catch this: it runs under the
  `platform` allocator, which has no fragmentation, and samples `nvidia-smi` rather than
  reading a peak. See #161.
