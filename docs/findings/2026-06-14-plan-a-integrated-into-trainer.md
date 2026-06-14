# Plan A (CausalRefiner) integrated into the production trainer — CPU + GPU smoke green

Status: integrated, smoke-validated, NOT yet run at scale
Date: 2026-06-14
Author: autonomous session (Matheus away; he authorized integration, smoke tests,
pausing the baseline, and pushing to main for later review)

## What this does

Wires the Plan A `CausalRefiner` — proven on toy tasks in
[2026-06-13-plan-a-depth-recurrence-works.md] — into the existing production trainer
so it can be trained at real scale (Dim=512, vocab 100352, seq 512) and compared
head-to-head against the UniversalReasoner baseline. The architecture is selected at
launch by an env var; nothing about the baseline path changed.

```bash
MODEL_ARCH=refiner venv/bin/python start_training.py --new-run   # Plan A
venv/bin/python start_training.py                                 # baseline (default)
```

## Design — a thin adapter, not a trainer rewrite

`plan_a_model.CausalRefiner` is kept pure and config-free so the *same* class runs in
the ablation harness and here (the proof tests what we'd ship). The seam is one new
file, `plan_a_trainer.RefinerForTraining`, which presents the exact interface the
trainer already drives — `__call__(tokens, max_steps, training, should_refresh) ->
ReasonerOutput` plus a `hunch_cache` buffer — so the grad step, validation probe, and
Orbax checkpoint plumbing work unchanged.

Plan A has no cross-window state, no forget gate, no slot-diversity term, so that
machinery degrades to **honest no-ops**, not faked work:
- `forget_cost` and `diversity_loss` are exactly `0.0`, so their loss schedules
  multiply zero and contribute nothing;
- the two training windows are independent (no carried state), so `should_refresh`
  is ignored — each window is a standalone causal LM prediction;
- `hunch_cache` is a vestigial `[1,1,dim]` zero buffer, never read, kept only so the
  trainer's `model.hunch_cache.value` bookkeeping writes stay valid;
- `max_steps` maps to refinement depth — the one dial Plan A uses (sampled per step
  in training, fixed at inference), same as the baseline's reasoning depth.

A refiner run has a different param tree, so it **must start fresh** (`--new-run`); it
cannot resume a reasoner checkpoint. Its checkpoints live in their own run dir, so the
two never collide.

## Param count — fair comparison

`REFINER_ENCODER_LAYERS=7` (default) → **77.65M params** vs the reasoner baseline's
**79.6M** (within 2.5%). `init_model_and_optimizer` now prints the live count for
whichever arch is selected, so the comparison is on the record in every run log.

## Validation

**CPU suite (full):** all pass, including 3 new integration tests
(`tests/test_plan_a_integration.py`): the adapter returns the right `ReasonerOutput`
with zero regularizers; the real `compute_grad_step` + `apply_grads` drive it with
finite, nonzero gradients and reduce the loss on a fixed batch; and the refiner's
param tree survives an Orbax save→restore round-trip bit-for-bit. The baseline path is
unchanged — its tests still pass.

**GPU smoke (real config, f16):** `tools/smoke_refiner_gpu.py`. CPU can't cover this —
CPU XLA can't lower the f16-with-f32-accumulation matmuls (config.py), so f16
numerical health and the 6GB VRAM fit only show up on the card. Result:
- f16 refine loop is **numerically healthy at every depth 1→8** — finite loss
  (~24 ≈ 2·ln(vocab), correct near-uniform init over two windows) and finite, nonzero
  grad norms. The deep unrolled loop does not under/overflow in f16.
- **Fits in 6GB** with the full optimizer state (Adam m+v + MultiSteps accumulator)
  resident, at the real 0.85 mem-fraction, worst-case depth 8.

### Bug this smoke caught (and fixed)

`CausalAttention` ran its q/k RMSNorms in f32 (for stability) but left `v` in the
compute dtype, so on GPU `dot_product_attention` got f32 q/k and f16 v and raised a
dtype-mismatch. Invisible on CPU (everything f32). Fixed by casting q/k back to the
compute dtype before attention — mirroring the baseline's `layers.py` pattern, a no-op
in f32 so the toy proof is unaffected. **This is exactly why the GPU smoke runs before
an overnight run, not after.**

## What is NOT done (the open question, unchanged)

This proves the integration is *correct and runnable*. It does **not** show Plan A
helps language modeling at 79.6M scale — that is still the open question from the toy
findings doc: web-text next-token prediction may not need the sequential composition
depth-recurrence buys. Answering it needs a real `MODEL_ARCH=refiner --new-run`
training run, with the current baseline as the control. That run is the next go/no-go
and is left for Matheus to green-light.

## Files

- `config.py` — `MODEL_ARCH`, `REFINER_ENCODER_LAYERS` (env-overridable).
- `plan_a_trainer.py` (new) — `RefinerForTraining` adapter.
- `trainer.py` — `init_model_and_optimizer` branches on `MODEL_ARCH`; prints param count.
- `plan_a_model.py` — f16 attention-dtype fix.
- `tests/test_plan_a_integration.py` (new) — 3 integration tests.
- `tools/smoke_refiner_gpu.py` (new) — real-config f16 GPU smoke.
