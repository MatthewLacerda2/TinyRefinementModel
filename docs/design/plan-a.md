# Plan A — causal within-window depth recurrence

Design doc. Implementation follows on the `feat/ablation-harness` branch, gated by
the test battery before any real run.

## Why

The cross-window hunch is inert (docs/findings/2026-06-13-cross-window-hunch-inert.md).
After the leak fix, the reasoning loop's only pathway was a compressed slot memory
carried to the *next* window, and the gradient killed it within ~750 steps. The
documented-working alternative — Universal Transformer / recurrent-depth — loops a
shared block **causally over the current positions** and feeds refinement back into
the prediction it refines. Plan A switches to that pathway.

## Core idea

Refine each position's representation by looping a shared transformer block K times
under a causal mask, then predict from the refined representation. Position t only
ever attends to positions ≤ t, on every iteration — so there is **no future-token
leak by construction**, and depth (K) directly affects the prediction for tokens it
is allowed to see (unlike the current arch, where the loop can't touch its own
window at all). K is sampled per micro-step in training (random depth, as today),
fixed at inference (MAX_STEPS_LIMIT).

## Architecture

```
embed → fixed encoder (2 layers, causal)
      → causal refinement loop  (shared block × K, causal self-attn, time-embed/step)
      → RMSNorm → LM head (tied embedding, f32 accumulation)
```

Refinement loop (replaces `_reasoning_loop`) — `jax.lax.scan` over K steps, carry is
`z` of shape [B, S, D]:
- `z_in = norm(z) + time_signal(time_embed[step])`
- `z_new = shared_block(z_in, self-attention, mask=pad_bias, q_pos=kv_pos=seq_pos, is_causal=True)`
- per-position update gate (kept, zero-bias init): `g = sigmoid(gate([z_new, z]))`,
  `z = g·z_new + (1−g)·z`
- diag per step: drift `‖z_new − z‖` (reuse temporal_drift logic).

**Removed:** the 32 shared slots and slot cross-attention; the InfoNCE slot-stability
/ diversity loss (no analogue on per-position states); `hunch_cache`, the hunch gate,
and any notion of refreshing carried state (cross-window memory, proven dead — Plan A
is within-window only); the slot-reading decoder cross-attention.

Until #105 the *trainer* still required all of that, so the adapter had to fake it:
a zero forget cost and a zero diversity loss for the schedules to multiply, and a
never-read `hunch_cache` for the loop's bookkeeping to write into. The trainer now
speaks a neutral contract (`lm_contract.py`) and the fakes are gone — Plan A
implements a forward pass and nothing else. What a window means for a model that
carries state between windows is expressed as `new_document`, a fact about the data;
Plan A ignores it because it genuinely carries nothing.

**Kept:** `RotaryAttention` with explicit causal mask (leak-correct since the scale
fix); random-depth sampling (`sample_reasoning_depth`); `time_embed`; the
f32-accumulation tied LM head; zero-init residual `down_proj` (keeps init loss ≈
ln(VOCAB)).

## Leak-freedom argument

Encoder causal; every refinement iteration is causal self-attention (q_pos ≥ kv_pos);
the LM head reads only the refined position-t state. By induction over iterations,
position t's representation depends only on input positions ≤ t at every depth, so
logits[t] cannot depend on tokens > t for any K. The causality invariant test must
confirm this empirically at depths 1, 2, 4, 8.

## Proof instrument (the point of the overnight work)

Fineweb perplexity is the wrong yardstick — recurrent-depth wins show on
reasoning/algorithmic tasks, not raw next-token CE. So Plan A is validated on the
tiny-config ablation harness:
- **Depth-needing toy tasks:** parity (cumulative XOR — needs sequential
  composition), modular addition, copy/reverse. Tiny vocab, seq ~64, dim 64–128,
  minutes per run.
- **Measure:** train tiny Plan A; compare held-out accuracy/CE at inference depth 1
  vs depth K. If depth K ≫ depth 1 on a depth-needing task while depth 1 plateaus,
  causal depth recurrence works — the claim the hunch failed to support.
- **Grokking watch:** these tasks show clean train→generalization transitions fast;
  depth recurrence is expected to grok where depth-1 cannot.

## Why the two arches don't share block code (the decision, written down once)

`layers.py` (reasoner blocks) and `plan_a_model.py` (the refiner's shared block)
implement the same SwiGLU transformer block twice. That duplication is
**deliberately frozen, not waiting to be shared** — and #105 is where the choice
got recorded instead of re-litigated on sight:

- The reasoner is now a *frozen control*. Its value is that a matched pair can be
  re-run against it, and the trained v1 control checkpoint can be revived exactly
  (#134). Refactoring its blocks — even into an identical-looking shared module —
  risks the one property it exists to have.
- So the rule is: the refiner's block is the one that evolves; the reasoner's is
  touched only to keep it *runnable*, never to make it prettier.

The cost of that freeze is worth stating plainly, because it is a trap for exactly
the comparison this repo cares about. The two round the SwiGLU hidden width
differently — reasoner up to a multiple of 256, refiner up to a multiple of 64:

| `LATENT_DIM` | reasoner hidden | refiner hidden |
|---|---|---|
| 512 | 1536 | 1408 |
| 960 (current) | 2560 | 2560 |

They agree at 960 and disagree at 512. So "both arches at the same `LATENT_DIM`"
is matched **by construction at some dims and only by coincidence at others** — at
512 the reasoner carries a wider MLP for free. Any cross-arch parameter-matched
claim must therefore quote *parameter counts*, not dim, and re-check them at the
dim it actually ran (`trainer._param_count` prints them at launch). Note this is
already the weaker kind of comparison — cross-model, observational, never causal
(CLAUDE.md); the matched one-variable pair is what carries a cause, and it varies
one knob within a single arch.

## Test gates (all green before any real-model run)

- causality invariant (no future leak) at depths 1, 2, 4, 8;
- `overfit_smoke`: drive a single batch to ~0 loss (the loop can learn);
- `init_loss` canary: CE ≈ ln(VOCAB) at step 0;
- golden run: regenerate (architecture changed; old trajectory void), re-arm the
  determinism guard on the new arch.

## Open questions (decide with ablation evidence, not now)

- Separate fixed-depth encoder + variable-depth loop, or one looped block from
  embedding to head (purer UT)? Default: 2-layer encoder + looped shared block.
- Per-position update gate: keep (ACT-adjacent, richer) or drop (pure UT loop)?
  Default keep, zero-bias init; ablate.
- Depth schedule: uniform random [1,K] (converged fine before) vs curriculum. Keep
  uniform.

## Risks

- A new arch written in a night carries residual bug risk past the tests
  (double-scale precedent) — morning sanity pass required.
- Positive tiny-task depth signal is directional; confirm at real scale before
  trusting transfer to fineweb.
