# Design — supervised serial latent scratchpad (#38)

Status: toy-proof stage. This document pins the exact wiring before any code, per
the AUTONOMY design-doc-first rule. The bet and its history live in issue #38 and
`docs/ROADMAP.md`; this is the buildable specification.

## The bet, in one paragraph

Reasoning should unroll as a serial latent scratchpad on the critical path: an
*ordered* set of K sub-slots, each written exactly once, each **supervised to carry
sub-result k** — a grade the loss cannot dodge, not a bonus it can ignore. The two
prior slot designs died by bypass (the gradient starved a side memory the model
could route around; see findings 2026-06-11 and 2026-06-13). This design is
non-bypassable twice over: per-slot supervision (a loss term only satisfiable by
storing the sub-result) and write-order causality (slot k reads only tokens and
slots < k, so information cannot flow backward).

## Toy task — chained affine maps (the decomposable target)

**Why not the issue's literal "chained modular arithmetic sums":** a running sum
mod m is commutative — the final answer has an order-free shortcut (add
everything), so it cannot certify seriality. The chain must be non-commutative.

**Task `affine_chain(K, m)`** with m prime (default m=7), a ∈ [1, m), b ∈ [0, m):

```
input tokens:  a_1 b_1 a_2 b_2 ... a_K b_K        (seq = 2K)
r_0 = 0
r_k = (r_{k-1} * a_k + b_k) mod m                  (k = 1..K)
```

- **Sub-result targets:** r_1 .. r_K (one per slot).
- **Final target:** r_K, predicted at the last input position.
- Composition of affine maps mod m is non-commutative and has no constant-depth
  shortcut (the cumulative-product regime, same family as statetrack) — getting
  r_K right genuinely requires the chain, and the chain decomposes exactly into
  the K sub-results. Chance = 1/m.
- Token values a (nonzero) and b share the vocab {0..m-1}; the model never sees
  r_k as an input token.

## Architecture — three arms, one variable apart

All arms share the same base: `embed(tokens) → causal encoder (2 × Block)` from
`plan_a_model.py`, dim 64, heads 4. Vocab m. Sizes are toy-lane; the harness
parametrizes them.

### Arm S — serial scratchpad (the bet)

One **shared** write block (a `Block` used cross-attention-style: slot queries
attend to a context), K write steps, slot k written once at step k:

```
slots = []
for k in 1..K:
    q_k    = slot_init + slot_index_embed(k)               # [1, dim] learned
    ctx_k  = concat(encoder_states, slots[1..k-1])          # tokens + EARLIER slots only
    s_k    = write_block(q_k, ctx_k)                        # one cross-attn + MLP pass
    slots.append(s_k)                                       # frozen — never rewritten
```

- **Per-slot grade:** `slot_logits_k = s_k @ W_readout` (shared [dim, m] head);
  `CE(slot_logits_k, r_k)`. This is the non-bypassable supervision.
- **Final answer:** a readout query attends to `concat(encoder_states, all slots)`
  → `CE(answer_logits, r_K)`. The readout may exploit the slots (that is the
  mechanism working), but the *grade* on each slot is what forces the chain to
  exist.
- **Loss:** `CE_final + λ · mean_k CE_slot_k`, **λ = 1.0 fixed** — pre-registered,
  not tuned; tuning λ after seeing results is exactly the goalpost-moving the
  working agreement forbids.

### Arm P — parallel-slot control (kills "order did nothing")

Identical module, identical parameters, identical per-slot supervision. One
change: all K slots are written in a **single step**, every slot's context is
tokens only (`ctx_k = encoder_states` for all k — no slot sees any other slot).
Order and slot-to-slot flow are removed; everything else is held fixed.

### Arm D — depth-only control (kills "slots did nothing beyond depth")

`CausalRefiner` at depth K on the same tokens, final-answer supervision only —
Plan A as-is, no scratchpad, no auxiliary loss. Params differ from S/P (inherent
to this control); S vs P is the matched pair, S vs D is the is-it-just-depth check.

## Proof gate (pre-registered)

- **Protocol:** K = 4, m = 7, dim 64, heads 4, enc 2, 2500 steps, batch 256,
  train pool 32768 / held-out 4096 sequences, seeds {0, 1, 2}, all three arms —
  matched pairs (same seed ⇒ same pools).
- **Metric:** held-out final-answer accuracy. Noise floor σ = per-arm seed spread.
- **WIN** (the issue's bar): S beats **both** P and D by ≥ 2σ_pooled on final
  accuracy.
- **KILL:** (a) write-path collapse — S's per-slot accuracies sit at chance while
  final accuracy trains (the RMT forget-gate failure mode, made visible here by
  the grades), or (b) no ≥2σ gap over P (order added nothing).
- Anything between: inconclusive, recorded as such in `docs/findings/`, no claim.

## What this stage does NOT claim

Toy-scale only. A win here earns the *next* rung (harness-scale LM probes), not a
production run — that needs the user's go per AUTONOMY. Convergence halting (#39)
stays blocked until this gate returns a verdict.

## Files

| Piece | Where |
|---|---|
| Task + three arms + runner | `scratchpad_harness.py` (root, mirrors `ablation_harness.py`; reuses `Block`/`CausalRefiner` from `plan_a_model.py`) |
| Wiring/causality/grade tests | `tests/test_scratchpad_harness.py` |
| Verdict | `docs/findings/` entry + PR closing #38 |
