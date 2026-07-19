# Design — fixed-budget overwrite scratchpad (#63)

Status: toy-proof stage, pinned before code per the AUTONOMY design-doc-first rule.
Builds on the #38 win (`docs/design/serial-scratchpad.md`,
`docs/findings/2026-07-03-serial-scratchpad-beats-controls.md`) and is informed by
the graveyard: the `forget_head`/cross-window hunch died because forgetting was a
polite request the gradient could ignore
(`docs/findings/2026-06-13-cross-window-hunch-inert.md`).

## The bet, in one paragraph

The #38 scratchpad gives every sub-result its own slot forever (K slots for K
steps) — it never has to choose what to discard. This asks whether forgetting
can be *learned* rather than installed, by making it **mandatory by capacity**:
a memory of S < K slots that K sequential writes must share. Writing sub-result
k+1 physically requires overwriting something already there. No forget gate, no
forgetting loss — the desk is just too small, and gradient descent must discover
which slot to spare and which to let churn.

## Architecture — `BudgetScratchpadNet`

One class covers both phases; the slot budget `S` is the only knob that
changes between arms (matched-pair discipline). Reuses `Block`/`CrossBlock`
exactly as `ScratchpadNet` does (`scratchpad_harness.py`).

```
embed(tokens) -> causal encoder (2 x Block) -> h                     # tokens, as before
memory = mem_init                                                     # [S, dim], learned init

for k in 1..K:
    q_k     = step_index_embed(k)                                     # [1, dim] learned, step identity
    v_k     = write_block(q_k, concat(h, memory))                     # ONE shared cross-attn+MLP block
    grade_k = v_k @ W_readout                                         # CE(grade_k, r_k) -- non-bypassable, as in #38
    addr    = softmax(addr_head(concat(q_k, mean(memory))))           # [S] -- which slot(s) to overwrite
    memory  = (1 - addr) * memory + addr * v_k                        # soft in-place write, no append
```

- **S = 1**: `addr` is a softmax over a size-1 axis, i.e. identically 1 — full
  overwrite every step, no policy to learn, no free parameter does anything.
  This is pure "forgetting by capacity" with zero addressing degrees of
  freedom — the null architecture the phase-1 hypothesis needs.
- **S > 1**: `addr` is a real S-way softmax computed from the step's identity
  and the current memory — the model must learn, from step index and content
  alone, when to spare a slot and when to let it be overwritten. Nothing forces
  a particular policy; a bad address (spread evenly, or always evicting the one
  slot that matters) is just as reachable by gradient descent as a good one.
- The **per-step grade** on `v_k` (not on the post-address memory) is carried
  over unchanged from #38: whatever the addressing does afterward, step k's
  write itself is still supervised to equal `r_k`, so a collapse of the write
  path itself is visible exactly as it was in #38 (the RMT-forget-gate failure
  mode this design is built to avoid).

## Task reuse — no new generator needed

Both phases reuse the existing `affine_chain_task(K, m)` from `scratchpad_harness.py`
(`r_k = (r_{k-1}*a_k + b_k) mod m`, non-commutative, chance = 1/m). Only how the
**final answer** is read off the sub-results changes:

- **Phase 1** (`overwrite` arm): final = `r_K`, identical to #38's target.
- **Phase 2** (`budget1`/`budget2`/`unlimited` arms): final = `(r_1 + r_K) mod m`
  — a recall task that genuinely needs r_1 held until the last step, not just
  the running composition.

## Closing the invertibility shortcut (Phase 2 only)

`m` is prime, so every `a_k` is invertible mod m — the affine chain is
invertible. A readout with the tokens still in context could in principle
recompute r_1 from r_K by inverting steps K..2 *directly from the tokens*,
without ever consulting the memory. That would let a 1-slot arm "solve" the
recall task without any retention at all, defeating the point of phase 2.

Fix: for the recall variant only, the final-answer readout is **slots-only**
(`read_tokens=False`, the wiring #62 already proved and tested) — tokens are
architecturally unreachable from the answer, so any correct recall of r_1
can only mean r_1 is still sitting in memory. Phase 1 keeps `read_tokens=True`
(matching #38's serial arm exactly) since the existing `parallel`/`depthonly`
controls already establish that a tokens-visible readout cannot shortcut this
chain without genuine serial slot flow.

## Arms

| arm | class | S | task target | read_tokens |
|---|---|---|---|---|
| `overwrite` (phase 1) | `BudgetScratchpadNet` | 1 | r_K | True |
| `serial` (phase 1 ceiling, already exists, #38) | `ScratchpadNet` | K (append) | r_K | True |
| `budget1` (phase 2 control) | `BudgetScratchpadNet` | 1 | (r_1+r_K) mod m | False |
| `budget2` (phase 2 bet) | `BudgetScratchpadNet` | 2 | (r_1+r_K) mod m | False |
| `unlimited` (phase 2 ceiling) | `ScratchpadNet` | K (append) | (r_1+r_K) mod m | False |

## Proof gate (pre-registered)

**Phase 1.** K=4, m=7, dim 64, heads 4, enc 2, 2500 steps, batch 256, train
pool 32768 / held-out 4096, seeds {0,1,2}. **Win:** `overwrite` matches `serial`
within 2σ_pooled on held-out final accuracy (forgetting-by-overwrite carries the
chain with O(1) memory). **Kill:** overwrite trails serial by >2σ (the single
slot cannot even carry a chain it must fully replace each step).

**Phase 2.** K=5 (K > 2 so budget-2 must genuinely choose, not just hold
everything), m=7, same sizes/steps/seeds, `affine_chain_task` unchanged, final
target `(r_1+r_K) mod m`. **Win:** `budget2` beats `budget1` by ≥2σ_pooled AND
sits within [budget1, unlimited] approaching `unlimited`. **Kill:** `budget2`
sits at the `budget1` level (no retention learned), or Phase 1 fails outright
(overwrite can't carry a chain, so testing selective retention on top is moot).

## Files

| Piece | Where |
|---|---|
| `BudgetScratchpadNet` + arms | `scratchpad_harness.py` (extends the #38 harness, reuses `Block`/`CrossBlock`) |
| Wiring guards | `tests/test_scratchpad_harness.py` |
| Verdict | `docs/findings/` entries + PR closing #63 |

## What this stage does NOT claim

Toy-scale only, per the #38 precedent — this earns a findings entry either way,
not an LM-scale architecture change. A negative result here (either phase) is
the third recorded data point on how forgetting mechanisms fail or succeed in
this repo, which the working agreement (CLAUDE.md, rule 5) treats as load-bearing
regardless of which way it goes.
