# Roadmap — from a finished plain base run onward

Status (2026-09-19): depth recurrence is retired
(`docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md`) and the live
model is the **plain transformer** (`MODEL_ARCH=plain`, 9 × 960, ~148M). The recipe it
trains with was measured this month by real-model matched pairs: AdamW's peak LR moved
from 1e-4 to 6e-4, and **Muon** beat even that by 1.32x in tokens with 380 MiB less memory
(`docs/findings/2026-09-18-muon-holds-1-32x-over-adamw-at-its-best-lr.md` and the chain it
links). What this project has never had is a **plain base model trained to completion** —
the thing every later change is compared against. That run is the next milestone, and
everything below is ordered around it.

## How this is tracked

This doc is the **narrative and the order** — the why, the stages, the gates. Per-item
state lives in **GitHub issues**; the rules that rank them (type tiers, unblockers first,
the idle card goes to `gpu` items, the base-run gate) live in `CLAUDE.md`, and
`python -m instruments.queue` computes the ranking. The order of work, per the owner:
first the repository (readable, lean, guarded by automation, rules that leave Claude no
ambiguity), then the tools (to code, research, investigate, and log), then the model.

## Stage 1 — before the base run (the `base-gate` label)

Everything that can be judged now **and only pays if the run has it** lands first
(`CLAUDE.md`, "Before a base run launches"). `gh issue list --label base-gate` is the
checklist; it has to be empty, or its leftovers waived by the owner by name. Grouped:

- **Correctness of what the model learns.** Model and trainer bugs whose fix changes the
  weights — the document separator masked as pad is the big one.
- **Telemetry.** Every quantity the run should log: per-block activations and residual
  scale, per-source gradient norms, the f16 margins, the optimizer's own knobs. A log
  the run did not write cannot be recovered from its checkpoints.
- **Speed and memory.** Every training-path optimization: freed memory decides the
  shape (a layer, batch 2, a longer window), speed decides how many tokens the run's
  days buy.
- **The recipe.** Adopt the measured recipe as the default; decide the shape (8 layers +
  batch 2, 9 layers, a 1024 window); decide how the run anneals and stops (WSD, to the
  owner's rule: val CE < 3.6 or 10 days, whichever first); and check that the winning
  recipe still wins on the mixture the long run actually ends on (65% code and math),
  since every recipe pair so far trained on the start of the ramp.
- **Surviving the week.** The run comes back by itself after a power cut; milestones
  do not fill the disk.

Measured on the card as matched pairs, each run under 24h and as many as a question
needs (the standing permission). The reference model is **SmolLM2-135M** on LAMBADA;
GPT-2-small's published number only calibrates the yardstick.

## Stage 2 — the base run

Launched through `make launch` against `experiments/base/specs/001-plain-base.toml`, once
the spec names the recipe Stage 1 chose. The owner authorizes it; nothing else about it
waits on a person. The CPU lane stays busy meanwhile with whatever does not gate it.

## Stage 3 — after the champion

A finished plain model gets a model card and becomes the champion; from then on an idea
is a short pair against its recipe, or a warm-start from its weights, never another base
run. First: tombstone the depth line and its apparatus (the refiner and reasoner stay
selectable only until then); extend the context (train short, then a brief phase at a
longer window); then the work that needs a capable base — inference speed, SFT, RL.

## Far future
- **RL post-training** — #29: preference / reasoning elicitation. On the AGI path but
  out of sequence — you RL on a base that already has capability to elicit; revisit
  only after pretraining + scaling produce a base worth aligning.

## Explicitly NOT planned
- bf16 *compute*: no tensor-core bf16 on Turing — f16 compute policy stands.
- Reviving the cross-window hunch with **soft auxiliary bonuses**: a bonus on the
  main loss for using an *optional* memory is what the gradient rejected twice
  (see finding); don't throw compute at a mechanism the data refused. The line:
  a *dedicated supervised target* that cannot be minimized without the slot is
  NOT this — that's the legitimate non-bypassable path the latent-scratchpad bet
  (#38) uses. Bonus = dead; grade = allowed.
- Chasing Chinchilla token counts as a target: it's a compute-allocation result,
  not a quality threshold; for a fixed model size it prescribes nothing.

## Graveyard
Killed ideas and closed post-mortems, with reasons, so they stay dead. New
tombstones land here — rule 5 of the working agreement sends every non-novel
result that killed or gates something to this section, one line each, linking
its PR.

- **SFT-on-plateau flip** — removed 2026-09-16 (#323), non-novel. The one time it
  fired, at opt step 5,055 of #157's 30,518, it killed that launch (CE 3.31 → 8.57,
  then the optimizer rebuild OOM'd); #157 was rewound past it and continued. The product is a base model; a fine-tune warm-starts
  from a stored champion in its own run. The plateau detector stays, as a report.
  Apparatus `instruments/sft_switch_smoke.py` and `tests/core/test_sft_autoflip_guard.py`
  removed with it, and the supervisor's plateau kill that guarded it.

- **Depth recurrence as the live architecture bet** — KILLED 2026-09-12. It works
  on sequential composition (`plan-a-depth-recurrence-works`, unretracted) and is
  *actively suppressed* on language: +0.003454 nats on code and nothing on prose or
  math, for 1.875× compute; the second refine pass costs 5.7 nats at 0.66B and
  nothing at 3.99B; the trained gate routes to 6 of 960 channels on prose. Post-norm
  bounds the activation blow-up (#235) and does **not** rescue it (#238, KILL). On
  the tree task the mechanism was designed for, depth 1 already solves every nesting
  level the setup can learn (#246). Full record:
  `docs/findings/2026-09-12-depth-recurrence-is-suppressed-not-exploited.md`.
  **Not killed:** the mechanism on tasks that need cumulative computation, and
  `MODEL_ARCH=refiner` itself, which is kept selectable the way `reasoner` is.
  `experiments/depth/eval_refiner_finetune.py`, the Stage-2 transfer probe this
  answered, removed 2026-09-16 (#321); `experiments/depth/playground.py`, the
  by-eye depth-sweep generator nothing ran any more, removed 2026-09-16 (#314).

- **Chunked (blockwise) attention on the live block** — KILLED 2026-08-03, apparatus
  removed 2026-09-15 (#291). Measured on the card against stock
  `dot_product_attention` at seq 512: +1.1% peak memory, +13.3% wall-clock, both
  gates failed — one score matrix is ~15 MB against a ~4.6 GB peak and remat already
  recomputes it. Its last reason to exist ("re-test if #23 widens the context") died
  when #23 closed wont-fix on 2026-07-24. Not novel (flash-style blocking is settled);
  the #66 PR is the record. `trm/model/attention.py`, the `chunked` branch, the
  `CHUNKED_ATTENTION` flag and `tests/core/test_chunked_attention.py` went with it.

### Post-mortems (non-novel; full record in PR history, guards in the tests)
- **double-applied attention scaling** (pre-June 2026): q was pre-scaled by
  1/√head_dim on top of `dot_product_attention`'s own scaling, and it survived weeks
  of training and log analysis — a scaled-twice softmax still trains, just worse.
  Only a reference-numerics test caught it. Guard:
  `tests/core/test_attention_numerics.py` (an independent naive recomputation of
  the score math, the one detector class that finds this kind of bug).
- **slot-future-leak** (2026-06-11, fixed f24f238): the v1 latent-scratchpad slots
  leaked future tokens into past predictions — a bidirectional summary exposed to
  causal decode positions, the textbook non-causal-path bug. Pre-fix "depth lowers
  CE" readings were leak bandwidth, not refinement (see the hunch-inert finding's
  pre/post contrast). Guards: both causality tests in `tests/core/test_model_invariants.py`.
- **cosine halting on the serial scratchpad — raw-latent signal** (#39, 2026-07-15,
  PR #96, closed unmerged; this line is the record): halting the write loop at the
  first slot with cosine(s_k, s_{k−1}) > τ cannot trade writes for accuracy. On an
  identity-tail split (oracle 3.25 writes against the fixed 4), 3 seeds: τ ∈ {0.5,
  0.7} do save (3.30 / 3.48 writes) but cost 0.14–0.17 accuracy (≈3–4σ_pooled), and
  every τ ≥ 0.98 holds accuracy while saving nothing (≥3.99 writes). Two crumbs
  worth keeping. (a) Converged and computing steps overlap badly in raw-latent
  cosine (0.59–0.65 ± 0.24 vs ≈0.01 ± 0.18); the leading *hypothesis* is that the
  slot-index embedding keeps even frozen-value latents apart — unmeasured, and the
  labels there also fire on coincidental residue collisions (~11% of "converged"
  steps), so treat both distributions as indicative, not as measured cause. (b) The
  same signal read on the **grade logits** separates cleanly (≈0.86 vs ≈−0.14) and
  was never run as a halting rule — that became the re-scoped #39, killed next.
- **cosine halting on the serial scratchpad — grade-logit signal** (#39, 2026-07-23,
  closed negative-result; the #39 PR carries the run, log `aux_39_protocol.log`):
  the space fix worked exactly as far as the diagnostic promised, and still died on
  the pre-registered bars. On `variable_chain_task` with exact k_eff labels
  (halt_off, K=4, m=7, dim 64, 10k steps, seeds {0,1,2}, full-depth ceiling
  0.9963 ± 0.0042): the gate separated (+0.942 ± 0.064 converged vs −0.005 ± 0.401
  computing, ≈3.3 pooled σ), and the ladder trades far better than raw latents ever
  did — τ=0.90 saves 0.60 of 4 writes for −0.019 accuracy where the latent rule
  cost 0.14–0.17. But no τ on the pre-registered grid satisfied all three bars:
  corr(halt step, k_eff) — the primary, per-instance bar — peaked at **0.789 < 0.8**
  (τ=0.80), and accuracy-within-2σ never co-occurred with writes ≤ 3.5 (τ=0.99:
  0.9952 acc but 3.82 writes). Reading: grade-logit cosine measures *something real
  but rate-shaped* — it fires at roughly the right frequency without being right
  about *which instance* is done, exactly the failure #123's corr bar exists to
  catch. Both scratchpad spaces are now dead; the structural objection stands — a
  write-once scratchpad has no iterated state for a fixed-point detector. The one
  untried home is the refiner's depth loop, where the state genuinely iterates:
  #140, with the transferable lessons (readout space, exact labels, gate-then-
  ladder) written into it. **Apparatus removed 2026-09-16 (#321)**, for this kill
  and #123's learned-halting kill together: `HaltingScratchpadNet`,
  `variable_chain_task`, the halt arms, the grade-logit gate and ladder and the
  `--halting` flag in `experiments/scratchpad/harness.py`, and their six tests in
  `tests/apparatus/test_scratchpad_harness.py`. The non-halting scratchpad line stays.
- **time-blind refiner — no step signal** (#86 third arm, 2026-07-10, PR #97, closed
  unmerged; this line is the record): pre-registered on #86 before any result and run
  as the third arm of the time-signal grid (statetrack, dim 96, seeds {0,1,2}).
  Killed on the extensibility bar — by variance, not by mean: seeds 0/1 extended past
  the trained depth as well as sinusoidal (d16 0.752 / 0.772), while seed 2
  deterministically collapsed during depth-8 *training* (0.277 / 0.238, reproduced
  twice, same init and data), inflating σ_pooled until the +0.5σ/+0.9σ reading failed
  the ≥2σ bar; the run declined to launder a 0.69-accuracy crater as noise. Reading:
  the state alone suffices for the computation — the explicit step signal is a
  *training stabilizer* for deep recurrence, not a computational necessity. Caveat:
  measured on PR #97's branch, whose table control clamped explicitly (the #122
  control on main lets overrun detonate); the sinusoidal verdict was independently
  re-measured and landed via #122 — this arm never was. **The #138 rematch ran
  2026-07-24** (per-pass grading on both arms, GPU, log `aux_138_protocol.log`):
  the collapse mode is cured — 3/3 seeds train clean at d8, including this crater
  seed — but the stabilized arm still loses trained depth 8 by −0.027 (**5.9σ**,
  parity everywhere shallower), so the pre-registered kill bar fires and
  **sinusoidal stays the production signal**; #86's none-first preference order
  never activates. The same grid found the arm *length-extends better* than
  sinusoidal (+2.3σ at d16), reversing #122's assignment under per-pass
  supervision — the extension behavior belongs to the (signal × supervision)
  pair. Full record:
  `docs/findings/2026-07-24-time-blind-rematch-collapse-cured-step-signal-holds-d8.md`.

### Killed in the #10 triage (with reasons, so they stay dead)
- **per-token halting / ACT**: collapses at small scale — documented dead-end.
- **MoE / mixture-of-slots**: MoE keeps all experts resident → spends VRAM (scarce on
  6GB) to save FLOPs (abundant) — backwards for this card; slots are tied to the dead
  cross-window arch.
- **logit softcapping**: redundant with the q/k RMSNorm already in place; Gemma-2
  itself dropped it. Marginal stability knob, not a capability lever.
- **diffusion LM**: a different paradigm, not an add-on — adopting it abandons the
  depth-recurrence bet for a second project. (Its useful piece, multi-token
  prediction, was salvaged as #27.)
- **learned / meta optimizers + reactive LR**: fragile, poor-ROI; reactive global LR
  also fights golden-run determinism. Frontier pretraining uses fixed schedules
  (cosine/WSD) + Adam's per-parameter adaptivity — a proper decay schedule covers the
  real intent. Killed.
- **multi-token prediction**: NOT a dead-end, but **scale-gated** — the literature
  (Gloeckle et al.; DeepSeek-V3 at 671B) shows the benefit emerges at ~3B+ and is
  weak/absent for small models, and its inference-speedup payoff isn't our bottleneck.
  Out of the roadmap at 79.6M; reconsider only if we ever pass ~1-3B.
