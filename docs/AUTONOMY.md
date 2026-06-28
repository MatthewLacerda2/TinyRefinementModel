# Autonomous Development — harness, guardrails, oversight

How Claude develops this model autonomously along the roadmap. The goal is to
remove the human from the inner loop entirely — read an issue, write the code,
test it, run it, judge the result, document it, ship it, pick the next — while
keeping the human as the *source of direction* and the *one who can stop it*, not
as a per-step approver.

Note on framing: this is not self-modification — the model running Claude is
fixed. It's Claude autonomously *developing the TinyRefinementModel* along the
roadmap. Referenced docs: `docs/ROADMAP.md`, `docs/findings/`, and `CLAUDE.md`
(the research doctrine this charter operationalizes).

## The decision (2026-06-27): fully unattended

The earlier charter kept a human go/no-go gate at every expensive run. That gate
is **removed**. The end-state is a self-driving loop:

> read an issue → write the code → run the test battery → run smoke + ablation →
> (if the idea earns it) launch a training run → compare against the previous
> stored model → write the finding → open **and merge** the PR → next issue.

This is a deliberate shift, and an honest one only because the *reason* the human
gate existed is being relocated, not discarded. The gate was there because
Claude's documented weakness is fluent confidence on a known-hard idea, flagged
too late (the cross-window-hunch night). Removing a human reviewer is safe only
when the judgment that reviewer supplied is **encoded into gates the loop cannot
talk itself past**. Until those gates are built and green, the loop does not run
unattended — the prior gated mode (kept at the bottom of this doc) still governs.

## Two sessions

- **The autonomous session** runs the loop above.
- **The human session** is for setting direction (filing/prioritising issues,
  proposing ideas, nudging), and — the load-bearing power — **stopping the loop,
  reclaiming the GPU, or shutting the box down** at any time. The human is the
  arbiter of *what to pursue* and the *kill switch*; the loop is the arbiter of
  *whether a given change earned its merge*, by the encoded criteria below.

## The safety chain — what must be green before the human gate comes off

These exist precisely so the loop cannot manufacture a false win or merge a
confidently-wrong change. The fully-unattended mode activates only when all of
them are in place; until then, gated mode governs.

1. **Noise floor — #17.** Seed-variance measured first, so "it improved" is
   distinguishable from luck. This is the core anti-false-win guard, and an
   unattended loop running idea after idea is *more* exposed to chance wins than a
   human at the wheel, not less. Without it, every comparison the loop makes is
   uninterpretable. Non-negotiable, and first.
2. **A stored baseline to compare against — #16 (v1).** The matched
   refiner-vs-baseline base run: the base-model-bar validity check *and* the first
   champion in the registry. "Compare against the previous model" has no referent
   until this exists.
3. **Docker repro-pin — #44.** Freezes each registered model's runnable
   environment so any past version reloads and reruns despite env drift — the
   "boot an old version to A/B it" capability the comparison step depends on.
4. **CI hardened into a merge guardrail — #45.** Branch protection on `main` plus
   required checks (golden-run determinism, a fast smoke, reference-numerics) so
   **nothing red ever merges**. This is what makes self-merge a gate, not a rubber
   stamp.
5. **code-review subagent — #46.** A standing reviewer that runs `/code-review`
   over the loop's own diff with fresh context and an adversarial read — the
   replacement for the human eye that self-merge removes. A "block" verdict stops
   the merge.

## The encoded judgment — how a result earns its merge (from CLAUDE.md)

The loop applies these mechanically; they are the doctrine made executable. A
change merges only when it clears every one that applies:

- **Clears the noise floor.** A delta counts only if it beats its matched control
  by the pre-registered margin against #17's measured spread. A single-seed number
  is not a result; report the spread.
- **One variable, matched pair.** Same seed, same data order, one knob. An
  unmatched comparison is discarded, not merged.
- **Kill-criterion pre-registered.** Every experiment issue states, before the
  run, the threshold that keeps or drops the idea (phrased against the noise
  floor). The loop cannot move the goalpost after seeing the number — the
  criterion is read from the issue, not re-decided.
- **Strong baseline.** No architecture verdict is read against a base that hasn't
  itself been trained properly (the base-model bar). Until v1 hits ~GPT-2-small,
  ablations are "stirring the mush" and are not merge-worthy claims.
- **Every result recorded.** Win or lose, the run emits a `findings/` entry. A
  recorded negative is the moat against re-trying a dead end.

When a result is **genuinely ambiguous against the noise floor**, the loop's
correct action is not to merge-or-revert on a coin flip — it records the
ambiguity in the finding, leaves the change unmerged, and moves on. Ambiguity is
a result, not a tie to be broken by enthusiasm.

## The ablation harness (the inner instrument)

The thing that makes the loop fast — most ideas die or rank here in CPU-minutes,
before any GPU run:

- **Tiny configs**: dim 64–128, 1–2 layers, seq ~64, tiny vocab — a run is
  minutes. The unit of fast iteration.
- **Depth-sensitive toy tasks**: modular arithmetic, parity, copy/sort, and the
  decompose-able tasks for the latent-scratchpad bet (#38) — where depth/
  structure genuinely has to do work and grokking shows fast. This is also the
  correct *proof instrument* (it sidesteps "fineweb perplexity is the wrong
  yardstick"). Code-completion is a secondary, slower, noisier realism check, not
  a fast grokking signal.
- **Ablation runner**: config-delta → fixed small budget → metric → Δ-vs-control
  → above-noise-floor Y/N → structured row → draft finding. One command.
- **Results log**: structured and comparable; `findings/` stays the
  human-readable, publishable layer.

Ablation results stay **directional** — best for cheaply killing bad ideas and
ranking options. A positive tiny-model result is promoted to a real run, not
treated as the verdict itself.

## Resilience — unattended runs fail loud, not silent (#40)

The loop runs across context resets, the nightly kernel lockup, and the single
serial GPU. So it needs:

- **A watchdog** on long runs → heartbeat to a pinned status issue; stop-and-flag
  on stalled loss, GPU-driver drop, or low disk headroom (all documented
  hazards). The motivating failure: a detached tokenization job stalled three days
  unnoticed because no tracked artifact showed liveness.
- **Resource guards in code, not prose**: disk-headroom precheck, GPU serial-queue
  lock (one job at a time), a compute-budget cap.
- **Resumability**: state visible enough that a fresh session knows exactly where
  the loop left off and continues without re-deriving context.

## Gated mode — in force until the safety chain is green

Until #17, #16, #44, #45, and #46 are done, the loop is **not** unattended. In
the meantime the standing envelope is: build/refactor code and run the CPU test
suite freely on a branch; run cheap ablations and smokes unsupervised; pause the
live baseline to grab the GPU for a brief smoke and resume it; commit and push.
A *real multi-hour training run* and a *keep-vs-revert on the production
architecture* still wait for an explicit human go. This paragraph is deleted the
day fully-unattended activates.
