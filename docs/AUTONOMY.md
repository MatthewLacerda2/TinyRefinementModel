# Autonomous Development — harness, guardrails, oversight

How Claude develops this model semi-autonomously along the roadmap. The goal is
to remove the iteration bottleneck (test theories on tiny models in minutes, not
days) while keeping the decisions that need human judgment with the human.

Note on framing: this is not self-modification — the model running Claude is
fixed. It's Claude autonomously *developing the TinyRefinementModel* along the
roadmap. Referenced docs: the optimization roadmap and findings/ entries.

## Build — the ablation harness (Claude builds these)

- **Tiny-config mechanism**: dim 64–128, 1–2 layers, seq ~64, tiny vocab, so a
  run is minutes. The unit of fast iteration.
- **Ablation runner**: config-delta → fixed small budget → metric → compare to a
  baseline. One command, structured result.
- **Depth-sensitive toy-task suite**: algorithmic tasks (modular arithmetic,
  parity, copy/sort) where depth genuinely has to do something and grokking is
  visible fast. This is also the correct *proof instrument* for Plan A — it
  sidesteps the "fineweb perplexity is the wrong yardstick" problem. A tiny
  code-completion set is a secondary, more-realistic-but-slower, noisier check —
  NOT for fast grokking signal (code is too rich to show the phase transition
  quickly; the literature's clean grokking is on algorithmic tasks).
- **Results log**: lightweight, structured, comparable. findings/ remains the
  human-readable, publishable layer.

## Guardrails — what makes unsupervised work safe

- **Compute budget is the real resource guardrail.** Ablations are capped small
  (minutes, tiny configs) and may run unsupervised. Any multi-hour or overnight
  training run on the *real* model needs the user's explicit go. The permission
  system blocking destructive/large actions is a feature — keep it.
- **A branch isolates code, not resources.** Unsupervised code work happens on a
  deletable branch (clean to discard). But a branch does NOT isolate compute
  spend, disk (runs/, checkpoints), or external side-effects — so the compute
  budget above, not the branch, is what protects resources.
- **Gate before run.** Workflow is: design doc → implement → pass the test
  battery (causality invariants / no leak, overfit-smoke drives a batch to ~0,
  init-loss canary, golden-run determinism) → only then run. Never launch on a
  red gate.
- **Pass/fail metric agreed in advance.** For each roadmap item the success
  criterion is set with the user beforehand. Claude executes and reports; the
  user owns the criterion, so Claude isn't both runner and judge.
- **Ablation results are directional.** Best for cheaply killing bad ideas and
  ranking options. A positive tiny-model result still needs confirmation at real
  scale before it's trusted.

## From the user — oversight that stays human

- **Go/no-go at expensive-run gates and direction calls.** This is Claude's
  weakest spot: the failure mode is fluent confidence on a known-hard idea,
  flagged too late (see the cross-window-hunch finding). "Claude rarely pushes
  back" is a reason to check *at the gates more*, not less — the one confidently-
  wrong direction, run unsupervised, costs a week instead of 20 minutes.
- **The frozen prefill plans/questions** (current blocker for the r50k rebuild).
- **Final arbiter of whether a direction is worth pursuing.**

## Workflow

plan (design doc) → implement on branch → gate (test battery) → cheap ablations
within budget (unsupervised OK) → report → [user go] → real-model run.
