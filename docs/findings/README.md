# Findings

Lab notebook of results worth publishing — novel results, or honest negative/neutral
results on novel approaches. One file per finding, plots stored alongside.

Everything here must be reproducible from what the entry records: the commit hash,
the run id, and the measurement command. If a claim can't be traced to those three
things, it doesn't go in this folder.

## Entries

- `2026-06-11-slot-future-leak.md` — latent-scratchpad memory slots leaked future tokens into past predictions
- `2026-06-13-cross-window-hunch-inert.md` — the cross-window "hunch" gives no next-window benefit (graveyard)
- `2026-06-13-plan-a-depth-recurrence-works.md` — Plan A depth recurrence earns its compute on state-tracking (#16)
- `2026-06-14-plan-a-integrated-into-trainer.md` — CausalRefiner integrated into the production trainer
- `2026-06-16-plan-a-depth-ablation.md` — Plan A depth ablation on the toy tasks
- `2026-06-18-plan-a-depth-transfer.md` — Plan A depth survives LM pretraining, exploited better than from scratch
- `2026-06-19-plan-a-depth-dense-sweep.md` — dense sweep corrects the d8 read: depth plateaus by ~d6
- `2026-07-03-serial-scratchpad-beats-controls.md` — #38: graded serial slots beat both controls, order is the variable
- `2026-07-03-weight-tying-memorization-null.md` — weight-tying memorization/generalization trade: null both ways
- `2026-07-04-gpt2-yardstick-calibrated.md` — the GPT-2-small yardstick reproduces the reference reading (#48)
- `2026-07-04-slots-only-readout-compression-real.md` — #62: a readout blinded to tokens stays within noise; compression real
- `2026-07-04-final-only-supervision-decomposition-is-taught.md` — #67: without the per-slot grade the scratchpad sits at chance; the decomposition is taught, not emergent
- `2026-07-10-grade-annealing-scaffold-not-crutch.md` — #73: the grade is a scaffold — annealed to zero mid-run the chain survives on final-answer loss (within 2σ), but seed variance grows ~7×
- `2026-07-12-anneal-floor-wins-onset-is-a-state.md` — #95: anneal to a floor of λ≈0.1 (control-level σ, 7× calmer than zero); no fixed earlier onset is reliable — the grade must stay until the chain is decodable through the deep slots

## Entry template

```markdown
# <Claim, stated as a sentence>

Status: preliminary | confirmed | retracted
Date: YYYY-MM-DD
Commit: <hash>  Run: <runs/run_id>  Measured with: <command>

## Setup
Model/config and training regime, in two or three sentences.

## Evidence
The numbers and plots. State the baseline being compared against.

## Limitations
Scale caveats, confounds, what would strengthen or kill the claim.
```

## Candidate entries (to be written once the clean 8k run reports)

- ACT-style learned halting collapses to minimum depth at small scale; the failure
  survives reward shaping and only disappears when halting is removed.
- Training with randomly sampled reasoning depth costs nothing in convergence
  versus fixed shallow depth, and makes inference-time depth monotonically helpful.
- Depth-curve methodology: held-out CE versus reasoning depth, with the hard-token
  quartile ranked at depth 1 so the slice is fixed before depth varies.
- Post-mortem: a double-applied attention scaling survived weeks of training and
  log analysis; reference-numerics tests as the only reliable detector class.
