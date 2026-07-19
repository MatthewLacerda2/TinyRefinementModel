# Findings

Lab notebook of results worth publishing — novel results, or honest negative/neutral
results on novel approaches. One file per finding, plots stored alongside.

Everything here must be reproducible from what the entry records: the commit hash,
the run id, and the measurement command. If a claim can't be traced to those three
things, it doesn't go in this folder.

## Entries

- `2026-06-13-cross-window-hunch-inert.md` — the cross-window "hunch" gives no next-window benefit (graveyard)
- `2026-06-13-plan-a-depth-recurrence-works.md` — Plan A depth recurrence earns its compute on state-tracking (#16)
- `2026-06-16-plan-a-depth-ablation.md` — Plan A depth ablation on the toy tasks
- `2026-06-18-plan-a-depth-transfer.md` — Plan A depth survives LM pretraining, exploited better than from scratch
- `2026-06-19-plan-a-depth-dense-sweep.md` — dense sweep corrects the d8 read: depth plateaus by ~d6
- `2026-07-03-serial-scratchpad-beats-controls.md` — #38: graded serial slots beat both controls, order is the variable
- `2026-07-03-weight-tying-memorization-null.md` — weight-tying memorization/generalization trade: null both ways
- `2026-07-04-slots-only-readout-compression-real.md` — #62: a readout blinded to tokens stays within noise; compression real
- `2026-07-04-final-only-supervision-decomposition-is-taught.md` — #67: without the per-slot grade the scratchpad sits at chance; the decomposition is taught, not emergent
- `2026-07-05-budget-scratchpad-overwrite-carries-chain.md` — #63 phase 1: a single physically-overwritten slot matches K append-only slots
- `2026-07-05-budget-scratchpad-recall-inconclusive-readout-confound.md` — #63 phase 2: RETRACTED (2026-07-16 addendum) — the eval scored recall arms against the wrong target, so the run decided nothing; rerun under the corrected eval
- `2026-07-05-truncated-backprop-depth-kill.md` — #64: gradient through only the last refinement steps collapses state-tracking — the trajectory gradient is load-bearing (kill)
- `2026-07-05-per-pass-supervision-islands.md` — #75: per-pass grades cannot replace the trajectory gradient (islands killed), but they stabilize deep recurrence (parity d8 rescued)
- `2026-07-07-dense-supervision-without-slots-collapses.md` — #79: the serial arm's exact supervision without slots collapses at the composition point; the offload is load-bearing
- `2026-07-10-grade-annealing-scaffold-not-crutch.md` — #73: the grade is a scaffold — annealed to zero mid-run the chain survives on final-answer loss (within 2σ), but seed variance grows ~7×
- `2026-07-12-anneal-floor-wins-onset-is-a-state.md` — #95: anneal to a floor of λ≈0.1 (control-level σ, 7× calmer than zero); no fixed earlier onset is reliable — the grade must stay until the chain is decodable through the deep slots
- `2026-07-15-f16-no-loss-scaling-no-dense-underflow.md` — #82: dense-kernel zero-grad fraction 0.0003 vs the 0.05 bar on the f16 no-loss-scaling path — underflow measurable but negligible at init; base run's early stretch is the confirming read
- `2026-07-16-budget-scratchpad-recall-rerun-readout-fine-writer-leaks.md` — #114: corrected-eval rerun of #63 phase 2 — the readout combines two values at 0.99 (old diagnosis dead), but the token-visible writer leaks the answer past the S=1 control, so retention is still untested
- `2026-07-18-budget-scratchpad-retention-win-slot-parking.md` — #116: retention under capacity pressure is real — leak closed, 2 slots for 5 writes hits 1.0000 on every seed by parking r_1 (~99.5% of later writes routed away from it), while the 1-slot in-vector carry trains unreliably
- `2026-07-18-sinusoidal-time-signal-depth-extrapolates.md` — #86: sinusoidal step signal matches the learned table at trained depths (2σ parity) and converts never-trained loops 9–16 into +0.11 accuracy under length shift, where the clamped table collapses to chance with NaN loss — depth becomes an open dial
- `2026-07-19-trajectory-visible-halting-collapses-incentive-not-observability.md` — #123: letting the halting head reread the whole thought trajectory does NOT prevent ACT-style collapse (99.5%+ mass on step 1, corr ≈ 0, every seed, solvable task) — the graveyard failure is an incentive pathology, not an observability one; halting pressure also rots full-depth competence 0.996 → ~0.6

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
