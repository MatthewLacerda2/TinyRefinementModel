# Findings

Lab notebook of results worth publishing — novel results, or honest negative/neutral
results on novel approaches. One file per finding, plots stored alongside.

Everything here must be reproducible from what the entry records: the commit hash,
the run id, and the measurement command. If a claim can't be traced to those three
things, it doesn't go in this folder.

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
