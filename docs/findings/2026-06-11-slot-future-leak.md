# Latent-scratchpad models can leak future tokens into past predictions through their memory slots

Status: confirmed
Date: 2026-06-11
Commit: f55c29b  Run: n/a (architectural, affects all runs to date)
Measured with: `venv/bin/python -m pytest tests/test_model_invariants.py` (the strict xfail)

## Setup

The Universal Reasoner architecture: a causal encoder over the token window, a
weight-shared reasoning block iterated over 32 latent slots that cross-attend to
the full encoded sequence bidirectionally, and a causal decoder that attends to
both the token sequence and the final slot state from every position.

## Evidence

The slot cross-attention is non-causal by design (the scratchpad is meant to
summarize the window). But the causal encoder places token t's content at
position t, the slots aggregate all positions, and the decoder exposes the
slots to every query position — so the prediction at position t has a path to
information about tokens after t, including its own target.

Empirical confirmation: perturbing the token at position 40 of a 64-token
window changes the logits at positions before 40. The test asserting causality
fails deterministically (kept as a strict expected-failure in the suite).

Consequences for any architecture of this family:

- Teacher-forced training CE is optimistic — the loss rewards routing future
  information through the scratchpad.
- Any "more reasoning steps lower CE" measurement made under teacher forcing
  is confounded: each extra iteration is another round in which the slots can
  ferry future content, so depth gains may measure leak bandwidth rather than
  refinement.
- At generation time there is no future to leak, so the learned behavior
  partially fails to transfer — consistent with our observation that extra
  reasoning steps improved measured CE but not the quality of generated text.

## Limitations

Demonstrated at 79.6M parameters on one architecture; the magnitude of the
training-signal distortion (vs the mere existence of the path) is not yet
quantified. A clean quantification: compare depth curves where slots are
computed from the same window vs only from the previous window. The structural
argument, however, applies to any design where a bidirectional summary is
visible to causal decode positions.
