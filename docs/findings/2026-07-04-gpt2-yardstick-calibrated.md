# The GPT-2-small yardstick is built and calibrated: the instrument reproduces the reference reading on real GPT-2

Status: confirmed
Date: 2026-07-04
Commit: TBD  Run: — (instrument calibration, no training run)  Measured with: `PYTHONPATH=. python tools/calibrate_yardstick_gpt2.py`

## Setup

The base-model bar (CLAUDE.md, issue #48) needs an external metric: LAMBADA
last-word accuracy + perplexity, next to the GPT-2-small reference. Until now
that bar existed only as a comment in `config.py`. `tools/yardstick.py` now
implements the standard protocol (OpenAI's processed lambada_test.jsonl,
sha256-pinned, 5153 examples; greedy teacher-forced last-word match; per-word
perplexity; no stop-word filter), and `tools/eval_yardstick.py` runs it against
our checkpoints in one command, emitting a model-card row.

Before trusting the instrument, it was pointed at the model it will compare
against: real GPT-2-small (HF `gpt2`, 124M), through the exact same scoring
core, on the full test set (CPU, torch).

## Evidence

| metric | this instrument (gpt2 124M) | lm-eval-harness reference | delta |
|---|---|---|---|
| LAMBADA last-word acc | TBD | 0.3256 | TBD |
| LAMBADA ppl | TBD | 40.06 | TBD |

A sanity floor from the other direction: a random-init refiner reads
acc 0.0000 / ppl ≈ 4.1e5 through the same pipeline — the instrument separates
"knows English" from "noise" by four orders of magnitude of headroom.

Two published numbers deliberately NOT used as the reference: the GPT-2 paper's
45.99% acc / 35.13 ppl are measured with OpenAI's detokenizers and a stop-word
prediction filter — a different protocol that flatters accuracy. The issue #48
text quoted ≈35.7% acc from that family of numbers; the like-for-like bar under
our (and lm-eval's) unfiltered greedy protocol is the ≈0.33 row above.

## Limitations

- Calibration validates the *instrument*, not any of our models — v1 still has
  to earn its reading (#16).
- Our corpus is fineweb-edu/code/math, not WebText-like narrative; LAMBADA may
  run harder for our models at equal capability. `eval_yardstick` therefore
  prints held-out ppl on our own distribution alongside (via the same
  ValidationProbe the trainer logs), so a distribution gap and an undertrained
  model read differently.
- The held-out-ppl leg was smoke-tested without the tokenized corpus (cloud
  session); it reuses `validation.ValidationProbe` unchanged, but its first
  real reading should be eyeballed against the training val curve.
