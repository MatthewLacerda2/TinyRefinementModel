# Cosine halting on the serial scratchpad fails its pre-registered gate — the latent-convergence signal is too noisy to trade writes for accuracy, idea killed

Status: confirmed (negative result; pre-registered kill criterion met)
Date: 2026-07-10
Commit: b87059f  Run: tiny-config ablation (synthetic, no runs/ id)  Measured with: `python scratchpad_harness.py --halting --seeds 0,1,2` (CPU, FORCE_F32_COMPUTE=1; run twice — with and without the logit diagnostic — verdict tables byte-identical)

## Setup

The #39 idea: let the model decide its own reasoning depth by stopping the
scratchpad's write loop when the latent stops moving — halt at the first slot k
whose cosine similarity to slot k−1 exceeds a threshold τ. Deterministic,
inference-only (DEQ/fixed-point family, explicitly not ACT); the trained models
are the exact #38 serial arms, untouched.

Design and criteria pinned before the run (docs/design/cosine-halting.md, and
pre-registered on the issue). Because every uniform #38 instance needs all K
steps, the payoff case was built in-distribution: an **identity-tail eval
split** with effective length L ~ U{1..4} and (a=1, b=0) steps after L, so the
true sub-results freeze at step L and a working halter saves writes (oracle
mean writes 3.25 vs the fixed 4). Verdict on one global τ from a fixed grid,
3-seed means, σ_pooled convention as in prior findings. KEEP required, at some
τ: (a) identity-tail halting accuracy within 2σ_pooled of identity-tail
fixed-K, (b) identity-tail mean writes ≤ 3.5, (c) uniform halting accuracy
within 2σ_pooled of uniform fixed-K.

## Evidence

Anchors first: uniform fixed-K reproduced #38 bit-for-bit (0.9988 / 0.9790 /
0.9988, mean 0.9922 ± 0.011). Identity-tail fixed-K is 0.9836 ± 0.027 — within
2σ_pooled of uniform (Δ = 0.009 vs a 0.058 bar), so the pre-registered
generalization-confound guard did not fire and no mixed-pool contingency arm
was needed.

Halting ladder, 3-seed means (identity-tail split; fixed-K = 0.9836, K = 4):

| τ | acc | mean writes | verdict |
|---|---|---|---|
| 0.5 | 0.8138 ± 0.036 | 3.296 | writes ✓, accuracy −0.170 ≈ **3.8σ_pooled** ✗ |
| 0.7 | 0.8411 ± 0.036 | 3.478 | writes ✓, accuracy −0.143 ≈ **3.2σ_pooled** ✗ |
| 0.8 | 0.8625 | 3.595 | writes ✗ |
| 0.9 | 0.9181 ± 0.021 | 3.809 | writes ✗ (and accuracy −0.066 ≈ 1.9σ) |
| 0.95 | 0.9671 | 3.935 | writes ✗ |
| ≥0.98 | 0.9832–0.9836 | 3.99–4.00 | accuracy ✓ but saves nothing |

The two thresholds that clear the savings bar lose ~0.15 accuracy at 3–4× the
noise floor; every threshold that preserves accuracy saves essentially no
writes. The same trade shows on the uniform split (τ=0.5: 0.9129 vs 0.9922,
≈5σ — false positives on genuinely-moving latents). **No τ satisfies
(a)+(b)+(c); the kill criterion is met.**

**The mechanism readout says why.** On the identity-tail split, cosine at
converged steps (r_k = r_{k−1}) averages 0.59–0.65 vs ≈ 0.01 at computing
steps — the signal separates *in the mean* but with σ ≈ 0.18–0.27 on both
sides, the distributions overlap heavily and no threshold cuts cleanly. Two
structural reasons, visible in the numbers: (1) converged-step cosine never
approaches 1 — each slot carries a distinct slot-index embedding and is
written by a fresh cross-attention pass, so even a frozen *value* does not
reproduce the previous *latent*; (2) the write-once scratchpad has no iterated
state contracting to a fixed point — the DEQ picture the idea imported assumes
an architecture this design deliberately does not have. The halting rule is a
fixed-point detector pointed at a machine with no fixed point.

Diagnostic-only secondary (pre-registered, no claim): the same signal on the
slot grade logits — identity-tail split, per-seed means 0.847–0.868 (±0.11–0.12
within-step spread) at converged steps vs −0.130 to −0.152 (±0.30) at computing
steps. Semantic commitment separates far better than the raw latents (≈4σ of
the smaller spread vs ≈2.5σ for latent cosine, and the computing-side mean is
*negative* — successive grades actively disagree while the chain is still
computing). If adaptive depth is ever revisited, the detector should live in
grade/logit space, not latent space. Recorded for the record, untested as a
halting rule.

## Limitations

Toy scale, one task family, K=4, one architecture (the #38 serial scratchpad),
3 seeds, a fixed 9-point τ grid. The kill is specific: *raw-latent cosine
halting on a write-once slot scratchpad*. It does not condemn convergence
halting on a genuinely iterated state (e.g. the CausalRefiner's depth loop,
where consecutive iterates of the *same* state exist — a different issue if
ever wanted), nor logit-space detectors (above). ACT/learned halting stays
killed per the roadmap; nothing here reopens it.
