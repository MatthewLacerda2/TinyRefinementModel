## Summary
<!-- One or two lines: what this PR does. -->

## What & why
<!-- The change and the reasoning. If it implements a finding or closes an issue, link it. -->

## Validation / smoke-test performed
<!-- What you actually ran. No multi-hour training run should be needed to review this. -->
- [ ] `JAX_PLATFORMS=cpu pytest tests/ -q` passes locally
- Other checks run:

## Risk
<!-- Numerics, VRAM, API churn (jax/flax are pinned), data path, checkpoint format. What could break? -->

## Checklist
- [ ] Did not weaken or delete any test (no skips/xfails/loosened asserts to make it pass)
- [ ] Smoke-tested; no multi-hour run was launched without sign-off
- [ ] Pinned deps unchanged, or the bump is justified above
- [ ] Findings/claims backed by evidence in the PR or a linked finding issue
