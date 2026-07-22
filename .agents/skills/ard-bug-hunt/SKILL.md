---
name: ard-bug-hunt
description: Diagnose adversarial-training or distillation bugs, upstream reproduction mismatches, NaNs, gradient and attack errors, unstable metrics, checkpoint/resume failures, DDP inconsistencies, and W&B duplication. Use for debugging and unexplained scientific regressions; do not use for ordinary feature implementation.
---

# ARD bug-hunting workflow

## Objective

Find the smallest evidence-backed root cause without repeatedly running expensive or unchanged tests. Do not patch symptoms by weakening attacks, broadening tolerances, changing seeds, or suppressing failures.

## Workflow

1. Read `AGENTS.md`, `docs/SCIENTIFIC_INVARIANTS.md`, and `docs/TEST_STRATEGY.md`.
2. State the observed failure precisely: expected, actual, first known bad state, config, seed, hardware, checkpoint, and W&B run ID if relevant.
3. Check the test ledger. Do not repeat an exact successful command whose source, test, config, environment, external commit, and fixtures are unchanged.
4. Build at most five hypotheses, ranked by plausibility and discriminating evidence.
5. Select the smallest reproduction. Prefer one fixed batch, one checkpoint, one attack step trace, or one resume boundary over a full training run.
6. Choose the next action by information gain. Inspect code or tensors before launching another GPU run when possible.
7. Compare against `.external/saad` for baseline parity when the upstream commit is present.
8. Once the cause is identified, specify a minimal regression test that fails before the fix and passes after it.
9. Hand the bounded fix to an implementation agent. Re-run only the failed/affected tests, then the impact-selected milestone checks.
10. Write a debug note under `docs/debugging/` for scientifically consequential bugs.

## Mandatory checks

Consult `references/ard_failure_modes.md`. At minimum inspect:

- pixel vs normalized space
- epsilon and step-size units
- projection and clamp order
- attack loss sign and gradient source
- train/eval mode and BatchNorm
- teacher parameter freeze vs input gradients
- detach and `no_grad` boundaries
- KL direction, temperature, and scaling
- stable sample indices and DDP sampler behavior
- AMP/GradScaler and finite values
- checkpoint completeness and RNG restoration
- W&B rank, run ID, resume, step monotonicity, and offline sync

## Output format

Return:

1. Failure signature
2. Evidence collected
3. Ranked hypotheses
4. Root cause or remaining uncertainty
5. Minimal regression test
6. Bounded fix specification
7. Tests that must run after the fix
8. Tests explicitly not rerun because their fingerprint is unchanged
