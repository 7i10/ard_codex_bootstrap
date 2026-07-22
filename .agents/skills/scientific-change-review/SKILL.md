---
name: scientific-change-review
description: Review adversarial robustness code changes for threat-model drift, gradient mistakes, numerical mismatch, reproducibility gaps, insufficient tests, evaluation leakage, DDP problems, and W&B lineage. Use for milestone or PR review; do not use for style-only review.
---

# Scientific change review

1. Read the plan, diff, `AGENTS.md`, and affected scientific contracts.
2. Map each changed file to possible effects on inputs, attacks, objectives, state, checkpoints, evaluation, tracking, and reported metrics.
3. Inspect the real execution path, not only the edited function.
4. Verify that any behavior change is explicit in config and documented.
5. Check the impact-selected tests and test ledger. Request only missing high-information tests; do not request repetition of unchanged successful suites.
6. For baseline code, compare equations/defaults with the pinned upstream implementation and documented paper behavior.
7. For W&B, check rank-zero ownership, run grouping, resume identity, config lineage, artifacts, and logging cadence.
8. Report findings by severity:
   - P0: invalidates results or corrupts data/checkpoints
   - P1: likely scientific or runtime bug
   - P2: meaningful robustness, reproducibility, or maintainability risk
9. Include file/symbol references and a minimal reproducer or targeted test when possible.
10. Do not edit files and do not include style-only comments.

If no consequential issue is found, state what was reviewed, which risks remain unverified, and which expensive tests were intentionally deferred.
