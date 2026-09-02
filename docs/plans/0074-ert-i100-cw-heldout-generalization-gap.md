# 0074 — I100 Clean-Wrong held-out generalization-gap audit

## Status

- Owner: Codex
- Status: complete
- Scope: read-only mechanism analysis; no training or intervention
- Source: I100 long-horizon endpoints at `dbac5d8` lineage and exact e99 parents

## Frozen design

Use the epoch-99 Student clean-correctness state to define validation
Clean-Wrong/non-Clean-Wrong groups. Pair each group against the same-seed
I100 control at e129/e149/e169/e189/e199. Use the registered fixed train
Clean-Wrong masks for e199 train direct/spillover. Do not tune thresholds or
select a new action.

## Completion

- [x] Reconcile source and endpoint lineage.
- [x] Generate the missing e99 validation CE20/KL10 features by no-update replay.
- [x] Verify stable IDs, attack identity, and weighted held-out reconciliation.
- [x] Compute validation CW/non-CW effects and e199 train direct/spillover.
- [x] Compute temporal response turnover and Plain/TPFM rescue-harm overlap.
- [x] Record unavailable e99 train-feature shift fields without imputation.
- [x] Write machine artifacts and human report.
- [x] Run changed tests and commit/push.

## Boundary

The e114 row-level endpoint was not recoverable in the current local artifact
inventory; it is explicitly marked unavailable. No e114 values were invented.
The missing train-side e99 feature parquet prevents a valid train-vs-validation
feature-shift or TPFM floor/cap regime comparison. No S2 intervention,
threshold search, combined treatment, or additional seed follows automatically.
