# ERT Clean-Wrong Generalization Diagnostic

Status: held-out feature replay implementation prepared; GPU execution pending
environment preflight. No training is part of this plan.

## Frozen objective

Separate Direct, train Spillover, and Held-out effects for the completed
Clean-Wrong Broad Screen and the completed reliability-gated CleanCE screen.
Train-derived CE20/KL10 Teacher-margin boundaries are applied unchanged to the
validation set; validation outcomes never define boundaries.

## Existing inputs to reuse

- Broad Screen C0–C15 epoch-84 train/validation CE-PGD20 endpoints.
- Gated G0–G3 epoch-84/89/94 train/validation CE-PGD20 endpoints.
- Epoch-79 train-only CE20/KL10 Teacher-margin feature artifacts and their
  fixed parent/mask hashes.

## Missing input and execution order

The repository has no hash-bound epoch-79 validation feature artifact. The new
read-only producer `scripts/analysis/ert_cw_validation_feature_replay.py`
therefore creates exactly four artifacts: L2/L4 × CE-PGD20/KL-PGD10, using the
same epoch-79 parents and validation split. No model or optimizer state is
modified.

After the four artifacts pass row-count, stable-ID/class, attack, parent, and
source-lineage checks, a CPU report will compute:

- Broad Screen Direct / non-Clean-Wrong train Spillover / validation Held-out;
- train-derived Q1–Q5 subtype transfer;
- G2/G3 direct, within-CW spillover, non-CW spillover, and Held-out effects;
- rescue/harm/net-rescue and clean/robust probability-margin deltas;
- descriptive Direct→Spillover→Held-out pattern classification.

The report will fail closed if validation feature lineage, endpoint attack
identity, stable IDs/classes, or train-derived quantile boundaries do not
match. No threshold, coefficient, new seed, official test, AutoAttack, or new
intervention will be started automatically.
