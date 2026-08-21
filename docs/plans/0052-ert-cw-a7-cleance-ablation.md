# ERT Clean-Wrong A7 CleanCE ablation

Status: complete; historical reuse audit, F3 canary/training, CE-PGD20 endpoints,
and factorial report are complete.

## Scientific correction

The proposal's arm names are not trusted.  The resolved historical treatment
fields show that A0 is baseline, A1 is CleanCE `0.15`, and A7 is **margin-only**
(`teacher_floor` with the frozen coefficient/floor/cap).  No historical arm is
CleanCE plus teacher-floor margin.  Therefore the factorial is assembled as:

| factorial arm | source |
|---|---|
| F0 | reuse historical A0 |
| F1 | reuse historical A1 |
| F2 | reuse historical A7 (margin-only) |
| F3 | fresh L2/L4 continuation (CleanCE + margin) |

This avoids rerunning three scientifically identical trajectories and prevents
the false claim that historical A7 was the full F3 arm.

## Frozen contract

- Parents: Chen L2/seed 1 and L4/seed 2, exact epoch-79 SHA recorded in the
  reuse audit.
- Cohort: fixed epoch-79 `student_clean_wrong` masks; no dynamic re-selection.
- Training attack: pixel-space KL-PGD10, epsilon `8/255`, step `2/255`, ten
  steps, random start, Teacher-clean target.
- Margin: `lambda=0.2388051152229309`, floor `0.03221710026264191`, cap
  `0.13952550292015076`, Teacher positive-floor target; Teacher frozen.
- CleanCE: F1/F3 extra coefficient `0.15`; full-batch mean, no selected-count
  normalization.
- Horizons: epoch 84, 89, 94; epoch 94 is primary.
- Endpoint: independent pixel-space CE-PGD20, epsilon `8/255`, step `2/255`,
  twenty steps, random start, eval mode, train/internal validation only.
- No lambda/floor/cap sweep, new seed, official test, or AutoAttack.

## Execution gates

1. Keep the clean-tree reuse audit and relevant source-component hash check.
2. Run focused tests and the F3 fixed-batch/no-update canary.  Verify the
   combined treatment adds both terms only on the fixed mask, does not create
   another PGD, and leaves Teacher gradients absent.
3. Launch only fresh F3 L2 and F3 L4 from the exact parents, with metrics-only
   W&B retention and local checkpoints/artifacts.
4. Run independent CE-PGD20 endpoints for F3 at 84/89/94 and combine them with
   the hash-bound reused F0/F1/F2 endpoint artifacts.
5. Report Direct, non-CW spillover, held-out overall, held-out Clean-Wrong,
   held-out CE20/KL10 Q1--Q5, factorial main effects/interaction, and Pareto
   summaries.  These are descriptive conditional effects, not training-seed
   population inference.
6. Stop after the report; do not automatically start sensitivity or another
   intervention.

## Completion record

- Reuse audit passed: historical A0/A1/A7 are F0/F1/F2 respectively; F3 was
  trained fresh for both seeds. The historical A7 was not mislabeled as F3.
- The one-epoch F3 L2 canary passed with finite metrics, exact parent lineage,
  fixed mask, combined loss path, and metrics-only W&B retention.
- F3 L2/L4 both completed epochs 80--94 and saved 84/89/94 checkpoints.
- All six F3 endpoints (two seeds, three horizons, train/validation) completed
  under independent CE-PGD20. Existing F0/F1/F2 endpoints were reused after
  stable-ID, class, attack, checkpoint, and row-hash validation.
- No lambda/floor/cap sweep, new seed, official test, or AutoAttack was run.
- The report is descriptive and does not auto-promote a treatment.

## Machine records

- Reuse audit: `docs/experiments/ert_cw_a7_cleance_reuse_audit_v1.json`
- Human audit: `docs/ERT_CW_A7_CLEANCE_REUSE_AUDIT.md`
- Final report: `docs/ERT_CW_A7_CLEANCE_ABLATION.md`
- Final machine record: `docs/experiments/ert_cw_a7_cleance_ablation_v1.json`
