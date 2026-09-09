# ERT / RSLAD Stage-Wise Augmentation Results

Status: complete. This is the preregistered two-seed internal-validation screen; no official test or AutoAttack was run.

## Contract and lineage

- Production source SHA: `bb68afc0ff505248f84c0263179ec24f0b346bcd`.
- Teacher: Chen2021LTD WRN34-10, SHA-256 `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.
- Parent: accepted CROPSHIFT seed-specific trajectories; hybrid prefix uses historical rows before the switch.
- Training attack: KL-PGD10, epsilon 8/255, step 2/255, random start, teacher-clean target.
- Endpoint: independent CE-PGD20, epsilon 8/255, step 2/255, 20 steps, random start, eval mode.
- Trajectory AUC metric: per-epoch internal validation `val_pgd_accuracy` under the frozen selection attack contract.
- W&B: metrics-only tracking; checkpoints and run bundles remain local.

## Final endpoint (validation, CE-PGD20)

| seed | schedule | late policy | switch | clean | robust | Δ robust vs CROPSHIFT | Δ clean |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | R100 | CROP_RE | 100 | 86.240% | 60.100% | +0.70 pp | +0.12 pp |
| 2 | R100 | CROP_RE | 100 | 86.200% | 60.100% | +0.84 pp | -0.02 pp |
| 1 | I100 | IDBH_WEAK | 100 | 85.820% | 60.600% | +1.20 pp | -0.30 pp |
| 2 | I100 | IDBH_WEAK | 100 | 86.040% | 60.300% | +1.04 pp | -0.18 pp |
| 1 | R150 | CROP_RE | 150 | 85.960% | 59.680% | +0.28 pp | -0.16 pp |
| 2 | R150 | CROP_RE | 150 | 86.200% | 59.900% | +0.64 pp | -0.02 pp |
| 1 | I150 | IDBH_WEAK | 150 | 85.780% | 60.160% | +0.76 pp | -0.34 pp |
| 2 | I150 | IDBH_WEAK | 150 | 85.920% | 60.160% | +0.90 pp | -0.30 pp |

## Hybrid trajectory and post-switch AUC

AUC is normalized with the repository's trapezoidal epoch convention. Deltas are schedule minus the same-seed CROPSHIFT control.

| seed | schedule | best robust (epoch) | last robust | full AUC | Δ full AUC | post-switch AUC | Δ post AUC | shock +1 | recovery epoch | throughput Δ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | R100 | 60.140% (199) | 60.140% | 0.516843 | +0.224 pp | 0.584093 | +0.452 pp | -0.04 pp | 104 | +1.88% |
| 2 | R100 | 60.160% (183) | 60.020% | 0.516365 | +0.190 pp | 0.583740 | +0.381 pp | -0.30 pp | 102 | -0.31% |
| 1 | I100 | 61.200% (197) | 60.920% | 0.519623 | +0.502 pp | 0.589668 | +1.010 pp | -0.10 pp | 104 | +1.76% |
| 2 | I100 | 60.680% (199) | 60.680% | 0.517595 | +0.313 pp | 0.586220 | +0.629 pp | -0.46 pp | 102 | -0.33% |
| 1 | R150 | 59.860% (192) | 59.840% | 0.515554 | +0.095 pp | 0.593480 | +0.388 pp | +0.00 pp | 151 | +1.97% |
| 2 | R150 | 60.040% (198) | 59.620% | 0.515508 | +0.104 pp | 0.594553 | +0.421 pp | -0.06 pp | 152 | -0.36% |
| 1 | I150 | 60.560% (197) | 60.220% | 0.516374 | +0.177 pp | 0.596798 | +0.719 pp | +0.00 pp | none | +1.92% |
| 2 | I150 | 60.500% (197) | 59.960% | 0.515947 | +0.148 pp | 0.596347 | +0.601 pp | -0.06 pp | 153 | -0.36% |

## Promotion gates (preregistered)

A schedule must pass both seeds for final robustness, full hybrid AUC non-degradation, post-switch AUC improvement, and the final-clean guardrail. No automatic promotion is made.

| schedule | final robust | full AUC | post-switch AUC | clean guardrail | throughput guardrail | qualifies |
|---|---|---|---|---|---|---|
| R100 | PASS | PASS | PASS | PASS | PASS | YES |
| I100 | PASS | PASS | PASS | PASS | PASS | YES |
| R150 | PASS | PASS | PASS | PASS | PASS | YES |
| I150 | PASS | PASS | PASS | PASS | PASS | YES |

## Interpretation

- `R100`/`R150` are CropShift → CROP_RE; `I100`/`I150` are CropShift → IDBH_WEAK.
- Endpoint values and trajectory AUC are reported separately; a final gain alone is not sufficient.
- S150 pre-switch endpoint is the hash-bound historical CROPSHIFT epoch-149 endpoint and is not re-attacked.
- Shock/recovery values are descriptive and do not trigger additional schedule tuning.
- Decision record: `human_review_required`; incumbent remains `CROPSHIFT fixed` pending human review.

## Limitations

- Two development seeds; no population-level seed inference.
- Validation endpoint is internal held-out CIFAR-10, not official test.
- No AutoAttack or additional timing/augmentation schedule was run.
- Hybrid AUC uses the accepted CROPSHIFT prefix and stage-wise suffix at the same epoch-metric contract.

Machine artifact: `docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json` (SHA-256 `8c6e273226df60bd94c6f0e1ed0c73548d9523015d07b9f2b27fdeb5e2538946`).
