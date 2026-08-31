# ERT / RSLAD History-Balanced Ordering Dev v2

Status: complete; dev seeds 1 and 2 only. The primary comparison is NEW_HISTORY minus NEW_CONTROL under the new sample-keyed training attack RNG contract.

## Paired results

| seed | final robust Δ | final clean Δ | post-100 AUC Δ | endpoint 149 robust Δ | endpoint 199 robust Δ |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.060 pp | +0.180 pp | -0.057 pp | -0.620 pp | +0.020 pp |
| 2 | -0.100 pp | -0.080 pp | -0.032 pp | -0.260 pp | -0.360 pp |

## Contract

- Prefix: frozen I100 (CropShift epochs 0–99, IDBH_WEAK epochs 100–199).
- Training attack: KL-PGD10, 8/255, 2/255, random start, teacher-clean target; random starts are keyed by attack seed, epoch, source ID, stream tag, and restart index only.
- NEW_CONTROL uses canonical epoch shuffle; NEW_HISTORY uses frozen H2 `margin_ema` risk, HIGH/MID/LOW 20/60/20, HIGH/MID/MID/LOW/MID interleave, exact-once exposure.
- Endpoint: common CE-PGD20 on the fixed internal validation split (5,000 rows).
- W&B policy: metrics-only; model and run-bundle uploads disabled.

## Decision

No automatic promotion is performed. Any confirmation-seed campaign requires human review.

Machine artifact: `docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_results.json` (SHA-256 `1b08ffa9ecfbaec61d5dccbf25022e297289069ed5cbe7dce7c9184b9e23a271`).
