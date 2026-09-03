# ERT I100 canonical S2×T1 boundary geometry audit

Status: complete read-only analysis. No training, intervention, threshold tuning, new seed, official test, or AutoAttack was run.

## Executive answer

- Exact e99 I100 parents and fixed validation S2×T1 masks were used. The anchor is the registered Student CE-PGD20 point; replay metadata resolves its random start as batch-index.
- Geometry uses the Student strongest non-true logit as a shared class pair. Input gradients are through pixel-space adapters; Teacher parameters are frozen and have no parameter grads.
- Primary scalars are normal cosine/mismatch and first-order L∞ distance proxies `m_pair / ||g||_1`; these are not exact distances.
- Decision: **BG2_DISTANCE_GAP_SUPPORTED**. Geometry is descriptive; no geometry-based intervention follows.

## Cohort and lineage

Parents: dev-1 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`, dev-2 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`. Teacher `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`. Endpoint attack `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`. Validation split `16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4`.

| seed | fixed validation S2×T1 n | mask IDs SHA |
|---|---:|---|
| dev-1 | 226 | `048707beadfba2d679933137c330e120e267f0c3d0a94c26a739bbe46ad8fae2` |
| dev-2 | 209 | `03a3c90b16b07076da16e82e8d9b83ad9ec602c49214c095249ed9b9f7a0ed48` |

The registered machine replay metadata is authoritative for the random-start stream: `evaluation_attack + batch_index`. The mask JSON's descriptive `sample_keyed_v1` label is retained as a provenance discrepancy and was not silently rewritten.

## Replay integrity

| seed | validation rows | scalar rows SHA | batched/single max error |
|---|---:|---|---:|
| dev-1 | 226 | `16c2e8144e3de5c1b7c0121ddac3e218c9931e83c6bc0e4e92cf35355562a4b7` | 8.941e-08 |
| dev-2 | 209 | `70705b2ce05e0722ce3bea5a4e8164b620047715e74edc56bb3be9aa2bcb177b` | 8.941e-08 |

## Cross-seed predictor AUROC

AUROC is descriptive; fit uses one seed and evaluates the other with fixed ridge logistic alpha=1.0 and fit-seed standardization.

### e104

| predictor | dev1→dev2 | dev2→dev1 |
|---|---:|---:|
| student_margin | 0.4759 | 0.4739 |
| teacher_margin | 0.7149 | 0.6710 |
| student_teacher_margins | 0.7122 | 0.6719 |
| student_distance | 0.4913 | 0.5058 |
| distance_gap | 0.7303 | 0.6870 |
| normal_mismatch | 0.4514 | 0.4141 |
| geometry_minimal | 0.6635 | 0.6631 |
| geometry_distance_relation | 0.7264 | 0.6783 |
| margin_plus_geometry | 0.6788 | 0.6683 |

### e109

| predictor | dev1→dev2 | dev2→dev1 |
|---|---:|---:|
| student_margin | 0.5283 | 0.5569 |
| teacher_margin | 0.6510 | 0.7134 |
| student_teacher_margins | 0.6522 | 0.7143 |
| student_distance | 0.4766 | 0.5092 |
| distance_gap | 0.6463 | 0.7261 |
| normal_mismatch | 0.4680 | 0.4038 |
| geometry_minimal | 0.6101 | 0.6956 |
| geometry_distance_relation | 0.6428 | 0.7116 |
| margin_plus_geometry | 0.6223 | 0.6837 |

### e114

| predictor | dev1→dev2 | dev2→dev1 |
|---|---:|---:|
| student_margin | 0.6058 | 0.5183 |
| teacher_margin | 0.6868 | 0.7602 |
| student_teacher_margins | 0.6738 | 0.7266 |
| student_distance | 0.4286 | 0.4605 |
| distance_gap | 0.6935 | 0.7767 |
| normal_mismatch | 0.5055 | 0.6076 |
| geometry_minimal | 0.6657 | 0.7614 |
| geometry_distance_relation | 0.6725 | 0.7273 |
| margin_plus_geometry | 0.6746 | 0.7177 |

## Geometry added-value contrasts

Values are AUROC(feature/predictor) minus AUROC(Student margin), using fixed cross-seed directions.

| epoch | direction | normal mismatch | Teacher−Student distance gap |
|---:|---|---:|---:|
| 104 | dev-1_to_dev-2 | -0.0245 | +0.2544 |
| 104 | dev-2_to_dev-1 | -0.0598 | +0.2131 |
| 109 | dev-1_to_dev-2 | -0.0602 | +0.1180 |
| 109 | dev-2_to_dev-1 | -0.1531 | +0.1692 |
| 114 | dev-1_to_dev-2 | -0.1003 | +0.0877 |
| 114 | dev-2_to_dev-1 | +0.0893 | +0.2584 |
## Geometry cells

Cells use within-seed medians descriptively only; they are not selectors.

### dev-1

| epoch | cell | n | failure rate |
|---:|---|---:|---:|
| 104 | high_alignment+adequate_distance | 48 | 0.7292 |
| 104 | high_alignment+low_student_distance | 65 | 0.7692 |
| 104 | low_alignment+adequate_distance | 65 | 0.6462 |
| 104 | low_alignment+low_student_distance | 48 | 0.5625 |
| 109 | high_alignment+adequate_distance | 48 | 0.7917 |
| 109 | high_alignment+low_student_distance | 65 | 0.7538 |
| 109 | low_alignment+adequate_distance | 65 | 0.6923 |
| 109 | low_alignment+low_student_distance | 48 | 0.5833 |
| 114 | high_alignment+adequate_distance | 48 | 0.8125 |
| 114 | high_alignment+low_student_distance | 65 | 0.8154 |
| 114 | low_alignment+adequate_distance | 65 | 0.6769 |
| 114 | low_alignment+low_student_distance | 48 | 0.6667 |

### dev-2

| epoch | cell | n | failure rate |
|---:|---|---:|---:|
| 104 | high_alignment+adequate_distance | 46 | 0.7826 |
| 104 | high_alignment+low_student_distance | 59 | 0.7627 |
| 104 | low_alignment+adequate_distance | 59 | 0.8136 |
| 104 | low_alignment+low_student_distance | 45 | 0.8222 |
| 109 | high_alignment+adequate_distance | 46 | 0.7826 |
| 109 | high_alignment+low_student_distance | 59 | 0.7966 |
| 109 | low_alignment+adequate_distance | 59 | 0.8475 |
| 109 | low_alignment+low_student_distance | 45 | 0.8000 |
| 114 | high_alignment+adequate_distance | 46 | 0.8913 |
| 114 | high_alignment+low_student_distance | 59 | 0.7458 |
| 114 | low_alignment+adequate_distance | 59 | 0.7966 |
| 114 | low_alignment+low_student_distance | 45 | 0.8222 |

## Secondary train S2×T1 replay

The train replay uses the e99 CropShift view and joins only the existing e114 train endpoint. It is secondary and descriptive.

| seed | n | e114 future-failure n | distance-gap AUROC | normal-mismatch AUROC |
|---|---:|---:|---:|---:|
| dev-1 | 2212 | 1641 | 0.6568 | 0.4759 |
| dev-2 | 2141 | 1567 | 0.6573 | 0.4871 |

## Interpretation and stop boundary

Geometry does not establish that mismatch or distance causes future failure, nor that alignment/distance distillation improves robustness. No new loss, route, threshold, or training is started. Full scalar details and hashes are in the machine artifact.
