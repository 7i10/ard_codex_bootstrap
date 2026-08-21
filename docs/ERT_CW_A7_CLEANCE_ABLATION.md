# ERT Clean-Wrong A7 CleanCE ablation

Status: completed 2x2 factorial assembly. Historical A0/A1/A7 are F0/F1/F2; only F3 was trained fresh.

## Frozen interpretation boundary

This is a two-seed, paired, descriptive ablation. Bootstrap/training-seed population inference is not claimed.
Historical A7 is margin-only; treating it as full A7 would be incorrect.

## F3 endpoint absolute metrics

| seed | epoch | split | clean | robust |
|---|---:|---|---:|---:|
| L2 | 84 | train | 0.7911 | 0.5023 |
| L2 | 84 | validation | 0.7806 | 0.4674 |
| L2 | 89 | train | 0.8244 | 0.4867 |
| L2 | 89 | validation | 0.8022 | 0.4420 |
| L2 | 94 | train | 0.7926 | 0.4840 |
| L2 | 94 | validation | 0.7802 | 0.4572 |
| L4 | 84 | train | 0.8137 | 0.5027 |
| L4 | 84 | validation | 0.8038 | 0.4558 |
| L4 | 89 | train | 0.8120 | 0.5164 |
| L4 | 89 | validation | 0.7982 | 0.4694 |
| L4 | 94 | train | 0.8138 | 0.4975 |
| L4 | 94 | validation | 0.7984 | 0.4572 |

## Epoch-94 held-out overall (all factorial arms)

| seed | arm | clean | robust | clean Δ vs F0 | robust Δ vs F0 |
|---|---|---:|---:|---:|---:|
| L2 | F0 | 0.7752 | 0.4544 | +0.0000 | +0.0000 |
| L2 | F1 | 0.7766 | 0.4380 | +0.0014 | -0.0164 |
| L2 | F2 | 0.7730 | 0.4626 | -0.0022 | +0.0082 |
| L2 | F3 | 0.7802 | 0.4572 | +0.0050 | +0.0028 |
| L4 | F0 | 0.7848 | 0.4722 | +0.0000 | +0.0000 |
| L4 | F1 | 0.7970 | 0.4658 | +0.0122 | -0.0064 |
| L4 | F2 | 0.7902 | 0.4828 | +0.0054 | +0.0106 |
| L4 | F3 | 0.7984 | 0.4572 | +0.0136 | -0.0150 |

## Epoch-94 Direct / Spillover / Held-out Clean-Wrong effects

Effects are paired accuracy deltas versus F0; positive values mean rescue exceeds harm.

| seed | arm | direct clean | direct robust | spillover robust | held-out CW clean | held-out CW robust |
|---|---|---:|---:|---:|---:|---:|
| L2 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2 | F1 | +0.0557 | +0.0039 | -0.0300 | +0.0349 | -0.0009 |
| L2 | F2 | +0.0183 | +0.0026 | +0.0043 | +0.0018 | +0.0018 |
| L2 | F3 | +0.0760 | +0.0099 | -0.0165 | +0.0459 | +0.0018 |
| L4 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L4 | F1 | +0.0901 | +0.0121 | -0.0136 | +0.0670 | +0.0154 |
| L4 | F2 | +0.0180 | +0.0074 | +0.0091 | +0.0190 | +0.0118 |
| L4 | F3 | +0.0887 | +0.0176 | -0.0188 | +0.0688 | +0.0090 |

## Epoch-94 factorial contrasts (paired accuracy deltas)

| seed | cohort/metric | CE no margin | margin no CE | CE given margin | margin given CE | interaction |
|---|---|---:|---:|---:|---:|---:|
| L2 | train_direct_cw_clean | +0.0557 | +0.0183 | +0.0578 | +0.0203 | +0.0020 |
| L2 | train_direct_cw_robust | +0.0039 | +0.0026 | +0.0074 | +0.0060 | +0.0035 |
| L2 | train_spillover_non_cw_clean | -0.0090 | +0.0006 | -0.0089 | +0.0007 | +0.0001 |
| L2 | train_spillover_non_cw_robust | -0.0300 | +0.0043 | -0.0209 | +0.0134 | +0.0091 |
| L2 | validation_cw_clean | +0.0349 | +0.0018 | +0.0440 | +0.0110 | +0.0092 |
| L2 | validation_cw_robust | -0.0009 | +0.0018 | +0.0000 | +0.0028 | +0.0009 |
| L2 | validation_overall_clean | +0.0014 | -0.0022 | +0.0072 | +0.0036 | +0.0058 |
| L2 | validation_overall_robust | -0.0164 | +0.0082 | -0.0054 | +0.0192 | +0.0110 |
| L4 | train_direct_cw_clean | +0.0901 | +0.0180 | +0.0706 | -0.0014 | -0.0195 |
| L4 | train_direct_cw_robust | +0.0121 | +0.0074 | +0.0102 | +0.0055 | -0.0020 |
| L4 | train_spillover_non_cw_clean | -0.0014 | -0.0026 | +0.0003 | -0.0008 | +0.0017 |
| L4 | train_spillover_non_cw_robust | -0.0136 | +0.0091 | -0.0279 | -0.0052 | -0.0143 |
| L4 | validation_cw_clean | +0.0670 | +0.0190 | +0.0498 | +0.0018 | -0.0172 |
| L4 | validation_cw_robust | +0.0154 | +0.0118 | -0.0027 | -0.0063 | -0.0181 |
| L4 | validation_overall_clean | +0.0122 | +0.0054 | +0.0082 | +0.0014 | -0.0040 |
| L4 | validation_overall_robust | -0.0064 | +0.0106 | -0.0256 | -0.0086 | -0.0192 |

## Held-out Clean-Wrong Q1--Q5 (epoch 94)

Q1--Q5 boundaries are derived from the epoch-79 train Clean-Wrong feature distribution and are applied to pre-treatment validation Clean-Wrong IDs. No outcome is used for binning.

### CE20 Teacher-margin bins: clean Δ vs F0

| seed | arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| L2 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2 | F1 | +0.0299 | +0.0279 | +0.0682 | +0.0153 | +0.0355 |
| L2 | F2 | -0.0050 | +0.0112 | +0.0057 | -0.0255 | +0.0148 |
| L2 | F3 | -0.0050 | +0.0056 | +0.0795 | +0.0357 | +0.0858 |
| L4 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L4 | F1 | +0.0235 | +0.1006 | +0.0455 | +0.0812 | +0.0794 |
| L4 | F2 | +0.0329 | +0.0335 | -0.0057 | +0.0305 | +0.0088 |
| L4 | F3 | +0.0235 | +0.0615 | +0.0455 | +0.0812 | +0.1059 |

### CE20 Teacher-margin bins: robust Δ vs F0

| seed | arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| L2 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2 | F1 | +0.0000 | +0.0000 | +0.0000 | -0.0153 | +0.0059 |
| L2 | F2 | +0.0000 | +0.0056 | -0.0057 | +0.0000 | +0.0059 |
| L2 | F3 | +0.0000 | +0.0000 | -0.0057 | -0.0153 | +0.0178 |
| L4 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L4 | F1 | +0.0094 | +0.0000 | +0.0227 | +0.0102 | +0.0265 |
| L4 | F2 | +0.0047 | +0.0000 | +0.0057 | +0.0152 | +0.0235 |
| L4 | F3 | +0.0047 | +0.0056 | +0.0057 | +0.0102 | +0.0147 |

### KL10 Teacher-margin bins: clean Δ vs F0

| seed | arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| L2 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2 | F1 | +0.0302 | +0.0757 | +0.0000 | +0.0267 | +0.0393 |
| L2 | F2 | +0.0101 | -0.0108 | -0.0372 | +0.0160 | +0.0181 |
| L2 | F3 | +0.0101 | +0.0108 | +0.0319 | +0.0642 | +0.0846 |
| L4 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L4 | F1 | +0.0441 | +0.0938 | +0.0514 | +0.0390 | +0.0912 |
| L4 | F2 | +0.0490 | +0.0156 | +0.0057 | +0.0195 | +0.0091 |
| L4 | F3 | +0.0441 | +0.0625 | +0.0229 | +0.0829 | +0.1033 |

### KL10 Teacher-margin bins: robust Δ vs F0

| seed | arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| L2 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2 | F1 | +0.0000 | +0.0000 | +0.0000 | -0.0107 | +0.0030 |
| L2 | F2 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0060 |
| L2 | F3 | +0.0000 | +0.0000 | -0.0053 | -0.0107 | +0.0151 |
| L4 | F0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L4 | F1 | +0.0000 | +0.0052 | +0.0229 | +0.0146 | +0.0274 |
| L4 | F2 | +0.0000 | +0.0000 | +0.0057 | +0.0244 | +0.0213 |
| L4 | F3 | +0.0000 | +0.0000 | +0.0114 | +0.0146 | +0.0152 |

The machine JSON also contains the same Q1--Q5 paired effects at epochs 84 and 89, train-side direct/spillover effects, quantile boundaries, stable-ID hashes, and all endpoint lineage hashes.

## Next decision

Do not launch lambda/floor/cap sensitivity automatically. Human review is required after comparing F2 (margin-only) and F3 (full combination).
