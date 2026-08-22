# ERT Clean-Wrong Teacher-Adaptive Margin RNG Stability Diagnostic

Status: completed read-only point-estimate diagnostic from the valid 0054 campaign.

The initial `--epochs 94` attempt from 0054 is excluded; only the valid `--epochs 95` v2 campaign and its epoch-84/89/94 endpoints are used.
No optimizer, scheduler, sample-state, checkpoint, or new training update was performed.

## Direct answers

- BASE continuation variance is reported from absolute endpoint metrics; the BASE delta in the 0054 report must not be used for this question.
- Treatment-effect continuation variance is reported after subtracting the matched same-replicate BASE.
- Existing epoch logs contain loss/accuracy/LR only; margin target, hinge, and regime quantities were not logged during training. Fixed-probe replay therefore localizes state/target/hinge divergence at epochs 84/89/94, not the unobserved per-step causal onset.
- Fixed replay uses the same deterministic probe IDs, epoch-aligned augmentation view, and KL-PGD10 initial-delta seed for R1/R2. Clip contraction is asserted.
- Gradient-vector comparison is not inferred from endpoint metrics; it is reported only if the separate focused no-update probe completes.

## Frozen contract

- Blocks: L2-R1, L2-R2, L4-R1, L4-R2; arms: N95, A100, N105; epochs: 84, 89, 94.
- Teacher-adaptive target: `clip(mT_adv, 0.03221710026264191, 0.13952550292015076)`; CleanCE is zero.
- Probe: 256 first sorted IDs from each registered epoch-79 Clean-Wrong mask.
- No population-level seed confidence interval is claimed; these are descriptive point estimates.

## Endpoint variance summary

The machine artifact contains absolute R1/R2 accuracy values, BASE absolute gaps, and paired treatment-effect gaps for each teacher, epoch, metric, and arm.

| teacher | epoch | metric | BASE gap (pp) | N95 effect gap (pp) | A100 effect gap (pp) | N105 effect gap (pp) |
|---|---:|---|---:|---:|---:|---:|
| L2 | 84 | validation robust | 0.840 | 1.180 | 1.280 | 1.900 |
| L2 | 89 | validation robust | 0.820 | 0.020 | 1.200 | 1.040 |
| L2 | 94 | validation robust | 1.880 | 2.060 | 1.540 | 0.460 |
| L4 | 84 | validation robust | 0.160 | 0.900 | 0.360 | 0.200 |
| L4 | 89 | validation robust | 0.280 | 1.320 | 1.560 | 1.400 |
| L4 | 94 | validation robust | 1.820 | 2.060 | 0.720 | 1.880 |

## Training-log divergence

The full epoch-80--94 absolute R1/R2 gap curves are stored in the machine artifact.  The available logs contain `train_loss`, clean/robust accuracy, validation metrics, and learning rates; margin-specific fields were absent and are not reconstructed from endpoint outcomes.

## Fixed-probe replay summary

For each teacher/arm/epoch, the machine artifact stores sample-wise absolute differences, R0--R3 transition counts, hinge disagreement, prediction disagreement, floor/cap/hinge boundary windows, and the target contraction check.

| teacher | arm | epoch | |ΔmT| | |Δtarget| | |Δdeficit| | regime disagreement | hinge disagreement | target contraction ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L2 | N95 | 84 | 0.024387 | 0.004024 | 0.060148 | 13.281% | 3.516% | 0.165 |
| L2 | N95 | 89 | 0.030256 | 0.005312 | 0.072229 | 21.094% | 4.688% | 0.176 |
| L2 | N95 | 94 | 0.039878 | 0.008658 | 0.098204 | 21.484% | 8.984% | 0.217 |
| L2 | A100 | 84 | 0.026362 | 0.005269 | 0.064531 | 16.406% | 4.297% | 0.200 |
| L2 | A100 | 89 | 0.029074 | 0.004687 | 0.074268 | 22.266% | 4.688% | 0.161 |
| L2 | A100 | 94 | 0.032420 | 0.006495 | 0.075695 | 18.359% | 5.078% | 0.200 |
| L2 | N105 | 84 | 0.032421 | 0.006797 | 0.076723 | 17.188% | 6.250% | 0.210 |
| L2 | N105 | 89 | 0.028572 | 0.005276 | 0.068832 | 19.141% | 3.906% | 0.185 |
| L2 | N105 | 94 | 0.037505 | 0.007534 | 0.094298 | 18.750% | 5.469% | 0.201 |
| L4 | N95 | 84 | 0.027851 | 0.005905 | 0.060082 | 22.656% | 3.906% | 0.212 |
| L4 | N95 | 89 | 0.032363 | 0.005682 | 0.075130 | 17.188% | 5.859% | 0.176 |
| L4 | N95 | 94 | 0.035166 | 0.008837 | 0.076749 | 17.188% | 5.469% | 0.251 |
| L4 | A100 | 84 | 0.032751 | 0.006174 | 0.075248 | 19.922% | 6.250% | 0.189 |
| L4 | A100 | 89 | 0.034191 | 0.006643 | 0.075735 | 18.359% | 4.688% | 0.194 |
| L4 | A100 | 94 | 0.029908 | 0.006318 | 0.077536 | 16.406% | 6.641% | 0.211 |
| L4 | N105 | 84 | 0.031080 | 0.006762 | 0.065627 | 19.141% | 6.250% | 0.218 |
| L4 | N105 | 89 | 0.032649 | 0.005422 | 0.070728 | 15.234% | 4.297% | 0.166 |
| L4 | N105 | 94 | 0.033192 | 0.007128 | 0.076353 | 16.406% | 7.422% | 0.215 |

## Descriptive link to final treatment-effect variance

Pearson/Spearman values use only six teacher×arm pairs and are exploratory; no p-value or causal claim is made.

| diagnostic at epoch | feature | Pearson | Spearman | n |
|---:|---|---:|---:|---:|
| 84 | final_effect_gap_vs_teacher_adv_margin_abs_diff | -0.744 | -0.657 | 6 |
| 84 | final_effect_gap_vs_target_abs_diff | -0.521 | -0.600 | 6 |
| 84 | final_effect_gap_vs_raw_deficit_abs_diff | -0.975 | -0.943 | 6 |
| 84 | final_effect_gap_vs_regime_disagreement_rate | -0.026 | +0.143 | 6 |
| 84 | final_effect_gap_vs_hinge_disagreement_rate | -0.710 | -0.543 | 6 |
| 89 | final_effect_gap_vs_teacher_adv_margin_abs_diff | +0.126 | +0.257 | 6 |
| 89 | final_effect_gap_vs_target_abs_diff | -0.319 | +0.257 | 6 |
| 89 | final_effect_gap_vs_raw_deficit_abs_diff | +0.222 | +0.257 | 6 |
| 89 | final_effect_gap_vs_regime_disagreement_rate | -0.092 | -0.257 | 6 |
| 89 | final_effect_gap_vs_hinge_disagreement_rate | +0.577 | +0.486 | 6 |

## No-update gradient probe

The focused probe uses the first 128 fixed IDs and the same deterministic KL-PGD10 seed protocol.  Cosines compare the full flattened Student parameter gradients between R1/R2; they are descriptive, not population inference.

| teacher | arm | epoch | base cosine | margin cosine | total cosine | margin/base norm gap |
|---|---|---:|---:|---:|---:|---:|
| L2 | N95 | 84 | 0.82955 | 0.81913 | 0.82927 | 0.004222 |
| L2 | N95 | 94 | 0.45515 | 0.60347 | 0.46194 | 0.012668 |
| L2 | A100 | 84 | 0.87877 | 0.73034 | 0.87454 | 0.008183 |
| L2 | A100 | 94 | 0.74893 | 0.76297 | 0.75151 | 0.000299 |
| L2 | N105 | 84 | 0.62595 | 0.58755 | 0.61528 | 0.027329 |
| L2 | N105 | 94 | 0.55340 | 0.71525 | 0.56651 | 0.028490 |
| L4 | N95 | 84 | 0.71881 | 0.75334 | 0.72267 | 0.001047 |
| L4 | N95 | 94 | 0.74440 | 0.77467 | 0.74930 | 0.001803 |
| L4 | A100 | 84 | 0.66896 | 0.76098 | 0.67552 | 0.003979 |
| L4 | A100 | 94 | 0.69026 | 0.81052 | 0.70021 | 0.012503 |
| L4 | N105 | 84 | 0.57817 | 0.70159 | 0.58232 | 0.014338 |
| L4 | N105 | 94 | 0.81929 | 0.80426 | 0.82031 | 0.003917 |

## Mechanism assessment

- BASE validation robust R1/R2 gaps span 0.16--1.88 pp across the reported epochs/teachers.  Treatment-effect gaps span 0.02--2.06 pp and are not uniformly larger than BASE gaps.  Baseline/general RSLAD stochasticity is therefore substantial, with incremental treatment variance only in some matched pairs.
- At epoch 84 the fixed probe already shows mean |ΔTeacher margin| 0.0244--0.0328, mean |Δtarget| 0.0040--0.0068, regime disagreement 13.3--22.7%, and hinge disagreement 3.5--6.2%.  This localizes propagation from already-diverged model states into target/hinge quantities, but does not identify the causal RNG stream.
- Hinge-boundary concentration is mixed: pre-registered hinge windows are small and the disagreement is not uniformly concentrated near them.  Clip contraction is satisfied, so the clip alone does not amplify Teacher-margin differences.
- The focused gradient probe gives weighted margin/base ratios in the range 0.049--0.092; cross-replicate base and margin cosines vary by pair, with no consistent margin-only direction collapse.  Effective-pressure variation is plausible but not isolated as the primary cause.
- Evidence ranking: (1) baseline/general RSLAD continuation stochasticity, (2) propagation of model divergence through Teacher target and hinge states, (3) possible effective-pressure variation, (4) hinge-switch instability as a mixed secondary mechanism.
- Recommended next direction: first address or characterize baseline/RSLAD continuation stability.  Do not automatically introduce target smoothing, adaptive lambda, a smooth hinge, or further floor/cap sweeps from this diagnostic alone.

## Interpretation boundary

This diagnostic can distinguish large BASE variance from additional treatment-effect variance and can show whether already-diverged checkpoints differ in target, regime, or hinge state.  It cannot identify which training RNG stream caused the divergence, and it cannot establish that target smoothing, adaptive weighting, or a smooth hinge improves performance.  Those remain human-review candidates only.

## Source

- Source Git SHA: `d2e3f65b264a88bfea3130129171d2d7a3d09ed4`; 0054 source: `10cfc5c277866e97a3853e2ca1cf9ec700fee990`.
- Machine artifact: `/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap/docs/experiments/ert_cw_margin_rng_stability_diagnostic_v1.json`.
- Fixed-probe replay artifact source SHA(s): 41e0b5398751e09d2b8665b67eaadf52fe615d9d; gradient artifact source SHA(s): 24ae2e8d109ad0c5e7c1da215f29dd03a2dd4b99.
