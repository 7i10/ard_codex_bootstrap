# ERT / RSLAD Historical Treatment-Response Analysis

Analysis source Git SHA: `3bcacd922d92455b7731ff2467f3a7c60eeb2bd2`

## Executive answers

- Historical `ert_state_overlay_v1` S3 labels are retained as `historical_selection_state`; canonical `student_state` is not relabeled. The canonical table's Clean-Wrong is S2, while S3 is clean-correct/adversarial-wrong.
- Existing artifacts support row-level Stage A, broad Clean-Wrong, confirmatory T1/T2/T3, dynamic S3, and history-smoothed S3 comparisons. Missing or aggregate-only artifacts remain explicitly unavailable.
- `accuracy_delta` is computed and asserted as `rescue_rate - harm_rate`; margin deltas are stored separately.
- The primary ST1W direct response is a heterogeneous, fixed-cohort descriptive estimand; cross-seed predictive validity is reported without pooled fitting or future features.
- Dynamic state smoothing evidence cannot be treated as treatment utility: state stability and response utility are separate.

## Namespace and estimand contract

| namespace | meaning | use |
|---|---|---|
| historical_selection_state | legacy overlay mask semantics | retained only for source annotation |
| canonical_analysis_state | current Student/Teacher predicates | Stage A and feature joins |
| direct | fixed selected training cohort | paired response |
| spillover | training complement | paired response |
| held-out | independent validation endpoint | transfer diagnostic |

## Primary ST1W / response prediction

- L2: direct n=9889, robust Δ=0.0441, clean Δ=-0.0010.
- L4: direct n=9368, robust Δ=0.0221, clean Δ=-0.0010.

## Canonical state treatment examples

| seed | arm | split | cohort n | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| L2 | ST1W | train | 9889 | -0.0010 | 0.0441 | 0.0376 | 0.0386 | 0.1021 | 0.0580 |
| L2 | ST1W | validation | 5000 | -0.0030 | 0.0150 | 0.0324 | 0.0354 | 0.0540 | 0.0390 |
| L2 | ST2W | train | 2162 | -0.0060 | -0.0005 | 0.0833 | 0.0893 | 0.0356 | 0.0361 |
| L2 | ST2W | validation | 5000 | 0.0096 | 0.0142 | 0.0390 | 0.0294 | 0.0528 | 0.0386 |
| L2 | ST3K05 | train | 1790 | 0.0179 | -0.0028 | 0.1017 | 0.0838 | 0.0112 | 0.0140 |
| L2 | ST3K05 | validation | 5000 | 0.0090 | 0.0098 | 0.0370 | 0.0280 | 0.0524 | 0.0426 |
| L4 | ST1W | train | 9368 | -0.0010 | 0.0221 | 0.0363 | 0.0373 | 0.0907 | 0.0686 |
| L4 | ST1W | validation | 5000 | -0.0042 | 0.0004 | 0.0310 | 0.0352 | 0.0450 | 0.0446 |
| L4 | ST2W | train | 2138 | 0.0589 | -0.0028 | 0.1085 | 0.0496 | 0.0323 | 0.0351 |
| L4 | ST2W | validation | 5000 | 0.0028 | 0.0132 | 0.0338 | 0.0310 | 0.0516 | 0.0384 |
| L4 | ST3K05 | train | 1771 | 0.0807 | -0.0147 | 0.1389 | 0.0582 | 0.0085 | 0.0232 |
| L4 | ST3K05 | validation | 5000 | 0.0174 | 0.0128 | 0.0456 | 0.0282 | 0.0556 | 0.0428 |

## Clean-Wrong action family examples

| seed | arm | split | cohort n | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| L2 | C4 | train | 8623 | 0.0260 | 0.0041 | 0.0675 | 0.0415 | 0.0150 | 0.0109 |
| L2 | C4 | validation | 5000 | 0.0094 | -0.0032 | 0.0350 | 0.0256 | 0.0406 | 0.0438 |
| L2 | C10 | train | 8623 | 0.0545 | 0.0131 | 0.0869 | 0.0324 | 0.0224 | 0.0093 |
| L2 | C10 | validation | 5000 | 0.0124 | 0.0004 | 0.0388 | 0.0264 | 0.0466 | 0.0462 |
| L2 | C11 | train | 8623 | 0.0151 | 0.0070 | 0.0639 | 0.0488 | 0.0169 | 0.0100 |
| L2 | C11 | validation | 5000 | 0.0108 | -0.0060 | 0.0384 | 0.0276 | 0.0428 | 0.0488 |
| L2 | C12 | train | 8623 | 0.0288 | 0.0103 | 0.0753 | 0.0465 | 0.0203 | 0.0100 |
| L2 | C12 | validation | 5000 | 0.0040 | -0.0018 | 0.0382 | 0.0342 | 0.0472 | 0.0490 |
| L2 | C13 | train | 8623 | 0.0006 | 0.0058 | 0.0484 | 0.0478 | 0.0159 | 0.0101 |
| L2 | C13 | validation | 5000 | 0.0010 | 0.0034 | 0.0302 | 0.0292 | 0.0408 | 0.0374 |
| L4 | C4 | train | 8925 | 0.0279 | -0.0001 | 0.0715 | 0.0436 | 0.0138 | 0.0139 |
| L4 | C4 | validation | 5000 | -0.0052 | -0.0062 | 0.0264 | 0.0316 | 0.0412 | 0.0474 |
| L4 | C10 | train | 8925 | 0.0771 | 0.0059 | 0.1081 | 0.0310 | 0.0184 | 0.0124 |
| L4 | C10 | validation | 5000 | 0.0142 | -0.0056 | 0.0406 | 0.0264 | 0.0412 | 0.0468 |
| L4 | C11 | train | 8925 | 0.0124 | 0.0063 | 0.0592 | 0.0467 | 0.0184 | 0.0121 |
| L4 | C11 | validation | 5000 | -0.0144 | -0.0056 | 0.0256 | 0.0400 | 0.0406 | 0.0462 |
| L4 | C12 | train | 8925 | 0.0376 | 0.0097 | 0.0761 | 0.0384 | 0.0217 | 0.0120 |
| L4 | C12 | validation | 5000 | -0.0034 | -0.0084 | 0.0312 | 0.0346 | 0.0410 | 0.0494 |
| L4 | C13 | train | 8925 | 0.0151 | -0.0016 | 0.0589 | 0.0438 | 0.0130 | 0.0146 |
| L4 | C13 | validation | 5000 | 0.0014 | -0.0050 | 0.0300 | 0.0286 | 0.0426 | 0.0476 |

Cross-seed Ridge uses alpha=1.0 and only anchor79 state features (`mS`, available online risk proxies, `mT`, DeltaT and fixed interactions). It is a descriptive response-prediction test, not a route selector.

## Direct to held-out association

- clean_wrong_broad L2 epoch 84: n=15, Spearman=0.7035714285714285.
- clean_wrong_broad L4 epoch 84: n=15, Spearman=0.2076999433764589.
- confirmatory_t123 L2 epoch 84: n=3, Spearman=1.0.
- confirmatory_t123 L2 epoch 89: n=3, Spearman=1.0.
- confirmatory_t123 L2 epoch 94: n=3, Spearman=0.5.
- confirmatory_t123 L4 epoch 84: n=3, Spearman=-0.5.
- confirmatory_t123 L4 epoch 89: n=3, Spearman=0.5.
- confirmatory_t123 L4 epoch 94: n=3, Spearman=-0.5.
- history_s3 L2 epoch 84: n=3, Spearman=1.0.
- history_s3 L2 epoch 89: n=3, Spearman=1.0.
- history_s3 L2 epoch 94: n=3, Spearman=1.0.
- history_s3 L4 epoch 84: n=3, Spearman=0.5.
- history_s3 L4 epoch 89: n=3, Spearman=0.5.
- history_s3 L4 epoch 94: n=3, Spearman=1.0.
- stage_a L2 epoch 84: n=12, Spearman=0.35664335664335667.
- stage_a L4 epoch 84: n=12, Spearman=-0.512280701754386.

## Historical action evidence

Broad C10 direct rows available: 2 seed/endpoint cells. Plain AdvCE is not present as an isolated historical arm; C12 is MART-inspired and must not be equated with plain AdvCE.

## Temporal and generalization caveats

Endpoint horizons are sparse and differ by campaign. Direct improvement with non-positive held-out response is classified as a generalization failure, not as successful treatment. No historical response rule is promoted to I100 or a future router.

## Final decision

The available evidence is best reported as `RESPONSE_NOT_PREDICTABLE` for a deployable universal selector at this stage, with `HISTORY_RESPONSE_SIGNAL` / `TEACHER_RESPONSE_SIGNAL` retained as descriptive hypotheses only where cross-seed rows support them. Action-family failure and direct-to-held-out mismatch are explicitly recorded in the machine artifact.

## Reproducibility

- Inventory: `docs/experiments/ert_rslad_historical_treatment_inventory_v1.json`
- Unified rows (local only): `/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap/.cache/analysis/ert-rslad-historical-treatment-response-v1/outputs/response_rows.parquet`
- No training, attack regeneration, coefficient tuning, or new seed was run.

## Required question checklist

1. Historical S1/S2/S3 and Teacher T1/T2/T3 are kept in separate namespaces; legacy S3 is never relabeled as canonical S3.
2. Integrated row-level families are Stage A, broad Clean-Wrong, confirmatory T1/T2/T3, dynamic S3, and history-smoothed S3; aggregate-only gated/A7/blocked-ordering sources are inventory-only.
3. Missing row content is marked unavailable rather than reconstructed from aggregate metrics.
4. ST1W shows positive direct robust response in both seeds, but held-out transfer is small (L2 positive, L4 near zero), so rescue/harm is heterogeneous.
5. History and Teacher response prediction is weak for signed ST1W response; the strongest cross-seed family remains below a reliable selector threshold.
6. Direct-to-held-out association is descriptive and seed/family dependent (broad Clean-Wrong L2 is stronger than L4; Stage A reverses direction).
7. ST1W failure is attributable to direct-to-held-out attenuation, not to an absent direct rescue signal.
8. ST2/T3 and Clean-Wrong effects vary by arm and seed; no universal state treatment is supported.
9. The reliability-gated Clean-Wrong artifact is included as aggregate historical evidence; no post-hoc gate is fitted here.
10. Action rankings are not stable enough across seeds/families for a deployable rule.
11. Temporal response rows exist for confirmatory/dynamic/history horizons; transitions are descriptive and no persistence rule is selected.
12. History smoothing reduced switching in its own experiment but does not establish treatment utility.
13. Hard routing remains an experiment-specific mechanism, not a validated response selector.
14. No continuous History-conditioned loss is fitted or promoted by this read-only analysis.
15. History×Teacher interactions are represented only as fixed low-capacity descriptive features.
16. New distillation targets are not introduced; existing endpoint targets and margins are reused.
17. The next minimal I100 experiment cannot be specified from this retrospective table alone; no I100 coefficient transfer is allowed.
18. Teacher-only, Student-only, and interaction hypotheses remain descriptive unless independently confirmed.
19. Clean-Wrong C10/C12/C13 are treated as separate action families; C12 is not plain AdvCE.
20. Held-out is always an independent validation estimand, never a relabeled direct cohort.
21. Sample-level rows use stable IDs and label joins; no row-order surrogate IDs are used.
22. Bootstrap confidence intervals were not invented where the registered response artifact did not contain a preregistered bootstrap contract.
23. I100 is not treated as a historical response target or automatically modified.
24. Final decision: `R4 RESPONSE_NOT_PREDICTABLE` for a deployable universal action selector; retain family-specific hypotheses for human review only.
