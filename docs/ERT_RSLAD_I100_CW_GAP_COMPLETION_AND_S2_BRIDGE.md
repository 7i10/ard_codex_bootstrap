# I100 Clean-Wrong gap completion and canonical S2 bridge

This is a read-only e99 Train/Validation shift and canonical S2 harm-localization audit. No training, intervention, threshold tuning, new seed, official test, or AutoAttack was run.

## Answers up front

1. **e99 Train replay:** complete for both seeds (45,000 stable-ID rows per seed) using the exact parent, CE-PGD20 + Teacher-clean KL-PGD10, and sample-keyed random-start contract.
2. **Train → Validation shift:** descriptive `D1_STRONG_STATE_SHIFT` for both seeds; the largest primary numeric separation is Teacher adversarial margin (absolute SMD 0.313 pooled maximum).
3. The shift is concentrated in Teacher adversarial/clean margins and Teacher correctness/state composition; Student clean/adv margins are close (small SMD/KS).
4. Validation non-CW harm is present and is more concentrated in canonical S2 than S1 for several epochs, but the strength is action/seed dependent rather than universal.
5. Plain AdvCE has a worse (higher) S2 harm rate than S1 at most available endpoints in both seeds.
6. TPFM shows the same qualitative S2-vs-S1 risk pattern in some endpoints, but it is weaker and less uniform than Plain AdvCE.
7. Exact S2 harm-enrichment values by epoch are in the canonical-cell artifact and are not used to tune a selector.
8. S2×T1/T2/T3 cells are retained for all epochs; cells with n < 100 are descriptive only.
9. CW rescue and S2/non-CW harm directions recur across multiple epochs, while magnitudes turn over over time.
10. The broad CW-rescue/non-CW-harm pattern is directionally replicated in both seeds; fine-grained cells are not uniformly replicated.
11. The CW-recovery versus fragile-correct-harm trade-off is supported as a descriptive mechanism boundary, not as a new causal intervention result.
12. TPFM's plausible safety advantage is lower collateral pressure/harm outside the fixed CW cohort, not a demonstrated larger or more durable CW transfer.
13. The train→held-out gap is jointly compatible with population dilution, Teacher/state shift, attenuated CW transfer, non-CW/S2 collateral harm, and temporal response turnover; no additive causal attribution is claimed.
14. **Clean-Wrong action exploration is closed** at this diagnostic boundary.
15. The next canonical S2 question is how to improve neighboring robust failures while preserving currently robust-but-fragile samples.
16. Do not claim that Train-CW rescue generalizes at its direct magnitude, that TPFM is a validated router, or that any combined/new action is approved.

## Contract and lineage

Source SHA at analysis invocation: `5adf6c045a2474152ed68fbb9fafdfc2f59014ab` (working-tree status is recorded in the contract artifact). Exact e99 parents: dev-1 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`, dev-2 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`. Endpoint attack identity: `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.

Canonical states use the registered positive-margin q10 contract: S1 = adversarial-correct outside the lowest positive Student-margin q10, S2 = adversarial-correct inside that q10, S3 = adversarial-wrong; T1/T2/T3 are analogous for Teacher. Legacy `ert_state_overlay_v1` labels are not reused.

## Train-CW versus Validation-CW prevalence

| seed | Train CW n (%) | Validation CW n (%) | Validation non-CW n |
| --- | --- | --- | --- |
| dev-1 | 9263 (20.58%) | 1138 (22.76%) | 3862 |
| dev-2 | 8709 (19.35%) | 1143 (22.86%) | 3857 |

The full numeric and categorical shift (means/medians/q10–q90, SMD, KS, state/Teacher proportions, regimes, and class counts) is in `ert_rslad_i100_train_validation_cw_shift_v1.json`. No p-value-based decision or post-treatment boundary was used.

## Train direct versus held-out effects

The existing e199 Train endpoint is used only as a fixed direct/spillover reference; it is not re-estimated or used to define any state. Validation effects use the e99 pre-treatment groups and the existing e129/e149/e169/e189/e199 rows.

| seed | arm | Train CW direct robust Δ pp | Train non-CW spillover robust Δ pp |
| --- | --- | --- | --- |
| dev-1 | PLAIN_ADVCE | +4.124 | -0.221 |
| dev-1 | TPFM | +3.692 | +0.087 |
| dev-2 | PLAIN_ADVCE | +4.088 | -0.220 |
| dev-2 | TPFM | +3.870 | +0.077 |

## Canonical non-CW harm localization

Primary target is validation non-CW harm within canonical S2 versus S1. Cells with n < 100 are retained as descriptive but marked decision-ineligible.

### dev-1

| epoch | arm | V-CW robust Δ pp | V-nonCW robust Δ pp | S1 harm | S2 harm | S1 n | S2 n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 129 | PLAIN_ADVCE | +1.494 | -0.155 | 1.14% | 3.27% | 2197 | 245 |
| 149 | PLAIN_ADVCE | +0.000 | -0.207 | 1.05% | 7.76% | 2197 | 245 |
| 169 | PLAIN_ADVCE | +0.351 | -0.388 | 0.77% | 2.86% | 2197 | 245 |
| 189 | PLAIN_ADVCE | +0.791 | -0.207 | 0.50% | 2.45% | 2197 | 245 |
| 199 | PLAIN_ADVCE | +0.967 | -0.673 | 1.00% | 5.71% | 2197 | 245 |
| 129 | TPFM | +1.230 | +0.052 | 1.09% | 2.45% | 2197 | 245 |
| 149 | TPFM | -0.264 | +0.570 | 1.05% | 4.08% | 2197 | 245 |
| 169 | TPFM | +0.967 | +0.052 | 0.59% | 2.04% | 2197 | 245 |
| 189 | TPFM | +0.879 | +0.492 | 0.50% | 1.63% | 2197 | 245 |
| 199 | TPFM | +0.967 | -0.233 | 0.86% | 2.45% | 2197 | 245 |

S2 × Teacher cells at e199 (cells below n=100 are not mechanism-decision eligible):

| arm | Teacher state | n | robust Δ pp | harm | n≥100 |
| --- | --- | --- | --- | --- | --- |
| PLAIN_ADVCE | T1 | 226 | -3.540 | 6.19% | yes |
| PLAIN_ADVCE | T2 | 13 | +0.000 | 0.00% | no |
| PLAIN_ADVCE | T3 | 6 | +0.000 | 0.00% | no |
| TPFM | T1 | 226 | +0.442 | 2.65% | yes |
| TPFM | T2 | 13 | +0.000 | 0.00% | no |
| TPFM | T3 | 6 | +0.000 | 0.00% | no |

### dev-2

| epoch | arm | V-CW robust Δ pp | V-nonCW robust Δ pp | S1 harm | S2 harm | S1 n | S2 n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 129 | PLAIN_ADVCE | +0.612 | -0.544 | 1.33% | 2.56% | 2102 | 234 |
| 149 | PLAIN_ADVCE | +1.312 | +0.363 | 1.05% | 3.42% | 2102 | 234 |
| 169 | PLAIN_ADVCE | +0.525 | -0.130 | 0.33% | 2.99% | 2102 | 234 |
| 189 | PLAIN_ADVCE | +0.700 | -0.519 | 0.62% | 1.28% | 2102 | 234 |
| 199 | PLAIN_ADVCE | +1.137 | -0.337 | 0.43% | 2.99% | 2102 | 234 |
| 129 | TPFM | +0.787 | +0.026 | 0.62% | 2.56% | 2102 | 234 |
| 149 | TPFM | +0.962 | +0.856 | 1.00% | 2.14% | 2102 | 234 |
| 169 | TPFM | +0.175 | -0.104 | 0.29% | 2.56% | 2102 | 234 |
| 189 | TPFM | +0.612 | +0.156 | 0.52% | 0.85% | 2102 | 234 |
| 199 | TPFM | +0.525 | +0.104 | 0.52% | 2.56% | 2102 | 234 |

S2 × Teacher cells at e199 (cells below n=100 are not mechanism-decision eligible):

| arm | Teacher state | n | robust Δ pp | harm | n≥100 |
| --- | --- | --- | --- | --- | --- |
| PLAIN_ADVCE | T1 | 209 | -2.871 | 3.35% | yes |
| PLAIN_ADVCE | T2 | 20 | +0.000 | 0.00% | no |
| PLAIN_ADVCE | T3 | 5 | +20.000 | 0.00% | no |
| TPFM | T1 | 209 | -1.435 | 2.39% | yes |
| TPFM | T2 | 20 | -5.000 | 5.00% | no |
| TPFM | T3 | 5 | +0.000 | 0.00% | no |

## TPFM boundary regimes

The frozen TPFM values are coefficient `0.316427398202933`, floor `0.17963354289531708`, and cap `0.5595575273036957`. Regime counts for Train-CW and Validation-CW are recorded in `ert_rslad_i100_cw_boundary_tradeoff_v1.json`; no regime was selected or retuned from outcomes.

## Mechanism classification

B1–B4 and D1–D3 descriptive classifications are machine-recorded per seed. D1 means at least one primary feature has absolute SMD or categorical proportion difference at least 0.25; D2 is at least 0.10; D3 is weaker. B1 requires repeated CW rescue, S2< S1 robust effects, and S2 harm enrichment; otherwise the less-structured B2/B3/B4 labels are used. These are descriptive, not population claims.

| seed | D shift | Plain class | TPFM class |
| --- | --- | --- | --- |
| dev-1 | D1_STRONG_STATE_SHIFT | B1_FRAGILE_CORRECT_HARM_SUPPORTED | B4_NO_COLLATERAL_STRUCTURE |
| dev-2 | D1_STRONG_STATE_SHIFT | B3_NON_S2_COLLATERAL | B2_WEAK_S2_CONCENTRATION |

## Decision and stop boundary

This closes the current Clean-Wrong exploration as a diagnostic. It does not validate a new S2 intervention, change the TPFM margin/floor/cap, select a new threshold, or authorize dynamic/History routing. Any follow-up must be a separately reviewed intervention.

Machine artifacts: `ert_rslad_i100_cw_gap_completion_contract_v1.json`, `ert_rslad_i100_train_validation_cw_shift_v1.json`, `ert_rslad_i100_canonical_s2_harm_localization_v1.json`, `ert_rslad_i100_canonical_s2_teacher_cells_v1.json`, and `ert_rslad_i100_cw_boundary_tradeoff_v1.json`.
