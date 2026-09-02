# I100 Clean-Wrong Held-out Generalization Gap Audit

## 結論

既存のI100 long-horizon endpointsと、epoch 99 parentから新たに生成した validation-only feature replay を stable ID で結合した。新しいtraining、係数・threshold tuning、または追加 intervention は行っていない。

主因は単純な population dilution だけではない。epoch 199 では、validation Clean-Wrong (V-CW) 自体は両actionで robust rescue が正方向だが、validation non-CW の robust effect が Plain AdvCE で負、TPFM でも小さく負〜中立である。この collateral effect と train→held-out の generalization gap が、約 +4 pp の train direct rescue を overall held-out へ移さない主な説明である。V-CW rescue は early endpoint から late endpoint まで完全には固定されず、ただし late まで効果が消滅する単純な temporal-decay だけでもない。

## Lineage / scope

- analysis source: `89f17818a3ee7d899c9a45cee57606aa60e9c93f`
- treatment source: `c6032f9dc09f938fd0b9fb87379cf16c3f0f26bb`
- exact e99 I100 parents: dev-1 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`, dev-2 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`
- endpoint: CE-PGD20, identity `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- fixed train CW masks: dev-1 9,263, dev-2 8,709 (registered action-transfer mask artifact)
- validation e99 state replay: CE20/KL10, 5,000 IDs per seed; no optimizer/scheduler/state update
- outcome rows available: e129/e149/e169/e189/e199; e114 row-level endpoint files were not present in the current local inventory and are reported as unavailable rather than imputed.

## Pre-treatment population

| seed | train CW | validation CW | validation non-CW |
| --- | ---: | ---: | ---: |
| dev-1 | 9,263 (20.6%) | 1,138 (22.8%) | 3,862 |
| dev-2 | 8,709 (19.4%) | 1,143 (22.9%) | 3,857 |

Validation CW prevalence is not small enough for dilution alone to explain the gap, although any CW effect is necessarily downweighted to roughly 23% in the overall validation mean.

## Primary state decomposition: robust accuracy effects (pp)

Effects are treatment minus I100_CONTROL, computed by stable-ID pairing. `V-CW` and `V-nonCW` are fixed from epoch-99 Student clean correctness; the outcome never defines the groups.

| seed | epoch | action | train direct CW | V-CW | V-nonCW | V-overall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| dev-1 | 129 | Plain AdvCE | +4.12* | +1.494 | -0.155 | +0.220 |
| dev-1 | 129 | TPFM | +3.69* | +1.230 | +0.052 | +0.320 |
| dev-1 | 149 | Plain AdvCE | +4.12* | +0.000 | -0.207 | -0.160 |
| dev-1 | 149 | TPFM | +3.69* | -0.264 | +0.570 | +0.380 |
| dev-1 | 169 | Plain AdvCE | +4.12* | +0.351 | -0.388 | -0.220 |
| dev-1 | 169 | TPFM | +3.69* | +0.967 | +0.052 | +0.260 |
| dev-1 | 189 | Plain AdvCE | +4.12* | +0.791 | -0.207 | +0.020 |
| dev-1 | 189 | TPFM | +3.69* | +0.879 | +0.492 | +0.580 |
| dev-1 | 199 | Plain AdvCE | +4.12 | +0.967 | -0.673 | -0.300 |
| dev-1 | 199 | TPFM | +3.69 | +0.967 | -0.233 | +0.040 |
| dev-2 | 129 | Plain AdvCE | +4.09* | +0.612 | -0.544 | -0.280 |
| dev-2 | 129 | TPFM | +3.87* | +0.787 | +0.026 | +0.200 |
| dev-2 | 149 | Plain AdvCE | +4.09* | +1.312 | +0.363 | +0.580 |
| dev-2 | 149 | TPFM | +3.87* | +0.962 | +0.856 | +0.880 |
| dev-2 | 169 | Plain AdvCE | +4.09* | +0.525 | -0.130 | +0.020 |
| dev-2 | 169 | TPFM | +3.87* | +0.175 | -0.104 | -0.040 |
| dev-2 | 189 | Plain AdvCE | +4.09* | +0.700 | -0.519 | -0.240 |
| dev-2 | 189 | TPFM | +3.87* | +0.612 | +0.156 | +0.260 |
| dev-2 | 199 | Plain AdvCE | +4.09 | +1.137 | -0.337 | +0.000 |
| dev-2 | 199 | TPFM | +3.87 | +0.525 | +0.104 | +0.200 |

`*` train direct is the e199 fixed-CW effect and is shown as a reference, not re-estimated at each validation epoch. For every available epoch, the weighted identity

$$
\Delta R_{overall}=\frac{|V\text{-}CW|}{5000}\Delta R_{V\text{-}CW}+\frac{|V\text{-}nonCW|}{5000}\Delta R_{V\text{-}nonCW}
$$

matches the paired overall effect to numerical tolerance.

At e199, V-CW clean effects are +3.251/+2.187 pp (Plain) and +2.548/+1.750 pp (TPFM) for dev-1/dev-2; V-nonCW clean effects are mildly negative (−0.078/−0.130 pp and −0.181/−0.104 pp). Thus clean recovery is present in the held-out CW subtype, but it does not imply robust transfer.

## Train direct vs held-out CW efficiency

$$E_{CW}=\Delta R_{V\text{-}CW}/\Delta R_{train\ direct\ CW}$$

is descriptive only:

| seed | Plain AdvCE | TPFM |
| --- | ---: | ---: |
| dev-1 | 0.234 | 0.262 |
| dev-2 | 0.278 | 0.136 |

The direct train gain is therefore only partly expressed in the held-out CW population, especially for TPFM in dev-2.

## Temporal response

Only e129–e199 row-level endpoints are available. The e129 rescue→e199 rescue retention among validation-CW rescuers is:

| seed | Plain AdvCE | TPFM |
| --- | ---: | ---: |
| dev-1 | 27.8% (5/18) | 23.5% (4/17) |
| dev-2 | 14.3% (2/14) | 13.3% (2/15) |

This shows response turnover. It does not support a pure “early rescue disappears” explanation because e199 V-CW net effects are still positive in both seeds and both actions; it is better described as temporal instability plus partial transfer.

## Plain AdvCE vs TPFM action overlap at e199

Rescue-set Jaccard / harm-set Jaccard:

| seed | scope | rescue Jaccard | harm Jaccard | Plain-only rescue | TPFM-only rescue |
| --- | --- | ---: | ---: | ---: | ---: |
| dev-1 | train CW | 0.583 | 0.063 | 116 | 79 |
| dev-2 | train CW | 0.596 | 0.222 | 101 | 80 |
| dev-1 | validation CW | 0.500 | 0.333 | 5 | 5 |
| dev-2 | validation CW | 0.545 | 0.667 | 7 | 3 |

The two actions overlap only partially, particularly for train rescues. This supports heterogeneous response, but not a claim that combining them would be beneficial.

## Feature-shift audit

The required e99 **train** feature parquet was not present in the recoverable local artifacts. The validation e99 feature rows were complete and hash-bound, but a train-vs-validation e99 distribution comparison would otherwise mix epochs or invent rows. Therefore the following are explicitly unavailable:

- train-CW vs validation-CW e99 KS/SMD/quantile comparison;
- train/held-out TPFM floor–cap regime density comparison;
- e99 train-side Teacher-margin distribution comparison.

Available validation e99 features show that V-CW is substantially harder than V-nonCW (Teacher adversarial-correct rate 42.4%/47.2% vs 96.3%/95.7% in dev-1/dev-2). This is descriptive and does not substitute for the missing train-side comparison.

## Mechanism classification

- **Plain AdvCE:** primary `G6_MIXED_MECHANISM`; secondary `G2_TRAIN_LOCALIZATION` and `G3_COLLATERAL_GENERALIZATION_HARM`. Direct train rescue is large, V-CW transfer is smaller, and V-nonCW harm is negative at e199 in both seeds.
- **TPFM:** primary `G6_MIXED_MECHANISM`; secondary `G3_COLLATERAL_GENERALIZATION_HARM`. It has smaller direct gain but positive V-CW and near-neutral V-nonCW effects at e199; the advantage over Plain is modest and not a proof of generalization.
- `G1_POPULATION_DILUTION` is a contributor but not sufficient: V-CW prevalence is ~23% and V-nonCW effects materially change the overall result.
- `G4_TEMPORAL_RESPONSE_DECAY` is not sufficient: response turnover is high, but late V-CW effects remain positive.
- `G5_TRAIN_HELDOUT_STATE_SHIFT` remains plausible but untested because e99 train feature rows are unavailable.

## Answers to the requested questions

1. Held-out CW does improve at e199 for both actions in both seeds, but by only +0.53–+1.14 pp.
2. Overall loss is not dilution alone; non-CW collateral effects are important.
3. Plain AdvCE has held-out non-CW robust harm in both seeds. TPFM is less harmful but not uniformly positive.
4. The train spillover −0.22 pp pattern is reproduced directionally by Plain on validation non-CW (−0.673/−0.337 pp at e199).
5. TPFM’s candidate advantage is reduced non-CW harm, not a demonstrated larger CW transfer.
6. Train-vs-validation e99 difficulty shift cannot be tested from currently recoverable rows.
7. TPFM floor/cap regime shift is unavailable for the same reason.
8. Direct rescue is not sample-stable across endpoints; e129→e199 retention is 13–28%.
9. Plain and TPFM rescue sets overlap only moderately (validation-CW Jaccard 0.50/0.545).
10. Fragile non-CW concentration cannot be fully tested without a pre-treatment robust-margin split; aggregate non-CW harm is nevertheless present.
11. Class concentration was not used to select or tune any action; no causal class claim is made.
12. The data are compatible with a CW-recovery / fragile-correct-harm boundary trade-off.

## Forbidden claims / next-step boundary

Do not claim that a large direct rescue generalizes, that TPFM is a validated online selector, or that a combined Plain+TPFM action is beneficial. The next canonical S2 study may use this as a hypothesis about CW rescue versus fragile-correct harm, but no S2 intervention, TPFM modification, threshold search, or new training was started here.

Machine artifacts: [contract](experiments/ert_rslad_i100_cw_heldout_gap_contract_v1.json), [state decomposition](experiments/ert_rslad_i100_cw_heldout_state_decomposition_v1.json), [temporal response](experiments/ert_rslad_i100_cw_heldout_temporal_response_v1.json), [action overlap](experiments/ert_rslad_i100_cw_heldout_action_overlap_v1.json), and [feature-shift availability](experiments/ert_rslad_i100_cw_heldout_feature_shift_v1.json).
