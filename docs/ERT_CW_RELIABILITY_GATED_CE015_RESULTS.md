# ERT Clean-Wrong Reliability-Gated CleanCE 結果

## 結論

epoch 94 の held-out CE-PGD20では、CE20/KL10 reliability gateによる改善は両seedで再現しなかった。G2はL2でわずかにG0を上回るがL4では下回り、G3は両seedでG0を下回る。したがって、Teacher reliability gateを実用介入としてSUPPORTEDとは判定しない。

この結果はsample uncertaintyのbootstrapではなく、2 training seedの確認結果である。official testとAutoAttackは未実施。

## 完了・lineage

- 8 trajectories、48 endpoint（train/validation × 84/89/94）を完了。
- Git SHA: `8544fed4505d423cefe6e89ad789f45c52488aac`
- Parent SHA: L2 `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`、L4 `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`
- Endpoint: CE-PGD20 / $\epsilon=8/255$ / step $2/255$ / random start / eval mode
- Selector: epoch 79固定、`teacher_adv_margin > 0`
- Machine report: `docs/experiments/ert_cw_reliability_gated_ce015_v1.json`

## Selector counts

| seed | CW | CE20 reliable | KL10 reliable | RR | RU | UR | UU | CE/KL Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | 8623 | 2908 | 3452 | 2798 | 110 | 654 | 5061 | 0.7855 |
| L4 | 8925 | 3119 | 3729 | 3020 | 99 | 709 | 5097 | 0.7889 |

## Held-out CE-PGD20（validation 5,000）

| seed | epoch | G0 robust | G1 robust Δ | G2 robust Δ | G3 robust Δ | G0 clean | G1 clean Δ | G2 clean Δ | G3 clean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | 84 | 47.14 | +0.04 | -1.96 | -0.50 | 77.82 | +1.24 | -0.20 | +1.36 |
| L2 | 89 | 45.32 | -0.26 | -0.34 | +0.98 | 78.88 | +0.68 | +0.16 | +0.36 |
| L2 | 94 | 45.44 | -1.64 | +0.44 | -1.06 | 77.52 | +0.14 | -0.40 | -0.24 |
| L4 | 84 | 46.42 | -0.56 | -2.06 | -1.18 | 79.02 | +1.42 | +0.12 | +0.84 |
| L4 | 89 | 47.54 | +0.82 | -0.72 | +0.54 | 78.24 | +2.04 | +0.46 | +1.28 |
| L4 | 94 | 47.22 | -0.64 | -0.56 | -1.14 | 78.48 | +1.22 | +0.78 | +0.44 |

## epoch 94 の解釈

- L2: G2−G0 は robust **+0.44 pp**、G3−G0 は **−1.06 pp**。
- L4: G2−G0 は robust **−0.56 pp**、G3−G0 は **−1.14 pp**。
- G2−G1 は robust L2 **+2.08 pp** / L4 **+0.08 pp**。全CW処置よりは改善するが、G0を両seedで上回らない。
- G3−G1 は robust L2 **+0.58 pp** / L4 **−0.50 pp**。KL10 proxyの再現性は不十分。
- clean accuracyは全処置で上がる傾向があるが、robustnessとのtrade-offが残る。

## 固定CW cohortのpaired train endpoint効果（epoch 94）

これは処置対象内の直接効果であり、held-out robustnessの代替ではない。全CWでclean/robust net rescueは正方向でも、validationへ一貫して波及していない。

| seed | arm | CW clean net rescue | CW robust net rescue | CE20 robust net rescue | KL10 robust net rescue |
|---|---|---:|---:|---:|---:|
| L2 | G1 | +5.64% | +0.37% | +0.86% | +0.67% |
| L2 | G2 | +2.44% | +0.79% | +2.13% | +1.74% |
| L2 | G3 | +3.53% | +0.32% | +0.62% | +0.70% |
| L4 | G1 | +8.67% | +1.60% | +3.91% | +3.33% |
| L4 | G2 | +5.46% | +1.62% | +4.17% | +3.43% |
| L4 | G3 | +4.48% | +1.28% | +3.33% | +2.71% |

## 判定

- Gating confirmed: **No**（G2/G3が両seedでepoch94 held-out robust > G0を満たさない）。
- Oracle only: **No**（CE20 G2も両seedで安定しない）。
- KL10 practical selector: **No evidence**（G3は両seedでG0を下回る）。
- Short-term signal: L2/L4で一部horizonに正差があるが、epoch94のcross-seed durabilityを満たさない。
- 次のthreshold変更、selector再調整、dynamic routing、new seed、official test、AutoAttackは自動開始しない。

## 注意

旧G0試行のepoch94不足・W&B resume衝突ログは残るが、今回採用した`G0_BASE_r4`以降の8 trajectoryは全て完了し、今回の結果表には含めていない。
