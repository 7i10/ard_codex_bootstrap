# ERT / RSLAD Student History Predictive Validity

## 結論（先に）

既存の固定I100 trajectoryを再利用し、追加training・追加attack生成なしで
BASEのreconvergence診断とStudent Historyの予測妥当性を解析した。confirm-cの
prefix checkpoint/metricsは、過去manifestに記録されたFerret上の既知パスからSHAを
照合して回収した。このため、5 seed (dev-1, dev-2, confirm-a/b/c) のdense
global trajectoryも完成した。

- BASEの5-seed robust SDはepoch 97で最大2.907 pp、epoch 199で0.272 ppへ縮小し、
  SD比は0.0937だった。低いt*時点のseedほどepoch 199までのgainが大きく、
  Spearmanは-1.000だった。この関係は記述的にcatch-upと分類するが、ROや
  reconvergenceの因果帰属ではない。
- cutoff=99の将来失敗率（150–199のrobust failure rate）について、dev-1/2で
  fitした固定Ridgeをconfirm-a/b/cへ評価した。P2（current correctness + current
  margin）に対するP4（P2 + frequency/forgetting/EMA/streak）のSpearman差は、
  +0.1560, +0.1551, +0.1587で3/3 positive、平均+0.1566だった。
  preregistered descriptive gate（3/3 positiveかつ平均≥0.02）は満たす。
- P3（history-only）はcutoff=99でSpearman 0.8432–0.8455とP2より高く、P4への
  追加改善は小さい。したがって今回のデータでは「現在のcorrectnessだけでなく、
  累積correctness・forgetting・margin EMA・streakを含む履歴」が将来failureを
  予測する証拠がある。これはHistory-conditioned Orderingを自動開始する許可ではない。
- 最終train sample-stateの5-seed集合では、all-seed correct=19.81%、all-seed
  wrong=12.62%、seed-sensitive=67.57%だった。global accuracyが近くても、
  sample populationは大きくseed依存である。

## Lineage / contract

- 解析時 source HEAD: 7c3320361965406f4acd5f5e9a756c9abb2751e8
- endpoint identity: CE-PGD20, pixel [0,1], epsilon=8/255, step 2/255,
  20 steps, random start, eval mode, hard-label CE
- endpoint attack SHA:
  7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2
- validation split SHA:
  16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4
- feature cutoffs: scientific epochs 49, 99, 149; primary cutoff 99
- primary target: 1 - (hits_199 - hits_149) / (seen_199 - seen_149)
- dev fit: dev-1, dev-2; confirmation evaluation: confirm-a/b/c
- Ridge alpha 1.0; standardization statistics are fit on dev rows only
- logistic secondary: L2 C=1.0; it predicts final robust-wrong status

全5 seed・4 cutoffのcheckpointはformat-v3、epoch boundary、pending-empty、45,000
stable-ID records、complete history statisticsを満たした。checkpoint filenameの
payload epochは科学epochより1小さいため、inventoryにはpayload epochとseenを併記
した。I100のepoch 49/99はCROPSHIFT prefixと同一の共有trajectoryである。

## BASE reconvergence / robust-overfitting diagnostic

BASEのfive-seed dense training metricからt*をデータ規則
(argmax of robust SD)で決め、t*=97となった。

| seed | robust@t* | robust@149 | robust@199 | gain t*→199 | gain 149→199 | best (epoch) | best−final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | 40.40% | 55.96% | 58.16% | +17.76 pp | +2.20 pp | 58.48% (179) | 0.32 pp |
| dev-2 | 46.86% | 56.36% | 58.52% | +11.66 pp | +2.16 pp | 58.64% (195) | 0.12 pp |
| confirm-a | 46.22% | 55.62% | 58.14% | +11.92 pp | +2.52 pp | 58.38% (163) | 0.24 pp |
| confirm-b | 46.20% | 56.20% | 58.58% | +12.38 pp | +2.38 pp | 58.88% (162) | 0.30 pp |
| confirm-c | 47.72% | 56.12% | 58.76% | +11.04 pp | +2.64 pp | 58.92% (191) | 0.16 pp |

Peak SD=2.907 pp (epoch 97), final SD=0.272 pp, peak range=7.32 pp, final
range=0.62 pp, macro ratio=0.0937。t*時点のrobustが低いseedほど後半gainが
大きいという単調な記述的関係であり、今回の解析だけで「robust overfittingが
原因」とは言わない。既存CE-PGD20 endpoint（99/149/199）の15 cellもattack/
stable-ID契約を満たして再確認した。

## Dense prefix recovery and global v2

confirm-a/b/cのCROPSHIFT prefixについて、Ferret上の以下の既知runから
epoch-metrics.jsonl（各100行）を回収した。

unseen-confirm-a-prefix-r2
unseen-confirm-b-prefix-r2
unseen-confirm-c-prefix-r2

回収SHAはそれぞれ、1c32130a...71391、9d6b88d8...81e24、
30d70a41...f0044で、各々epoch 0–99が連続している。新しいdense global artifact
では、CROPSHIFTとI100の全5 seed・epoch 0–199を、prefix + suffixとして結合した。
endpointからmetricを補完していない。

| arm | mean robust@49 | mean robust@99 | mean robust@149 | mean robust@199 | final SD | final range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 46.06% | 46.89% | 56.05% | 58.43% | 0.272 pp | 0.62 pp |
| CROPSHIFT | 47.40% | 46.64% | 57.33% | 59.78% | 0.532 pp | 1.18 pp |
| I100 | 47.40% | 46.64% | 57.66% | 60.74% | 0.118 pp | 0.32 pp |

これはI100が平均trajectoryを上方シフトし、最終global spreadも小さかったことを
示す記述結果である。variance reductionだけ、またはRNG sourceの単独因果とは解釈
しない。

## Student History semantics / inventory

sample_stateはSampleStateStore format-v3を使用し、各epochでstable IDごとに
1回のdetached FP32 observationをepoch boundaryでmergeする。今回のfeatureは
以下の固定定義で、classやstable IDは使っていない。

| family | features |
| --- | --- |
| P0 | current robust correctness |
| P1 | current last_margin |
| P2 | current correctness + current margin |
| P3 | inclusive robust-correct frequency, forgetting rate, margin EMA, current correct streak rate |
| P4 | P2 + P3 |

frequencyとforgettingは各checkpointの実際のseenで割った。未来のendpointや
validation metricをfeatureに使っていない。teacher_* snapshot fieldsはlineage
監査用で、予測featureには使用していない。

## Primary predictive validity (cutoff=99)

値は、dev-1/2でfitしたRidgeを各confirmation seedへ一度だけ適用した
Spearman相関（予測future failure vs observed future failure）である。

| family | confirm-a | confirm-b | confirm-c | mean |
| --- | ---: | ---: | ---: | ---: |
| P0 | 0.6231 | 0.6218 | 0.6205 | 0.6218 |
| P1 | 0.6875 | 0.6906 | 0.6865 | 0.6882 |
| P2 | 0.6875 | 0.6906 | 0.6865 | 0.6882 |
| P3 | 0.8432 | 0.8455 | 0.8450 | 0.8446 |
| P4 | 0.8434 | 0.8456 | 0.8452 | 0.8447 |

P4−P2は+0.1560/+0.1551/+0.1587（平均+0.1566）。3 seedすべて同方向で、
事前固定したgateを満たす。ただしP3→P4の差は約+0.0002であり、今回の証拠は
「履歴情報の価値」を支持する一方、「current stateを加えたP4がP3より必要」とは
支持しない。

### Cutoff sensitivity

| cutoff | P2 mean | P3 mean | P4 mean | P4−P2 mean |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 0.6867 | 0.8226 | 0.8235 | +0.1368 |
| 99 | 0.6882 | 0.8446 | 0.8447 | +0.1566 |
| 149 | 0.5943 | 0.9053 | 0.9070 | +0.3127 |

cutoff=49は5 seedで評価でき、cutoff=149は直後の150–199 future windowを評価する。
いずれもsecondary descriptive analysisで、cutoffやfeatureをoutcomeを見て選んで
いない。

## Secondary final robust-wrong classification

固定L2 logisticのconfirmation結果（ROC-AUC / PR-AUC / Brier）は次の通り。

| family | confirm-a ROC / PR | confirm-b ROC / PR | confirm-c ROC / PR |
| --- | ---: | ---: | ---: |
| P2 | 0.7271 / 0.6385 | 0.7279 / 0.6313 | 0.7236 / 0.6319 |
| P4 | 0.7794 / 0.7339 | 0.7811 / 0.7333 | 0.7762 / 0.7290 |

P4のBrierは0.1858/0.1857/0.1872で、P2の
0.2072/0.2073/0.2087より低い。これは将来failure rateの主解析を置換しない
補助結果である。

## Sample-level stochasticity context

I100のfinal train sample-stateで、5 seed中のrobust-correct回数kは次の通り。

k=0: 5,680 (12.62%)
k=1: 5,255
k=2: 6,463
k=3: 8,336
k=4: 10,352
k=5: 8,914 (19.81%)

seed-sensitive (1<=k<=4) は67.57%であり、global metricのreconvergenceは
sample populationの一致を意味しない。これはHistory featureがfuture failureを
予測する今回の結果と整合するが、Historyを使った介入効果の証明ではない。

## Recovery / artifact outputs

- ert_rslad_base_reconvergence_ro_diagnostic_v1.json
- ert_rslad_confirmation_prefix_dense_metric_recovery_v1.json
- ert_rslad_student_history_inventory_v1.json
- ert_rslad_student_history_predictive_validity_v1.json
- ert_rslad_five_seed_global_stochasticity_v2.json
- analysis implementation: scripts/analysis/analyze_ert_rslad_student_history.py
- dense telemetry validator: scripts/verify_dense_training_metrics.py

大容量checkpointや5-seed row matrixはGitへ追加せず、SHAと取得元をartifactへ記録
した。Ferretからの回収は既存checkpoint/metricsのコピーのみで、新規GPU jobは行って
いない。

## Interpretation and stop

今回の範囲では、Student Historyは将来のrobust failureを、current correctness/
marginより安定して予測する記述的証拠が3/3 confirmation seedで得られた。主効果は
P3でほぼ説明され、P4追加成分は小さい。したがって次段階で検討できる仮説は、
「将来failure-sensitiveなsample populationをHistoryで記述する」ことである。

ただし今回から次を主張しない。

- History-conditioned Orderingが性能を改善すること
- 予測相関が介入効果を保証すること
- RNG source単独の因果 attribution
- 5 seedを母集団の有意性検定として扱うこと
- I100が普遍的最適policyであること

新しいtraining、endpoint再生成、feature/threshold tuning、History intervention、
Ordering、official test、AutoAttackは開始していない。次の介入設計はこの停止点で
human reviewへ戻す。
