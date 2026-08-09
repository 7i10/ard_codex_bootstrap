# FF/NR 次段階エビデンス結果

この報告はtrain splitの既存CE-PGD20 replayとonline stateだけを使ったCPU解析です。official CIFAR-10 test、AutoAttack、新規trainingは実行していません。

## A. Future Failure seed agreement

primaryは全45,000 stable-ID universe、majority/allは別endpointです。

| endpoint | L2 count | L4 count | raw Jaccard | null mean | observed-null | kappa | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 12343 | 12348 | 0.8569 | 0.1590 | 0.6979 | 0.8938 | 0.9172 | 0.9248 |
| all | 11369 | 11421 | 0.8464 | 0.1449 | 0.7014 | 0.8886 | 0.9172 | 0.9248 |

Permutation null固定: seed=3102/3103相当、10,000 fixed-count null samples（ID permutationと超幾何分布が同値）。paired bootstrapは2,000回で、これはtraining-seed CIではありません。
majorityのpaired 95% CI: Jaccard [0.8509, 0.8630]、kappa [0.8891, 0.8986]。

## B. Cross-seed teacher incremental information

各cellは片方のseedでfitし、もう片方でevaluateしたFF（anchor時点current-correct）です。standardizationはfit seedだけで計算しました。

| endpoint/anchor/direction | M AUROC | H AUROC | M+D AUROC | H+D AUROC | M+H AUROC | M+H+D AUROC | Δ(M+D−M) | Δ(H+D−H) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority:anchor39:L2_fit_L4_eval | 0.9267 | 0.9168 | 0.9913 | 0.9914 | 0.9403 | 0.9910 | +0.0646 | +0.0746 |
| majority:anchor39:L4_fit_L2_eval | 0.9379 | 0.9190 | 0.9922 | 0.9915 | 0.9479 | 0.9918 | +0.0542 | +0.0724 |
| majority:anchor59:L2_fit_L4_eval | 0.9239 | 0.9288 | 0.9888 | 0.9903 | 0.9426 | 0.9887 | +0.0649 | +0.0616 |
| majority:anchor59:L4_fit_L2_eval | 0.9343 | 0.9292 | 0.9912 | 0.9908 | 0.9494 | 0.9907 | +0.0569 | +0.0615 |
| majority:anchor79:L2_fit_L4_eval | 0.9336 | 0.9298 | 0.9921 | 0.9903 | 0.9492 | 0.9912 | +0.0585 | +0.0605 |
| majority:anchor79:L4_fit_L2_eval | 0.9237 | 0.9291 | 0.9904 | 0.9897 | 0.9467 | 0.9899 | +0.0667 | +0.0606 |
| all:anchor39:L2_fit_L4_eval | 0.9306 | 0.9179 | 0.9924 | 0.9926 | 0.9429 | 0.9921 | +0.0618 | +0.0747 |
| all:anchor39:L4_fit_L2_eval | 0.9437 | 0.9242 | 0.9935 | 0.9929 | 0.9530 | 0.9932 | +0.0498 | +0.0687 |
| all:anchor59:L2_fit_L4_eval | 0.9279 | 0.9290 | 0.9899 | 0.9912 | 0.9447 | 0.9896 | +0.0620 | +0.0622 |
| all:anchor59:L4_fit_L2_eval | 0.9382 | 0.9338 | 0.9921 | 0.9922 | 0.9528 | 0.9918 | +0.0539 | +0.0585 |
| all:anchor79:L2_fit_L4_eval | 0.9349 | 0.9317 | 0.9926 | 0.9914 | 0.9503 | 0.9918 | +0.0577 | +0.0597 |
| all:anchor79:L4_fit_L2_eval | 0.9297 | 0.9325 | 0.9922 | 0.9915 | 0.9510 | 0.9918 | +0.0625 | +0.0590 |

## C. Teacher-correct subset

Teacher-correct subsetのD連続値は、Teacher wrong/correctのbinary splitと分離して解釈します。

| endpoint/anchor/run | n | FF count | D AUROC | quartile FF rates |
| --- | ---: | ---: | ---: | --- |
| majority:anchor39:L2 | 23421 | 842 | 0.9859 | Q0=0.000, Q1=0.000, Q2=0.001, Q3=0.143 |
| majority:anchor39:L4 | 23608 | 872 | 0.9863 | Q0=0.000, Q1=0.000, Q2=0.001, Q3=0.147 |
| majority:anchor59:L2 | 23952 | 858 | 0.9854 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.143 |
| majority:anchor59:L4 | 24220 | 942 | 0.9845 | Q0=0.000, Q1=0.000, Q2=0.001, Q3=0.154 |
| majority:anchor79:L2 | 24323 | 940 | 0.9842 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.154 |
| majority:anchor79:L4 | 24474 | 913 | 0.9848 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.149 |
| all:anchor39:L2 | 23421 | 641 | 0.9865 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.109 |
| all:anchor39:L4 | 23608 | 685 | 0.9872 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.116 |
| all:anchor59:L2 | 23952 | 631 | 0.9856 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.105 |
| all:anchor59:L4 | 24220 | 735 | 0.9849 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.121 |
| all:anchor79:L2 | 24323 | 695 | 0.9857 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.114 |
| all:anchor79:L4 | 24474 | 695 | 0.9850 | Q0=0.000, Q1=0.000, Q2=0.000, Q3=0.113 |

## D. IRT

Bartoldsonの同一contract CE-PGD20 replayはローカルに存在しないため、[104,109,114] sensitivityを推測・再構成していません。新規GPU replayはこの段階では自動起動していません。

## 判定

この結果だけでTeacher dominanceを介入へ採用しません。cross-seed delta、Teacher-correct subset、chance-adjusted agreement、IRT artifact availabilityを確認後、初めてRoute A/Bの係数dry-runとepoch-79 short pilotへ進みます。

再現用の集計正本は [ffnr_next_evidence_v1.json](experiments/ffnr_next_evidence_v1.json)、実行入口は `scripts/analyze_ffnr_next_evidence.py` です。入力ParquetのSHA-256とendpoint、anchor、null/bootstrap設定をJSONへ保存しています。
