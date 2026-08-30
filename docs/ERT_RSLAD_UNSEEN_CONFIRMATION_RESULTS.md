# ERT / RSLAD Unseen-Seed Confirmation Results

## 結論

3つの未使用 seed bundle (`confirm-a/b/c`) で、固定済みの
`CROPSHIFT 0--99 -> IDBH_WEAK 100--199` (`I100`) を確認した。
独立 CE-PGD20 endpoint の epoch 199 robust accuracy は、同一 bundle の
`CROP_SUFFIX` を3/3で上回った。training trajectory の final robust accuracy、
full AUC、post-100 AUC も3/3で同方向だった。

従って、今回の範囲では `I100` の trajectory 改善は再現された。ただし、
これは3 confirmation seedによる記述的確認であり、training-seed母集団への
有意性や普遍的な最適 switch epochを主張しない。clean accuracy はほぼ維持
されたが、I100が常にわずかに低い傾向も併記する。

## Lineage と評価契約

- production source SHA: `f4f8592290fae61f15a75bb4eed6c5244c5a690e`
- seed registry: [`ert_rslad_unseen_seed_confirmation_registry_v1.json`](experiments/ert_rslad_unseen_seed_confirmation_registry_v1.json)
- registry freeze SHA: `9b96c1afc59f8618aa5f46aa9f3f2f93c7ce5941`（production SHAの祖先）
- Teacher: Chen2021LTD WRN34-10, SHA256
  `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`
- arms: `BASE`, `CROP_PREFIX`, `CROP_SUFFIX`, `I100_SUFFIX`
- training attack: KL-PGD10（既存RSLAD契約）
- independent endpoint: eval-mode CE-PGD20, pixel `[0,1]`, $\epsilon=8/255$,
  step `2/255`, 20 steps, random start, hard-label CE
- endpoint split: fixed internal validation 5,000 samples
- endpoint attack identity SHA256:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- endpoint split identity SHA256:
  `16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4`
- official test / AutoAttack: 未実施

registryは事前freeze commit (`9b96c1a`) に作成され、実行時の source
(`f4f859`) はその子孫である。実行manifest、checkpoint、endpointはすべて
実行時 source SHAを記録しており、この差はtechnical/orchestration追加を含む
後続commitである。seed bundleやscientific armを結果後に変更していない。

## Independent CE-PGD20 endpoint（primary）

値は validation accuracy（%）。`e199` は checkpoint payload epoch 198 に
対応する canonical scientific epoch 199 である。

| bundle | BASE clean | BASE robust | CROP_SUFFIX clean | CROP_SUFFIX robust | I100 clean | I100 robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| confirm-a | 85.68 | 58.22 | 85.68 | 60.40 | 85.62 | 61.18 |
| confirm-b | 86.16 | 58.58 | 86.18 | 59.88 | 85.96 | 60.56 |
| confirm-c | 86.10 | 58.18 | 86.46 | 60.28 | 86.00 | 60.90 |
| mean | 85.98 | 58.33 | 86.11 | 60.19 | 85.86 | 60.88 |

I100 - CROP_SUFFIX の epoch 199 差分は、robust がそれぞれ
`+0.78`, `+0.68`, `+0.62` pp（平均 `+0.69` pp）、clean が
`-0.06`, `-0.22`, `-0.46` pp（平均 `-0.25` pp）だった。

epoch 149 endpoint でも robust は、BASE `56.43%`、CROP_SUFFIX `57.18%`、
I100 `57.42%` であり、I100 は CROP_SUFFIX を平均 `+0.24` pp上回った。
全30 endpoint JSON（BASE 12、CROP_SUFFIX 6、I100 6、prefix診断6）が存在し、
全て5,000 rows、source SHA、attack identity、split identityが一致した。

## Training trajectory（既存 validation metric）

これは各training runが記録した内部 validation trajectory の要約であり、
独立 endpoint の代替ではない。AUCはrepoの正規化trapezoidal convention。

| bundle | arm | final clean | final robust | best robust (epoch) | full AUC | post100 AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| confirm-a | BASE | 86.28 | 58.14 | 58.38 (163) | 0.504912 | 0.570720 |
| confirm-a | CROP_SUFFIX | 86.16 | 60.30 | 60.64 (195) | 0.516818 | 0.585502 |
| confirm-a | I100_SUFFIX | 86.02 | 60.74 | 61.10 (198) | 0.518675 | 0.589259 |
| confirm-b | BASE | 86.40 | 58.58 | 58.88 (162) | 0.505669 | 0.573018 |
| confirm-b | CROP_SUFFIX | 85.88 | 60.06 | 60.20 (192) | 0.517028 | 0.582949 |
| confirm-b | I100_SUFFIX | 85.98 | 60.60 | 60.60 (199) | 0.518391 | 0.585696 |
| confirm-c | BASE | 86.10 | 58.76 | 58.92 (191) | 0.505718 | 0.573615 |
| confirm-c | CROP_SUFFIX | 86.12 | 60.12 | 60.44 (188) | 0.517080 | 0.586008 |
| confirm-c | I100_SUFFIX | 85.96 | 60.76 | 60.90 (180) | 0.518460 | 0.588782 |

3 bundle平均は次の通り。

| arm | final clean | final robust | full AUC | post100 AUC |
| --- | ---: | ---: | ---: | ---: |
| BASE | 86.26 | 58.49 | 0.505433 | 0.572451 |
| CROP_SUFFIX | 86.05 | 60.16 | 0.516975 | 0.584820 |
| I100_SUFFIX | 85.99 | 60.70 | 0.518509 | 0.587912 |

I100 - CROP_SUFFIX は、final robust `+0.54/+0.54/+0.64` pp、full AUC
`+0.186/+0.136/+0.138` pp、post100 AUC `+0.376/+0.275/+0.277` ppで、
いずれも3 bundle同方向だった。final clean差は
`-0.14/+0.10/-0.16` ppで、robust改善に対する小さなclean trade-offがある。

`CROP_PREFIX` は suffix fork 前の共通prefix診断であり、各bundleの epoch 99
endpoint robust は `47.34% / 43.70% / 46.84%`。prefixを最終armとして比較する
ものではない。

## 事前に定めた判断との対応

- **I100 vs CROP_SUFFIX**: independent epoch 199 robust は3/3で改善。
- **trajectory objective**: final robust、full AUC、post100 AUC が3/3で改善。
- **clean guardrail**: I100 - CROP_SUFFIX の最大低下は independent endpointで
  `-0.46` pp、training metricで `-0.16` pp。`-1.0` pp guardrailには該当しない。
- **seed consistency**: 3 bundleでrobust差の符号は一致。
- **population inference**: 3 seedの記述統計のみ。bootstrap CIやtraining-seed
  有意性検定はこの confirmation では行っていない。

## Orchestration / technical audit

endpoint campaign は現在完了しており、実行中プロセスは0件。v2/v3で発生した
off-by-one、Ferretの同時 `git fetch` lock race、v4 helper path doubling は
technical failureであり、scientific source/config/seed/attackを変更していない。
失敗した2 endpointはv5で同一identityのまま再実行し、Ferret GPU1で正常終了した。

既存の監査記録は
[`ERT_RSLAD_UNSEEN_CONFIRMATION_ORCHESTRATION_AUDIT.md`](ERT_RSLAD_UNSEEN_CONFIRMATION_ORCHESTRATION_AUDIT.md)、
機械可読の全結果とendpoint/checkpoint/row hashは
[`ert_rslad_unseen_confirmation_results_v1.json`](experiments/ert_rslad_unseen_confirmation_results_v1.json)
に保存した。W&Bはmetrics-onlyで、model / full run-bundleはアップロードしていない。

## 判定と残作業

今回の有限 confirmation set では、`I100` を incumbent として維持する根拠が得られた。
追加の timing sweep、再training、official test、AutoAttackは自動開始しない。

残る研究課題は、必要に応じて既存の5 seed（development 2 + confirmation 3）を
用いた sample-level stochasticity / local divergence の記述分析である。今回の
compact endpoint集計だけから、その分析の結果を推測しない。
