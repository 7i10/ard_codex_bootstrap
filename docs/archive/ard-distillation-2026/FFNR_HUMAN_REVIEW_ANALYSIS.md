# FFNR人手判定とクラス偏りの分析

- contract: `ffnr_human_review_analysis_v1`
- replay epoch: `199`（各runのCE-PGD20観測）
- panel: `200`件、full replay: L2 `45000`件 / L4 `45000`件
- 画像panelは候補抽出済みの診断panelであり、ランダムなtest集合ではない。panelの率は記述統計、full replayの率はクラス別検証値として分けて読む。

## 結論

1. `possible_label_error`は0件で、今回のpanelから明白なラベルノイズの大量混入は確認できない。
2. `ambiguous`は25/200（12.5%）。クラス別ではdeerが7/20（35%）で最大、airplane/catが各4/20（20%）と続く。
3. ambiguous群の学生robust誤り率は64.0%、clear_matchは59.4%。教師adv誤り率は56.0%対53.7%で、今回のpanelだけではambiguousが誤りを強く説明するとは言えない。
4. full replayの学生robust誤りはbird/catが約45%、deer/dogが約40%で高い。これは人手ambiguous率（deer、cat）と一部整合するが、同一原因の証明ではない。
5. 教師adv混同行列では、bird→deer/frog、cat↔dog/frog、deer→frog、airplane→ship、truck→automobileが大きい。学生の誤予測先は入力Parquetに保存されていないため、学生について同じ混同行列はまだ作れない。

## 人手分類別のモデル誤り（panel条件付き）

| human group | n | student robust error | student clean error | teacher adv error | teacher clean error |
| --- | ---: | ---: | ---: | ---: | ---: |
| clear_easy | 136 | 58.8% | 28.7% | 52.2% | 23.5% |
| clear_hard | 39 | 61.5% | 43.6% | 59.0% | 43.6% |
| ambiguous | 25 | 64.0% | 36.0% | 56.0% | 32.0% |

## クラス別のpanel結果

| class | n | ambiguous | clear easy | clear hard | student robust error | teacher adv error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| airplane | 20 | 4 (20.0%) | 12 | 4 | 60.0% | 55.0% |
| automobile | 20 | 0 (0.0%) | 18 | 2 | 60.0% | 50.0% |
| bird | 20 | 1 (5.0%) | 15 | 4 | 55.0% | 50.0% |
| cat | 20 | 4 (20.0%) | 13 | 3 | 65.0% | 55.0% |
| deer | 20 | 7 (35.0%) | 10 | 3 | 65.0% | 50.0% |
| dog | 20 | 1 (5.0%) | 15 | 4 | 60.0% | 55.0% |
| frog | 20 | 3 (15.0%) | 10 | 7 | 65.0% | 65.0% |
| horse | 20 | 1 (5.0%) | 18 | 1 | 60.0% | 60.0% |
| ship | 20 | 2 (10.0%) | 14 | 4 | 55.0% | 50.0% |
| truck | 20 | 2 (10.0%) | 11 | 7 | 55.0% | 50.0% |

## full replayのクラス別誤り

各runは45,000件、epoch 199のstrong CE-PGD20観測です。

| class | L2 student robust error | L4 student robust error | L2 teacher adv error | L4 teacher adv error |
| --- | ---: | ---: | ---: | ---: |
| airplane | 20.9% | 18.9% | 14.8% | 14.8% |
| automobile | 13.0% | 11.8% | 7.8% | 7.6% |
| bird | 45.2% | 45.2% | 33.9% | 33.8% |
| cat | 44.9% | 45.8% | 34.8% | 35.1% |
| deer | 40.3% | 39.9% | 25.2% | 25.3% |
| dog | 39.7% | 40.3% | 31.4% | 31.6% |
| frog | 24.5% | 22.9% | 16.6% | 16.6% |
| horse | 17.7% | 18.2% | 12.9% | 12.8% |
| ship | 11.5% | 12.8% | 7.0% | 7.0% |
| truck | 16.4% | 18.2% | 12.8% | 13.1% |

## 教師advの主な混同（full replay）

| true → predicted | L2 | L4 |
| --- | ---: | ---: |
| bird → deer | 485 | 481 |
| dog → cat | 471 | 490 |
| cat → frog | 417 | 439 |
| deer → frog | 415 | 435 |
| bird → frog | 416 | 417 |
| cat → dog | 407 | 411 |
| airplane → ship | 348 | 353 |
| cat → deer | 327 | 321 |
| frog → deer | 292 | 297 |
| dog → frog | 287 | 298 |
| dog → deer | 285 | 270 |
| horse → deer | 233 | 236 |

### 教師cleanの主な混同

| true → predicted | L2 | L4 |
| --- | ---: | ---: |
| dog → cat | 247 | 247 |
| cat → frog | 187 | 187 |
| bird → deer | 160 | 160 |
| cat → deer | 156 | 156 |
| bird → frog | 153 | 153 |
| cat → dog | 148 | 148 |
| dog → deer | 136 | 136 |
| horse → deer | 109 | 109 |

## 限界と次の測定

- panelは候補panelであり、クラス別の人手ambiguous率をCIFAR-10全体のラベル品質率へ一般化しない。
- `student_robust_correct`とmarginは保存されているが、学生のclean/adv predicted classそのものは保存されていない。そのため学生のdog↔cat等の誤予測先は未確定である。必要なら次のGPU replayでstudent predicted class（clean/adv）を明示的に保存する。
- ambiguous群をそのまま除外・label correction・KD downweightのmaskには使わない。hidden cohort別のambiguous率と、teacher wrong-confidence・student persistent/recovered状態を結合してから介入を設計する。
- 教師混同行列の大きな組合せは、teacher targetの危険な領域候補として使えるが、介入効果を意味しない。
