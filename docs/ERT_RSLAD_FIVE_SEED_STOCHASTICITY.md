# ERT / RSLAD Five-Seed Global & Sample-Level Stochasticity

## 結論（先に）

既存artifactだけで、5 seed (`dev-1`, `dev-2`, `confirm-a/b/c`) の
`BASE`、`CROPSHIFT`、`I100`を再学習なしに解析した。5,000件の固定 validation
row は、3 arm × 4 endpoint (`49, 99, 149, 199`) の全60 cellで揃い、stable-ID
とlabelが一致した。

- epoch 199 の endpoint robust 平均は `BASE 58.232%`, `CROPSHIFT 59.844%`,
  `I100 60.708%`。I100 は CROPSHIFT より `+0.864 pp` だった。
- epoch 199 の global training metric の5-seed SD/rangeは、`BASE
  0.272 pp / 0.62 pp`, `CROPSHIFT 0.532 pp / 1.18 pp`, `I100 0.118 pp /
  0.32 pp`。I100は平均を上方シフトし、最終global spreadも小さかった。
- それでも sample-level pairwise robust disagreement は `BASE 6.84%`,
  `CROPSHIFT 6.54%`, `I100 5.87%` 残った。globalの近さは sample population
  の同一性を意味しない。
- I100 の最終 rescue は seed間でほぼ同じsampleに集中していない。CROPSHIFT
  からの rescue が5/5 seedで起きたsampleは `1/5000`、lossは `2/5000` だった。

従って今回の記述的証拠は、I100を「randomnessを消した」とは示さない。主効果は
平均trajectoryの改善であり、併せて最終global spreadとsample disagreementも
低下した、という分解が妥当である。

## Lineage / contract

- 解析時 HEAD: `17cd2d3bb4ef28b1ec567cac49def2a37c023d3f`
- endpoint: CE-PGD20、pixel `[0,1]`、$\epsilon=8/255$、step `2/255`、20
  steps、random start、eval mode、hard-label CE
- attack identity:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- validation split identity:
  `16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4`
- `I100` の epoch 49/99 は `CROPSHIFT` と共有prefixで、独立runとして二重計上
  していない。
- endpoint欠損はなく、再生成・GPU使用・trainingは行っていない。
- official test / AutoAttack / bootstrapによるseed有意性推論は行っていない。

可用性の制約として、confirmation seedのCROPSHIFT/I100についてepoch 0--99の
dense training metricsは保存されていなかった。そこは欠損のまま扱い、endpoint
から補完していない。BASEは全5 seed、post-100 suffixは全5 seedでdense metricsを
保持している。

## Artifact inventory

全5 seed × 3 arm × 4 endpointのrow artifactは各5,000行で、ID/label joinに成功
した。availabilityと各row SHAは
[`ert_rslad_five_seed_artifact_inventory_v1.json`](experiments/ert_rslad_five_seed_artifact_inventory_v1.json)
に記録した。

## Global trajectory（training metric）

epochごとの5-seed平均・SD・rangeは機械artifactの `by_epoch` に保存した。
epoch 199の要約は以下の通り。

| arm | mean clean | mean robust | robust SD | robust range |
| --- | ---: | ---: | ---: | ---: |
| BASE | 86.308% | 58.432% | 0.272 pp | 0.62 pp |
| CROPSHIFT | 86.144% | 59.780% | 0.532 pp | 1.18 pp |
| I100 | 86.020% | 60.740% | 0.118 pp | 0.32 pp |

BASEは全200 epochで5 seedが揃い、peak SDは epoch 97 の `2.907 pp`、final SDは
`0.272 pp`（比率 `0.094`）だった。CROPSHIFT/I100はconfirmation prefixの
dense rowsがないため、全期間の5-seed peakから厳密なmacro reconvergenceを主張
しない。post-100ではI100の最終SD `0.118 pp` と、CROPSHIFTの `0.532 pp` が
観測された。I100の5-seed pairwise mean absolute gapも post-100 共通区間で
`0.368 pp`、CROPSHIFTは `0.512 pp` だった。

stage別の平均値・coverage、pairwise distance、peak/final dispersionは
[`ert_rslad_five_seed_global_stochasticity_v1.json`](experiments/ert_rslad_five_seed_global_stochasticity_v1.json)
に保存した。図の基礎データは同artifactとcache内CSV/SVGにある。

## Sample-level robust population

endpoint rowから計算した5-seed平均（accuracy）とseed間指標は次の通り。

| epoch | arm | robust mean | clean mean | pairwise disagreement | Jaccard |
| ---: | --- | ---: | ---: | ---: | ---: |
| 49 | BASE | 45.744% | 76.268% | 17.432% | 0.6802 |
| 49 | CROPSHIFT/I100 | 46.404% | 77.328% | 16.536% | 0.6976 |
| 99 | BASE | 46.212% | 77.436% | 16.672% | 0.6945 |
| 99 | CROPSHIFT/I100 | 47.392% | 78.304% | 16.592% | 0.7020 |
| 149 | BASE | 56.456% | 84.848% | 9.104% | 0.8508 |
| 149 | CROPSHIFT | 57.244% | 85.032% | 9.756% | 0.8430 |
| 149 | I100 | 57.560% | 84.696% | 9.620% | 0.8458 |
| 199 | BASE | 58.232% | 86.176% | 6.840% | 0.8891 |
| 199 | CROPSHIFT | 59.844% | 86.132% | 6.544% | 0.8963 |
| 199 | I100 | 60.708% | 85.888% | 5.868% | 0.9078 |

epoch 199のrobust-frequency $k_i$（5 seed中correctな回数）は以下の通り。

| arm | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | entropy mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 1,737 | 202 | 149 | 145 | 212 | 2,555 | 0.0810 |
| CROPSHIFT | 1,685 | 168 | 145 | 149 | 209 | 2,644 | 0.0773 |
| I100 | 1,662 | 181 | 113 | 134 | 182 | 2,728 | 0.0696 |

従って I100 では all-seed robust (`k=5`) が `54.56%`、all-seed non-robust
(`k=0`) が `33.24%`、seed-sensitive (`1<=k<=4`) が `12.20%` だった。BASEの
`51.10% / 34.74% / 14.16%`、CROPSHIFTの `52.88% / 33.70% / 13.42%` より、
populationのseed依存部分は小さい方向である。

## Margin stability

epoch 199の adversarial probability margin では、seed内SDの平均が
`BASE 0.03745`、`CROPSHIFT 0.03531`、`I100 0.03151`。margin sign agreementは
それぞれ `85.84% / 86.58% / 87.80%`、seed間rank correlationは
`0.9726 / 0.9761 / 0.9810` だった。I100はmarginのばらつきも低いが、これは
deterministicなRNG原因の同定ではなく記述的な差である。

## Rescue / loss consistency

epoch 199の同一stable IDについて、I100 vs CROPSHIFTのrobust rescue回数（5 seed
中何回rescueされたか）は、`0/1/2/3/4/5 = 4514/343/118/22/2/1`。lossは
`4666/255/59/13/5/2` だった。したがって平均robust向上はあるが、毎seedで同じ
sampleを救う単純な一様効果ではない。CROPSHIFT vs BASEの対応表も
`rescue_consistency.csv` として生成した。

## 研究質問への回答

1. **Final variability:** global training metricのSD/rangeは上表の通り。I100が
   最小だった。
2. **途中のspread:** BASEでは epoch 97 の2.907 ppからfinal 0.272 ppへ縮小。
   他armのconfirmation prefixは欠損しているため、全期間の5-seed比較はしない。
3. **Macro reconvergence:** BASEには明瞭な縮小、post-100 I100にも小さいfinal
   spreadが見える。ただし全arm・全期間の厳密な5-seed比率ではない。
4. **Globalが近くてもsample disagreement:** はい。I100でも最終5.868%が残る。
5. **Final population差:** I100のk=5が54.56%、k=0が33.24%。同一robust集合ではない。
6. **arm比較:** I100のdisagreement/Jaccard/entropyは3 arm中最も安定方向。
7. **Randomness reductionか:** 「抑制した」と因果的には言わない。低い最終SDと
   disagreementという記述的整合性はある。
8. **主効果:** I100はまず平均robust trajectoryを上方シフトし、同時に最終spread
   も小さかった。variance reduction単独とは解釈しない。
9. **Margin variability:** I100で最小方向、sign agreement/rank correlationは最大方向。
10. **stable subsets:** 上記k histogramの通り。
11. **I100 rescue consistency:** 5/5 rescueは1 sampleのみで、seed-specificなrescueが
    大半だった。
12. **local forkとの関係:** epoch79 forkは制御された短期摂動、今回の5 seedは独立
    full-run。数値を同一varianceとして比較せず、「局所分岐と長期global縮小は両立
    し得る」とだけ接続する。
13. **History / Orderingの根拠:** global finalが改善してもsample-level disagreement
    が残るため、次段階でHistoryを記述的targetとして検討する根拠はある。ただし
    interventionは今回開始していない。
14. **主張してはいけないこと:** RNG source単独の因果 attribution、training-seed母集団
    への有意性、I100の普遍的最適性、sampleをintrinsically difficultとする断定。

## Artifacts

- [`ert_rslad_five_seed_artifact_inventory_v1.json`](experiments/ert_rslad_five_seed_artifact_inventory_v1.json)
- [`ert_rslad_five_seed_global_stochasticity_v1.json`](experiments/ert_rslad_five_seed_global_stochasticity_v1.json)
- [`ert_rslad_five_seed_sample_stochasticity_v1.json`](experiments/ert_rslad_five_seed_sample_stochasticity_v1.json)
- ローカル大容量行列・CSV・依存なしSVG: `.cache/analysis/ert-rslad-five-seed-stochasticity-v1/outputs/`

## STOP / 次段階

今回のread-only解析は完了。新しいtraining、endpoint再生成、seed追加、timing変更、
History/Ordering intervention、official test、AutoAttackは開始しない。次の実験では、
今回保存したseed-sensitive sample IDs・margin variability・rescue consistencyを
入力候補として、別途human review済みの設計を作る。
