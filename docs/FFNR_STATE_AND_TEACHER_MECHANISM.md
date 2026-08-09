# FF/NR 3-state, margin-response, and Teacher-mechanism analysis

更新日: 2026-08-10  
Status: Chen ERT L2/L4 train-split point analysis; no new training, intervention,
official test, AutoAttack, or bootstrap.

## 固定した入力と定義

対象は Chen ERT の L2 (seed 1) / L4 (seed 2) です。feature は CE-PGD20
replay の epochs `39, 59, 79`、outcome は共通 terminal epochs
`189, 194, 199` です。primary は3 checkpoint中2回以上wrong (`majority`)、
secondary は3回すべてwrong (`all`) とし、結果を混ぜていません。
stable-ID/class universe は45,000件で、feature/outcome/online stateのlineage
とattack identityを照合しています。

Student/Teacher の probability margin は true-label 側が正になるように、

$$
m_S^{clean}=p_S(y|x)-\max_{c\ne y}p_S(c|x),\quad
m_S^{adv}=p_S(y|x_S^{adv})-\max_{c\ne y}p_S(c|x_S^{adv}),
$$

$$
m_T^{clean}=p_T(y|x)-\max_{c\ne y}p_T(c|x),\quad
m_T^{adv}=p_T(y|x_S^{adv})-\max_{c\ne y}p_T(c|x_S^{adv}).
$$

attack response は `DeltaS = mS_clean - mS_adv`、`DeltaT = mT_clean -
mT_adv` です。保存された correctness と margin 符号の不一致は fail-closed
にしました。clean correctness は3-stateへ混ぜず、別flagとして保持しています。

再現用正本は [machine-readable report](experiments/ffnr_state_and_teacher_mechanism_v1.json)
（SHA-256 `c36ca3cdec17e6e7f7a9efbd267318cab88cb62803d1cb9e7a5020635ca8ca29`）です。
解析 source SHA は `e682fb3681d118155b819713e951839629bbe903`、dirty=false です。

## 連続 risk surface

current-correct cohort内で、Student/Teacher marginの低い側をriskとして固定
quantile bin化しました。anchor 39/59/79、majority/allの全曲線と Student×Teacher
5×5 surface はJSONに保存しています。代表的にanchor 79/majorityでは、最安全binから
最危険binへのFuture Failure率は次の通りです。

| run | Student `-mS_adv` | Teacher `-mT_adv` | Teacher `DeltaT` |
|---|---:|---:|---:|
| L2 | 0.0% → 36.4% | 0.0% → 62.3% | 0.6% → 30.8% |
| L4 | 0.0% → 39.1% | 0.0% → 63.6% | 1.0% → 29.3% |

これは強い予測的関連であり、Teacher responseがtraining failureを引き起こす
因果効果を意味しません。

## 3-state候補と2-state簡約

候補は outcome を見て自動選択せず、current-positive marginの下位10/20/25/33%
を Fragile Correct 候補として列挙しました。anchor 79/majorityの例:

| run / measure | fragile fraction | tau | fragile FF rate | safe FF rate | wrong FF rate |
|---|---:|---:|---:|---:|---:|
| L2 / Student | 10% | 0.0547 | 5.24% | 0.40% | 25.15% |
| L2 / Teacher | 10% | 0.1707 | 36.62% | 0.22% | 99.73% |
| L4 / Student | 10% | 0.0558 | 5.12% | 0.25% | 27.16% |
| L4 / Teacher | 10% | 0.1714 | 35.46% | 0.20% | 99.50% |

全候補の件数、3×3 cell、`Safe/Fragile/Wrong` 対 `Correct/Wrong` および
`Safe/Risky` は機械可読reportへ保存しました。現時点で最終 tau は決めません。
同一seedのFF率最大化や介入結果による閾値調整はしていません。

## Teacher signal の分解

cross-seed の low-complexity model は次を比較しました。

| model | features |
|---|---|
| M0 | strong current Student `mS_adv` |
| M0_history | online margin-EMA risk |
| M1 | M0 + `mT_clean` |
| M2 | M0 + `DeltaT` |
| M3 | M0 + `mT_clean` + `DeltaT` |
| M4 | M0 + `mT_adv` |

`L2→L4` / `L4→L2` の anchor 39/59/79、majority/allを全て報告しています。
anchor 39/majorityの代表値は:

| direction | M0 AUROC | M0 history | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|---:|
| L2→L4 | .9213 | .9168 | .9862 | .9105 | .9925 | .9919 |
| L4→L2 | .9327 | .9190 | .9872 | .9079 | .9926 | .9925 |

Teacher clean margin はStudent current stateへ大きな追加情報を持ち、`DeltaT`
単独の追加は弱い一方、`mT_clean + DeltaT` の組合せは両方向でM1をさらに上回ります。
この結果は次のどちらかを単独で採用する根拠ではありません。

- Teacher intrinsic clean difficulty が主因
- Student-crafted attackでTeacherが崩れる response が追加情報

Teacher clean/adversarial correctness条件別の件数・FF率、Student response、Teacher
responseはJSONの `teacher_conditional_decomposition` に保存しています。Teacherが
adversarial wrongのcohortはほぼ全てFFですが、これはbinary splitが極端に分離して
いることを示すだけで、KD downweightの有効性を示しません。

## Student response と既存 Route A/B

`DeltaS` を同じ表に含め、Teacher responseが単なるStudent responseのproxyかを
cross-seedで比較できるようにしました。既存 Route A/B の保存済みpilotは別計画の
training-state KL-PGD10 endpointであり、common eval-mode CE-PGD20のcausal endpoint
ではありません。そのため、既存の selected/random rescue-harm 表は参考の
heterogeneity evidenceとして扱い、新しい因果効果とは再解釈していません。

既存pilotでRoute B selected が random より方向的に良かった差（L2 +1.16 pp、L4
+1.32 pp）は、bootstrap CI が0を跨ぐため確証ではありません。今回のstate overlay
から新しい treatment、q、係数、routingを自動起動していません。

## IRT、self-attack、gradient alignment

Bartoldson/IRTにはこの計画と同一の `[39,59,79]` feature と `[189,194,199]`
terminal outcomeを満たすartifactがありません。既存の `[104,109,114]` replayを
同一endpointの代用にはせず、IRT cross-seed/cross-teacher claim は blocked です。

Teacher self-attack と gradient alignment は、Teacher responseの追加情報が事前
固定gateを通った場合だけ実行する条件付き拡張です。今回の point analysisだけから
自動起動していません。したがって、今回の結論は prediction/association/conditional
association に限定し、transfer geometryやcausal treatment effectとは呼びません。

## 次の判断

1. 人間が quantile候補または事前固定した change-point候補を最大2つに絞る。
2. M1/M3の差を別seed・IRTの同一endpointで確認するか、IRT dense checkpointを先に
   回収する。
3. その後にのみ、state別のRoute A/B候補を同一parent・matched-random controlで
   小規模screenする。Best validation robust accuracyをprimaryにする。
4. method固定後に未使用seedで確認し、official PGD、最後にAutoAttackへ進む。

今回の実験ではGPU replay、新規training、test/AAを行っていません。CPU point解析は
L2/L4を別processで実行し、各single-runは約13--18秒、mergeは約13秒でした。
