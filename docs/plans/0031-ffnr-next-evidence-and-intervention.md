# FF/NR 次段階エビデンス回収・介入検証計画

実行ログ: 2026-08-09 にepoch79親のmask準備、loss-scale dry-run、FFNR causal arm/runtime（C79/RA/RAR/RB/RBR）を実装。focused contract test後にGPU pilotを開始する。Route B primaryはq=5%に固定し、q=10%は感度候補として未実行。

## 0. 目的

この計画の目的は、Chen ERTで確認されたFF/NRの構造とTeacher responseの強い予測信号をもとに、残っている科学的な未確認事項を回収し、その後に小規模な因果介入を行うことである。

最終的に答えたい問いは次の4点。

1. Future Failure GTのseed間一致は、Failure prevalenceが増えたことだけでは説明できないか。
2. Teacher signed wrong-class dominanceは、Student difficultyを考慮してもseedを跨いで追加情報を持つか。また、その意味はERTとIRTで同じか。
3. FFとcurrent-wrong/NRで異なるmechanismに対し、原因仮説に合った介入を行うと対象sampleを改善できるか。
4. 高risk sampleを選んで介入すること自体にutilityがあり、matched-random sampleへ同じ介入をするよりglobal robust generalizationを改善できるか。

この段階ではofficial CIFAR-10 testとAutoAttackを開かない。

---

## 1. 実行開始時のrepo reconciliation

実行前に必ず `origin/master` をfetchし、最新HEADと以下を確認する。

基準として確認済みの状態:

```text
origin/master: a31f5cc
docs/FFNR_FORECASTING_STATUS.md
docs/plans/0030-ffnr-decomposition-and-strong-nr.md
```

正式strong diagnostic artifact:

```text
.cache/analysis/ffnr-strong-diagnostics-6a90011-v1/
```

formal report SHA-256:

```text
cfdd3f66174562815b7550ee1d7cd7d02752eaaa160d45426c872e8c12eec269
```

既存artifact、checkpoint、replay、stable-ID join、attack lineageを再利用し、同じGPU replayを重複させないこと。

新しいcommitやanalysisが追加されている場合は、その結果を先に読み、以下の計画で既に完了している項目を再実行しない。

---

## 2. 現時点で固定する科学的事実

### 2.1 Chen ERTのFuture Failure endpoint

Chen ERT L2/L4では、共通terminal window

```text
[189, 194, 199]
```

を用いる。

primary endpoint:

```text
3 checkpoint中2回以上wrong = majority
```

secondary sensitivity:

```text
3 checkpointすべてwrong = all
```

`majority` と `all` の結果を混ぜない。

このcommon-terminal endpointは、L2のbest-centered windowと完全に同一ではない。
seed比較のために共通化されたendpointであることをreportへ必ず残す。

---

### 2.2 Teacher signed wrong-class dominance

Teacher signed wrong-class dominanceを次で定義する。

$$
D_T(x_i^{\mathrm{adv}})
=
\max_{c \neq y_i} p_T(c \mid x_i^{\mathrm{adv}})
-
p_T(y_i \mid x_i^{\mathrm{adv}})
$$

値が大きいほどTeacherがtrue classよりwrong class側へ傾いている。

Chen ERTでは、current-correct FF予測において非常に強いassociationが確認されている。

ただし、標準RSLADのtraining contractでは、Student adversarial logitsのKD targetはTeacher clean logitsである。

標準RSLADは概念的に以下を用いる。

$$
\mathrm{KL}
\left(
p_S(x^{\mathrm{adv}})
\;\|\;
p_T(x_{\mathrm{clean}})
\right)
$$

したがって、$D_T(x_S^{\mathrm{adv}})$ は標準RSLADで直接模倣されるtargetそのものではない。

現時点では、

> Student-crafted adversarial inputに対するTeacher responseが、将来のStudent failureを強く示すdiagnostic signalである

と解釈する。

「Teacherのadversarial wrong targetをStudentが模倣したためFailureした」とは解釈しない。

---

### 2.3 FFの構造

anchor 79のChen ERTではFuture Failureは一様ではない。

主要なsnapshot subtype:

- transient-correct: 約半数
- stable-then-forgotten: 約3割
- oscillating: 約2割弱

したがって、FF全体を「一度学習したsampleのforgetting」とみなさない。

FFへの介入では、主仮説を単純なmemory retentionではなく、

> 十分なrobust marginを獲得できていないsampleのboundary stability

へ置く。

future subtypeは介入selectionに使わない。
subtypeはfuture informationを含むため、介入後のheterogeneity analysisにのみ使用する。

---

### 2.4 Current-wrong / NRの構造

strong-domainのcurrent-wrongには少なくとも次が存在する。

- persistent-wrong
- recovered-relapsed
- recovered-stable

persistent-wrongは比較的seed-stableで、Student clean-wrongおよびTeacher adversarial-wrongが多い。

recovered-relapsed / recovered-stableではTeacherはほぼcorrectである。

したがってcurrent-wrong全体へ一つの介入を適用しない。

---

### 2.5 Human image review

blinded image panelの人手確認は完了済みとする。

同じpanelの生成・再レビューをこの計画では行わない。

人手確認の具体的なannotation結果がrepo内に構造化されていない場合、Codexは内容を推測しない。
label-noise、ambiguity、rare-subpopulationに関する新しい割合を自動で作らない。

介入対象から明らかなmislabeled sampleを除外する必要があるなど、人手annotationの具体的結果が実験contractに必要になった場合のみ、その情報がrepoに存在するか確認する。
存在しない場合は勝手に仮定せず、人間へ確認する。

---

# Part A. Future Failure GTのchance-adjusted seed agreement

## A1. 目的

現在のraw Jaccardだけでは、

> strong attackでFailure prevalenceが増えたため集合overlapも機械的に増えた

という説明を完全には排除できない。

固定済みprimary endpointに対して、seed間agreementがprevalence-only nullをどの程度上回るか確認する。

GTを再選択するための実験ではない。

---

## A2. 対象

primary:

```text
Chen L2 vs L4
endpoint = [189, 194, 199] majority
```

secondary:

```text
same endpoint, all
```

他のGT candidate familyを再探索しない。

---

## A3. 必須指標

同一45,000 stable-ID universe上で以下を計算する。

- seedごとのFuture Failure count
- seedごとのprevalence
- intersection count
- union count
- raw Jaccard
- raw agreement rate
- Cohen's kappa
- future failure frequencyのseed間Spearman
- future failure frequencyのseed間Pearson
- frequency agreement matrix

3-checkpointのfuture failure frequencyは、

$$
f_i^{\mathrm{future}}
\in
\left\{
0,\frac{1}{3},\frac{2}{3},1
\right\}
$$

とする。

これを`failure probability`とは呼ばず、`future failure frequency`と呼ぶ。

---

## A4. Jaccardのindependence null

各seedのpositive countを固定したままstable IDをpermuteし、independence nullを作る。

推奨:

```text
10,000 permutations
```

出力:

- null Jaccard mean
- null 2.5 percentile
- null 97.5 percentile
- observed Jaccard
- observed - null mean
- observed / null mean
- empirical tail probability

主要結論はp-valueではなくeffect sizeで記述する。

---

## A5. Bootstrap

stable IDをunitとしたpaired bootstrapを行う。

対象:

- raw Jaccard
- Cohen's kappa
- Spearman
- Pearson

95% CIを出す。

seed1/seed2のsample対応を保持したpaired resamplingとする。

このCIをtraining-seed generalizationのCIとは呼ばない。

---

## A6. 判定

以下が同時に成立すれば、

> primary CE-PGD20 Future Failureは、prevalenceだけでは説明できない強いsample-level seed reproducibilityを持つ

と記述してよい。

- observed Jaccardがnullから十分離れる
- Cohen's kappaが高い
- future failure frequencyにも明確なseed相関がある

成立しない場合はraw Jaccardの解釈を弱める。

---

# Part B. Teacher responseの残りのpredictive evidence

## B1. 既に完了している解析を重複しない

以下は完了済みなので再実行しない。

- Teacher correct / wrong別Future Failure rate
- strong Student marginとのSpearman
- online margin EMAとのSpearman
- within-run class-stratified OOF
- Student marginにTeacher dominanceを追加したdelta
- Student margin + historyにTeacher dominanceを追加したdelta

---

## B2. Cross-seed generalization

残る重要な確認は、Teacher signalの追加価値がtraining seedを跨ぐかである。

Chenについて:

```text
L2でfit -> L4でevaluate
L4でfit -> L2でevaluate
```

を行う。

anchorは39/59/79を個別に報告する。

### Student-only model

最低限:

- strong current Student logit-margin risk
- online margin-EMA risk

必要ならclassを固定effectとして追加してよい。

### Student + Teacher model

Student-only modelへ

- Teacher signed wrong-class dominance

のみ追加する。

preprocessing、scaling、calibration parameterはtrain seedのみでfitする。

test seedの統計量を使ってstandardizeしない。

---

## B3. Metrics

各directionについて:

- AUROC
- AUPRC
- log loss
- Brier score

を報告する。

さらに、

- delta AUROC
- delta AUPRC
- delta log loss
- delta Brier

を報告する。

目的は最高精度モデルを作ることではない。

問いは、

> Teacher signalのincremental informationがseedを跨いでも残るか

だけである。

---

## B4. Teacher-correct subset

現在のAUROC約0.99は、Teacher correct / wrongのbinary splitが大きく寄与している。

そこでTeacherがStudent attack上でcorrectなsampleだけに限定して、signed dominanceの連続値がまだ意味を持つか確認する。

Teacher-correct subset内では通常 $D_T < 0$ である。

見るもの:

- $D_T$ quantile別Future Failure prevalence
- strong Student marginをquantile conditioningした後のrisk gradient
- AUROC / AUPRC
- cross-seed rank correlation
- cross-seed fit/eval

ここでsignalが弱くなってもnegative resultとして受け入れる。

その場合、

> Teacher correctness itself is the dominant ERT signal

と解釈する。

---

# Part C. Bartoldson IRTのdevelopment sensitivity

## C1. 目的

Teacher signed dominanceがChen ERT固有なのか、IRTでも同様なのか確認する。

ただし現在のBartoldson checkpoint inventoryではChenと同一のBest-centered/common-terminal contractを構築できない。

したがって同一endpoint比較を装わない。

---

## C2. 第一段階

新しい200 epoch trainingを始める前に、既存checkpointだけで構成可能なdevelopment sensitivityを1つ事前固定する。

候補:

```text
[104, 109, 114]
```

このwindowを使う場合は、

```text
non-Best-centered IRT development sensitivity
```

と明記する。

Chenとの絶対AUROCを同じGTとして直接比較しない。

---

## C3. IRTで見るもの

最低限:

- Teacher adversarial correct / wrong prevalence
- Teacher signed dominance distribution
- Future Failure prevalence
- signed dominance AUROC / AUPRC
- Student marginとの相関
- historyとの相関
- Student-only vs Student + Teacher incremental metrics
- Teacher-correct subset内のsignal

---

## C4. 判断

### IRTでTeacher wrongが極端に少なくsignalが弱い場合

IRTではTeacher failureではなく、

- Teacher non-response
- overconfidence
- Student-Teacher local mismatch

が重要な可能性を残す。

Chen ERTとIRTを一つのuniversal Teacher scoreへ無理に統合しない。

### IRTでもsignalが強い場合

Teacher responseがteacher typeを超えた一般的robust learnability markerである可能性が高まる。

ただしendpointが異なるため、publication-gradeのIRT confirmationには別途dense-checkpoint runが必要。

---

## C5. 新規Bartoldson dense runのgate

以下のいずれかを満たす場合のみ、epoch 100前後をdense-saveする新規non-intervened Bartoldson runを提案する。

- IRT development sensitivityが研究ストーリーを左右する
- Chenで介入pilotが成功し、teacher-type generalization確認が必要
- current inventoryのcensoringが論文上の主要な弱点になる

自動で2本の長期runを開始しない。

---

# Part D. 介入仮説

ここから先はpredictionではなくtreatment utilityを検証する。

Future subtypeや将来persistentかどうかをselectionに使わない。

selection時点までに得られる情報だけを使う。

初回は固定anchor・固定maskのshort-horizon pilotとし、dynamic online methodはその後に分離する。

---

## D1. Route A: joint-hard current-wrong route

### 仮説

次のようなsampleでは、Studentはcleanでも難しく、Student attackがTeacherにも通る。

- Student adversarial wrong
- Student clean wrong
- Teacher adversarial wrong

この群では、標準RSLADのadversarial KDだけに依存するより、

> selected sampleのadversarial KD圧力を弱め、true-label adversarial CE anchorを追加する

方がStudentの回復に有利な可能性がある。

注意:

Teacher adversarial prediction自体は標準RSLADのKD targetではない。

したがってこの介入は、

> wrong adversarial Teacher targetを修正する

ものではない。

Teacher adversarial failureを、joint difficulty / shared vulnerabilityを示すrouting signalとして利用する。

---

## D2. Route Aのpilot selector

anchor時点でのみ利用可能な情報で固定maskを作る。

mechanism pilotの候補条件:

- Student strong CE-PGD20 wrong
- Student clean wrong
- Teacher wrong on the same Student-crafted CE-PGD20 adversarial input

future persistenceは条件に含めない。

realized countとclass distributionをreportする。

このselectorはstrong replayを必要とするため、最終deployable selectorではない。
まずtreatment utilityを調べるためのmechanism selectorとして扱う。

---

## D3. Route Aのtreatment

以下は変えない。

- attack generation
- epsilon
- PGD steps
- random-start policy
- clean KD
- nonselected sample loss
- optimizer
- scheduler

selected sampleについてのみ:

1. adversarial KD coefficientを下げる
2. adversarial true-label CEを追加または増加する

概念的には、

$$
L_i^{A}
=
\lambda_{\mathrm{KD}}
L_{\mathrm{KD,adv},i}
+
\lambda_{\mathrm{CE}}
L_{\mathrm{CE,adv},i}
+
L_{\mathrm{KD,clean},i}
$$

とする。

baselineとの差を最小にする。

---

## D4. Route Aの係数決定

validation performanceを見ながら係数を探索しない。

trainingを開始する前に、baseline checkpoint上の1回のloss-scale dry-runを行う。

候補は最大2組まで。

選択基準:

- selected sampleのmedian total loss scaleがbaselineから極端に変わらない
- gradient normが爆発しない
- KDを完全に捨てる設定を初手にしない
- true-label CEがdominantになりすぎない

係数が一意に決まらない場合は比較表を出して人間へ確認する。

---

## D5. Route B: boundary-risk current-correct route

### 仮説

FFの過半はstableなknowledgeを失うというより、一時的にcorrectになったboundary-near sampleである。

TeacherがStudent attack上でもcorrectなsampleについてはTeacherをsoftenする根拠が弱い。

したがって、

> Teacher KDを維持したまま、selected sampleにtrue-label adversarial CEを追加し、current robust boundaryを安定化する

介入を試す。

---

## D6. Route Bのpilot selector

anchor時点で:

- Student strong CE-PGD20 correct
- Teacher correct on the same Student-crafted adversarial input

をeligibilityとする。

その中からStudent boundary riskでrankする。

primary candidate:

- strong current Student logit-margin risk

deployable sensitivity:

- online margin-EMA risk

future FF subtypeはselectionに使わない。

---

## D7. Route Bのtreatment

Teacher KDは変更しない。

selected sampleについてのみadversarial true-label CEを追加する。

概念的には、

$$
L_i^{B}
=
L_{\mathrm{RSLAD},i}
+
\lambda_{\mathrm{CE}}
L_{\mathrm{CE,adv},i}
$$

とする。

このpilotでは、

- Teacher temperature変更
- KD target変更
- attack strength変更
- feature loss追加

を同時に行わない。

一つのmechanismだけを検証する。

---

# Part E. Selection budgetの事前固定

## E1. Route A

Route Aはrule-based selectorなので、top-qへ無理に変換しない。

条件を満たしたsampleのrealized fractionをそのまま報告する。

matched-random armは同じsample数にする。

---

## E2. Route B

Route Bではtop-qが必要。

既存score/GTから以下を出す。

- top 5%
- top 10%
- top 20%

各qについて:

- precision
- recall
- lift
- selected count
- class balance
- seed間mask stability

GT-count oracle-qはanalysis-onlyとする。

---

## E3. qの選択

long trainingの結果を使ってqを選ばない。

short-horizon pilot前に最大2候補まで絞る。

推奨は、

- high precision / low coverage
- moderate precision / higher coverage

のPareto上2点。

一意に決まる場合は1点のみ。

人間判断が必要な場合はpilot開始前に停止する。

---

# Part F. Short-horizon randomized mechanism pilot

## F1. Anchor

第一候補はepoch 79。

理由:

- D3/D4/D5 diagnosticが最も揃っている
- LR milestone 100より前
- 10から15 epochの介入を同一scheduler stageで観測できる

実行前にcheckpoint lineageとoptimizer/scheduler stateを確認する。

epoch 79が技術的に使えない場合、勝手に別anchorへ変えず理由を報告する。

---

## F2. Horizon

推奨:

```text
79 -> 84
79 -> 89
79 -> 94
```

つまり+5 / +10 / +15 epoch。

LR boundaryを跨がない。

---

## F3. Arms

各seedについて同一parent checkpointからforkする。

最低限:

```text
C0: Control
    interventionなし

A1: Route A selected
    joint-hard selector + Route A treatment

A2: Route A matched-random
    same treatment, matched random IDs

B1: Route B selected
    boundary-risk selector + Route B treatment

B2: Route B matched-random
    same treatment, matched random IDs
```

5 GPUが利用可能なら5 armをparallelにしてよい。

ただしhost差が実験結果へ入らないよう、

- GPU model
- CUDA/PyTorch
- deterministic flags
- data path
- seed contract

を確認する。

host差がある場合はarm assignmentをseed間でswapするか、同一host内paired comparisonを優先する。

---

## F4. Matched-random contract

単純randomではなく、可能な限り以下をmatchする。

Route A:

- same eligibility side
- class
- Student clean-correct/wrong state
- Student strong margin quantile

Route B:

- current-correct
- Teacher-correct
- class
- Student strong margin quantile

exact matchingが不可能な場合はmatching qualityをreportする。

random selection seedを固定しhash-bindする。

---

## F5. Pairing

fork時点で揃える。

- Student weights
- optimizer state
- scheduler state
- RNG state
- dataloader order
- augmentation seed
- attack random-start policy
- batch construction
- nonselected sample treatment

介入後はgradientが異なるので、同じattack seedでも各armのadversarial imageは異なってよい。

評価時は各armを独立にattackする。

controlのadversarial imageを介入armへ流用しない。

---

## F6. Short-horizon primary outcomes

selected IDsはanchor 79で固定する。

各evaluation checkpointでcommon CE-PGD20を用い、

- selected-sample robust accuracy
- selected-sample robust logit margin
- rescue count
- harm count
- net rescue
- selected-sample clean accuracy

を出す。

rescue:

```text
Control wrong, Treatment correct
```

harm:

```text
Control correct, Treatment wrong
```

net rescue:

```text
rescue - harm
```

---

## F7. Global secondary outcomes

- validation CE-PGD20 accuracy
- validation clean accuracy
- nonselected-sample robust accuracy
- train robust loss
- clean/robust flip rate

short horizonではglobal Bestが更新されなくても即No-Goとはしない。

ただし明確なglobal harmがある場合は停止する。

---

## F8. 必須比較

Route A:

```text
A1 vs C0
A1 vs A2
```

Route B:

```text
B1 vs C0
B1 vs B2
```

解釈:

### Selected treatmentがControlより良く、randomよりも良い

routing utilityの証拠。

### Selected treatmentがControlより良いがrandomと同等

treatment自体は有効かもしれないが、selector utilityは未支持。

### Selected treatmentがControlより悪い

そのrouteはNo-Go。

### local rescueは増えるがglobal validationが悪化

sample-local mechanism evidenceとしては残るが、main methodとしてはNo-Go候補。

---

## F9. Short-pilot統計

L2/L4のpaired seedを別々に報告する。

sample-level:

- paired bootstrap CI
- risk difference
- margin delta
- rescue/harm difference

training-level:

- seedごとのarm delta

45,000 sampleをtraining replicateとして扱わない。

seed平均だけで方向の不一致を隠さない。

---

# Part G. Positive routeのcomponent ablation

short-horizonでpositiveなrouteだけ実施する。

## G1. Route Aがpositiveの場合

composite treatmentのどちらが効いたか確認するため、次の小ablationを追加してよい。

- KD downweight only
- CE anchor only
- KD downweight + CE anchor

全て同じselected mask。

このcomponent ablationはlong-horizon full training前に行う。

Route Aがnegativeなら実施しない。

---

## G2. Route Bがpositiveの場合

Route BはCE追加だけなので、追加component ablationは原則不要。

必要ならCE coefficient sensitivityを最大2値だけshort horizonで確認する。

validation performanceを何度も見ながら探索しない。

---

# Part H. Deployable selectorへの変換

strong CE-PGD20 Teacher responseを毎epoch全sampleへ計算する方式は、最終methodとして高コストになりうる。

mechanism pilotがpositiveなrouteだけ、deployable selectorへ変換する。

---

## H1. Candidate selector hierarchy

### Level 1: Student-only streaming

優先:

- current correctness
- margin EMA
- correctness frequency
- streak / flip state

既存training stateから得られるO(1) scalarを優先する。

### Level 2: Training-attack Teacher response

必要な場合のみ、

> training中に既に生成しているStudent adversarial inputへTeacher forwardを追加する

方式を試す。

強いCE-PGD20を毎epoch別生成しない。

### Level 3: Student prefilter + Teacher response

計算削減のため、

1. Student-only scoreでcandidateを絞る
2. candidate subsetだけTeacher responseを計算する

方式を許可する。

---

## H2. Strong selectorとのproxy agreement

成功したmechanism selectorをreferenceとして、

- precision
- recall
- Jaccard
- rank correlation
- selected-sample treatment-response enrichment

を比較する。

単にscore correlationが高いだけで選ばない。

最終的に重要なのは、

> positive treatment responderをcheap selectorが濃縮できるか

である。

---

## H3. Overhead

実測する。

- wall-clock / epoch
- Teacher forward count
- GPU memory
- communication overhead

ImageNetへの将来scaleを考え、preferred methodはbaseline比おおむね20%以内の追加training costを目標とする。

20%を超えた場合でもmechanism resultは棄却しないが、

```text
oracle / expensive selector
```

として分離し、cheap proxyを探す。

---

# Part I. Dynamic online routing

固定mask pilotに成功した後だけ実装する。

## I1. 原則

最終methodではsampleを永久にhigh-riskと固定しない。

FF subtypeのseed overlapが低く、current state自体が移動するため、

> every-epoch state update

を基本にする。

---

## I2. Warmup

warmupを恣意的に長くしない。

候補:

```text
5 epochs
10 epochs
20 epochs
```

既存trajectoryからoffline simulation可能な範囲で、

- score reliability
- selected fraction
- consecutive Jaccard
- entry / exit
- FF <-> current-wrong transition

を比較する。

最短で安定するwarmupを選ぶ。

long training結果を使ってwarmupを最適化しない。

---

## I3. 毎epoch更新

各sampleについてstreaming stateのみ保持する。

例:

- margin EMA
- correctness frequency
- current streak
- fast / slow EMA

全logit historyや画像を保存しない。

ImageNet scaleでO(1) state / sampleを維持する。

---

## I4. Treatment strength

最初の1から数epochだけ介入強度をrampする案は許可する。

ただし、

- q
- treatment coefficient
- warmup
- ramp

を同時にgrid searchしない。

一度に一つの設計判断だけを行う。

---

# Part J. Long-horizon development experiment

dynamic online routeがfixed-mask pilotの効果を再現できた場合のみ行う。

## J1. Primary outcome

最終methodのprimary development metric:

```text
Best validation CE-PGD20 robust accuracy
```

secondary:

- Last validation CE-PGD20
- robust overfitting gap
- clean validation accuracy
- class-wise robustness
- selected/nonselected sample accuracy
- treatment exposure count

---

## J2. Sample-level causal endpoint

sample-level比較では、各arm自身のpost-treatment Best epochをGT定義に使わない。

baseline/controlから事前に固定したfuture evaluation epochsを使う。

理由:

arm-native Best epochはpost-treatment variableであり、treatmentによって動く可能性がある。

---

## J3. Arms

positive routeごとに最低限:

```text
Control
Dynamic selected treatment
Dynamic matched-random treatment
```

Route A/B両方がpositiveでも、最初からjoint interventionへ混ぜない。

各routeを単独で評価する。

両方単独で有効だった場合のみjoint routingを次段階として提案する。

---

## J4. Seeds

L2/L4相当の2 development seedsでpaired評価する。

method freeze前にunused confirmation seedをdevelopmentへ流用しない。

---

## J5. Go criteria

method candidateとしてGoとするには、最低限:

- selected treatmentがmatched-randomより一貫して良い
- Best validation robustnessがControlを悪化させない
- 可能なら2 seedともBest improvementが同方向
- clean accuracyの崩壊がない
- treatmentが一部sample rescueだけでなくglobal generalizationへ波及する

local rescueのみでglobal Bestが改善しない場合:

```text
mechanism evidence = positive
training method = No-Go or redesign
```

と分けて記録する。

---

# Part K. IRTへのtransfer確認

Chenでmethod candidateが成立した後、IRTで同じrouteを盲目的に適用しない。

Part Cの結果に基づいて判断する。

### ERT signalとIRT signalが異なる場合

Teacher-specific routingを認める。

例えば:

- ERT: Teacher failure / shared vulnerability route
- IRT: Teacher non-response / mismatch route

のように別設計とする。

### 共通signalがある場合

同じmethodをIRTへconfirmationする価値がある。

publication-grade比較にはdense checkpoint contractを新規freezeする。

---

# Part L. Official evaluation gate

以下が完了するまでofficial CIFAR-10 testとAutoAttackを使用しない。

- intervention route固定
- selector固定
- q固定
- warmup固定
- coefficient固定
- development seedsでGo
- unused confirmation seedで方向確認

その後にのみofficial evaluation planを別途作成する。

---

# Part M. W&Bとartifact contract

production trainingは全てW&Bへ記録する。

最低限:

- run ID
- source Git SHA
- dirty / clean state
- parent checkpoint SHA
- teacher checkpoint SHA
- dataset identity
- selector definition
- selector hash / selected ID hash
- q / realized fraction
- treatment coefficients
- attack identity
- optimizer / scheduler
- RNG seed
- wall time
- GPU identity

analysis artifactはnon-overwritingとしSHA-256を付ける。

---

# Part N. 必須report

次段階report:

```text
docs/FFNR_NEXT_EVIDENCE_STATUS.md
```

最低限のsections:

1. GT chance-adjusted agreement
2. Cross-seed Teacher incremental information
3. Teacher-correct subset
4. IRT development sensitivity
5. Route A selector definition
6. Route B selector definition
7. q decision
8. Short-horizon arm table
9. Selected-sample rescue / harm
10. Matched-random comparison
11. Global validation effect
12. Go / No-Go decision
13. Remaining uncertainty

cache:

```text
.cache/analysis/ffnr-agreement-*/
.cache/analysis/ffnr-cross-seed-teacher-*/
.cache/analysis/ffnr-irt-sensitivity-*/
.cache/analysis/ffnr-causal-pilot-*/
```

存在しないartifactを形式だけ作らない。

---

# Part O. Codexの裁量範囲

## 自動で決めてよい

- CPU parallelization
- bootstrap / permutation implementation
- deterministic fold implementation
- stable-ID joins
- hash / lineage schema
- focused unit tests
- matching algorithmの実装詳細
- GPU job scheduling
- obvious bug fix
- report formatting

## 自動で決めてはいけない

- Future Failure GTの再選択
- endpointをAUROCで変更
- IRTとERTの異なるendpointを同一GTとして扱う
- future subtypeをonline selectorへ使用
- qをlong-run結果で再調整
- 多数のloss coefficient grid search
- attack strength変更を介入と同時に追加
- Teacher target自体を新規にadversarial predictionへ置換
- official test / AutoAttack使用
- unused confirmation seedのdevelopment使用
- positive local rescueだけでmethod successと宣言
- Teacher signed dominanceについて因果が証明されたと記述

複数案が科学的に同程度で一意に決められない場合は、比較表を出して人間へ質問し停止する。

---

# Part P. Stopping rules

以下では無理に次段階へ進まない。

### GT agreementが弱い

Future Failureをsample-stable targetとして扱う主張を弱める。

### Teacher incremental signalがcross-seedで消える

Teacher dominanceをdevelopment-specific markerとして扱い、main selector候補から降格する。

### Route A / Bのselected interventionがmatched-randomを上回らない

routing utilityはNo-Go。

### Selected sampleは改善するがglobal validationが悪化

mechanism positive / method No-Go。

### Dynamic cheap selectorがfixed strong selectorのbenefitを再現できない

expensive oracle mechanismとして分離し、scalable method claimをしない。

---

# Part Q. 最終的に答えるResearch Questions

### RQ1

Future FailureはFailure prevalenceを補正してもseedを跨いでsample-levelに再現するか。

### RQ2

Teacher signed wrong-class dominanceはStudent current risk/historyを超える情報をtraining seedを跨いで持つか。

### RQ3

Teacherがcorrectなsampleの中でもTeacher signed dominanceの連続値は意味を持つか。それともTeacher correctnessのbinary eventが主因か。

### RQ4

ERTとIRTでTeacher responseの意味は同じか。

### RQ5

joint-hard current-wrong sampleに対して、adversarial KDを弱めtrue-label CE anchorを追加すると回復が増えるか。

### RQ6

teacher-correct boundary-risk FF sampleに対して、KDを維持したtrue-label adversarial CE追加はfuture instabilityを減らすか。

### RQ7

高risk selectorへ介入することは、同じtreatmentをmatched-random sampleへ適用するより有効か。

### RQ8

fixed strong diagnostic selectorで確認したtreatment utilityを、低コストなonline streaming selectorで再現できるか。

### RQ9

sample-level rescueは最終的にBest validation robust accuracyの改善へつながるか。

---

# 最終原則

この段階では、

```text
強いpredictive association
        ↓
chance-adjusted / cross-seed evidence
        ↓
mechanismに合わせたfixed-mask randomized pilot
        ↓
matched-randomに対するrouting utility
        ↓
cheap online selectorへの変換
        ↓
dynamic long-horizon intervention
        ↓
global robust generalization
```

の順に進める。

どこかの段階で証拠が途切れた場合は、その段階までを研究結果として保持し、次の段階を無理に正当化しない。

---

## Execution status in this repository

- [x] 2026-08-09: Reconciled `origin/master` at `c44758b`; reused the existing
  Chen strong replay, online state, and stable-ID universe.  No replay was
  duplicated.
- [x] 2026-08-09: CPU Part A completed for Chen L2/L4, common terminal
  `[189,194,199]`, with separate `majority` and `all` endpoints.  The fixed
  positive-count independence null uses a hypergeometric sampler, which is
  mathematically equivalent to ID permutation and avoids materializing large
  permutation arrays.  Paired stable-ID bootstrap uses 2,000 replicates.
- [x] 2026-08-09: CPU Part B completed for anchors `39/59/79`, with both
  fit/evaluate directions, fit-seed-only standardization, M/H/M+D/H+D/M+H/
  M+H+D metrics, deltas, and Teacher-correct subset diagnostics.
- [ ] Part C is blocked: no existing Bartoldson CE-PGD20 replay for the frozen
  `[104,109,114]` development sensitivity was found.  No IRT endpoint is
  inferred from a different attack or silently reconstructed.
- [ ] Parts D--J (loss dry-run, epoch-79 randomized mechanism pilot, component
  ablation, deployable selector, and long-horizon training) remain closed until
  the evidence gate and coefficient dry-run are reviewed against this report.

- [x] 2026-08-09: Retrieved the existing Chen L2/L4 epoch-79 W&B `last:v15`
  parent artifacts without downloading the multi-GB run bundle.  Parent hashes,
  optimizer/scheduler/RNG/sample-state payloads, and scheduler milestones were
  checked.
- [x] 2026-08-09: Generated selection-time-only Route A/B masks and matched
  random controls for both Chen seeds.  Fixed q candidates are 5% and 10%;
  no future endpoint or test data was read by the selector command.
- [x] 2026-08-09: Completed one fixed-batch GPU loss/gradient scale dry-run per
  parent.  Conservative candidates are Route A `(KD=0.5, CE=0.25)` and Route B
  `(KD=1.0, CE=0.25)`.  This is not a performance result.
- [ ] Route A/B runtime arm schema and selected-only adversarial CE branch are
  not yet implemented; no causal training pilot has been launched under an
  incorrect legacy arm contract.

Canonical CPU result: `docs/experiments/ffnr_next_evidence_v1.json`.
