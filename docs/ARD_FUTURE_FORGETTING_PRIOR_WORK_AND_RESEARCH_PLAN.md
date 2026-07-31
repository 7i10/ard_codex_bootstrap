# ARDにおける将来Robust Forgetting予測の先行研究と今後の研究方針

**作成日:** 2026-07-31
**対象:** Adversarial Robustness Distillation（ARD）における、学生の学習履歴、将来のrobust forgetting、教師信号、sample-wise intervention
**位置付け:** 先行研究で既に示されていること、今回の実験で新たに得られたこと、今後提案する研究を明確に分離する。

---

## 0. 結論

主要な一次文献を確認した範囲では、以下の三つはそれぞれ先行研究に存在する。

1. **通常学習において、サンプルごとの学習・忘却履歴が安定したデータ特性を表すこと**
2. **敵対的訓練において、adversarial learning stabilityがサンプル品質やrobust overfittingと関係すること**
3. **ARDにおいて、教師の過信、transfer consistency、学生にとってのrobustly unlearnable samplesが蒸留失敗と関係すること**

しかし、主要文献の範囲では、次の組合せはまだ確立されていない。

> **ARDの対象学生自身の早期学習履歴から、将来robust forgettingするサンプルをprospectiveに予測し、その予測力をteacher entropyと直接比較した上で、予測結果をsample-wiseな蒸留介入へ接続する研究**

今回のseed-0解析では、historical student riskがteacher low-entropy riskよりfuture forgettingを大幅によく予測した。一方、そのstudent riskをteacher riskと乗算し、uniform target softeningへ直結した現行v1は性能改善に失敗した。

したがって、今後の研究は次の順序で進める。

\[
\boxed{
\text{予測信号の妥当性}
\;\rightarrow\;
\text{介入対象の妥当性}
\;\rightarrow\;
\text{介入方法の妥当性}
}
\]

研究の中心は、単一の新しいweight式を提案することではない。

> **どのサンプルが将来不安定になるかを学生履歴から予測し、そのサンプルに何を行うべきかを因果的に切り分ける。**

---

# 1. 用語の整理

本研究では、以下を混同しない。

## 1.1 現在のadversarial difficulty

現在checkpointで、対象サンプルが敵対的に分類しにくい程度である。

例:

\[
m_i^{(t)}
=
p_{S_t}(y_i\mid x_{i,\mathrm{adv}}^{(t)})
-
\max_{c\neq y_i}
p_{S_t}(c\mid x_{i,\mathrm{adv}}^{(t)})
\]

- \(m_i^{(t)}>0\): 正解クラスが最大
- \(m_i^{(t)}\approx0\): 境界付近
- \(m_i^{(t)}<0\): 誤分類

これは現在時点の状態であり、将来も学習不能であることを意味しない。

## 1.2 Robust learning stability

複数epochにわたり、そのサンプルを敵対的条件でどの程度安定して正解できたかを表す。

例:

\[
s_i
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf 1[
S_t(x_{i,\mathrm{adv}}^{(t)})=y_i
]
\]

## 1.3 Robust forgetting

一度敵対的条件で正解できたサンプルが、後のcheckpointで誤分類へ戻ることである。

\[
F_i^{(t)}
=
\mathbf 1[
S_{t-1}(x_{i,\mathrm{adv}})=y_i
\land
S_t(x_{i,\mathrm{adv}})\neq y_i
]
\]

本研究の主要なprospective outcomeは、ある履歴時点以後にforgetting eventを起こすかである。

## 1.4 Robustly unlearnable sample

特定の学生architecture、脅威モデル、学習条件の下で、複数seed・複数訓練法でも一貫して頑健正解できないサンプルを指す。

単一runのnegative marginや一度のforgettingだけでは、robustly unlearnableとは呼ばない。

## 1.5 Training utility

そのサンプルを現在の学習更新に用いることで、validation/test robustnessが改善するかを指す。

重要な区別は次である。

\[
\boxed{
\text{将来忘却されやすい}
\neq
\text{学習から除外すべき}
}
\]

予測精度と介入効果は別の研究問題である。

---

# 2. 先行研究で既に示されていること

## 2.1 通常学習におけるexample forgetting

### Toneva et al., ICLR 2019

**An Empirical Study of Example Forgetting during Deep Neural Network Learning**

Tonevaらは、通常の教師あり学習で、訓練サンプルが正解から誤分類へ遷移することをforgetting eventとして定義した。

主な知見は以下である。

- サンプルごとのforgetting頻度には大きな差がある。
- forgettingしやすいサンプルとunforgettableなサンプルの順位は、architectureやseedをまたいで一定の安定性を持つ。
- forgettingしやすいサンプルは、境界付近、曖昧、誤ラベルなどの可能性が高い。
- 多くのunforgettable examplesを除いても性能を維持できる場合がある。
- 一方、forgettingされる例は情報量の高いhard exampleでもあり、単純除外が常に適切とは限らない。

### 本研究との関係

これは「学生自身の学習履歴にサンプル固有の情報がある」という基礎的な先行研究である。

ただし、

- 通常学習である
- adversarial examplesを扱わない
- ARD教師信号を扱わない
- 将来robust forgettingをteacher entropyと比較しない

という違いがある。

---

## 2.2 敵対的訓練におけるlearning stability

### Dong, Liu, Shang, 2021

**Data Quality Matters for Adversarial Training: An Empirical Study**

この研究は、敵対的条件で各サンプルが正解されたepochの割合をadversarial learning stabilityとして扱った。

主な知見は以下である。

- learning stabilityのサンプル順位は、seed、訓練期間、PGD-AT、TRADES、MART、architectureをまたいで比較的一貫する。
- learning stabilityが低いサンプルはrobust overfittingへ強く関係する。
- 低stabilityサンプルを除外すると、robust test accuracyやrobust overfittingが改善する場合がある。
- サンプルごとの敵対的学習履歴は、ATにおけるデータ品質の代理量として機能し得る。

### 本研究との関係

現在の結果に最も近いAT先行研究である。

ただし、この研究のlearning stabilityは基本的に全訓練履歴から得る**事後指標**であり、次の点は十分扱っていない。

- 学習前半の履歴から、後半のforgettingをprospectiveに予測すること
- teacher entropyとの直接比較
- ARDにおける蒸留信号のroute選択
- 予測と介入utilityを独立した実験として比較すること

したがって、本研究の新規性候補は「履歴を使うこと」自体ではなく、**早期prospective prediction、teacher signalとの比較、ARD interventionへの接続**にある。

---

## 2.3 現在の難しさに基づくATのsample-wise制御

以下のAT研究は、現在のmargin、attack difficulty、誤分類状態などに基づいてサンプル処理を変える。

| 方向 | 代表例 | 主に測るもの | 典型的な処理 |
|---|---|---|---|
| 誤分類例重視 | MART | 現在の誤分類 | misclassified examplesを強く学習 |
| 境界近傍重視 | GAIRAT | PGDが誤分類へ到達するstep数 | boundary近傍へ大きなweight |
| 確率margin | MAIL | true-class確率と最大誤答確率の差 | instance reweighting |
| small-margin例の損失変更 | SOVR | 現在のsmall margin | lossを変更 |
| 攻撃を弱める | Friendly AT | current attack difficulty | early-stopped PGD |
| vulnerable data選択 | VDAT等 | current vulnerability | 計算予算配分 |

### 本研究との違い

これらは主として、

\[
\text{現在難しいか}
\]

を利用する。

本研究が対象とするのは、

\[
\text{現在までの履歴から、将来forgettingするか}
\]

である。

また、現在難しい例を強く学ぶ研究と、低stability例を除外・弱化する研究が両方存在する。この矛盾は、難しさ、学習可能性、utilityが同じ概念ではないことを示している。

---

## 2.4 他分野における将来forgetting予測

### Hacohen & Tuytelaars, ICML 2025

**Predicting the Susceptibility of Examples to Catastrophic Forgetting**

continual learningで、learning speedなどから各例が将来catastrophic forgettingを受けやすいかを予測し、replay sample選択へ使う。

### Jin & Ren, ICML 2024

**What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement**

language model refinementで、更新後に忘却される例を予測し、予測された例をreplayする。

### Zhou et al., AISTATS 2021

**Curriculum Learning by Optimizing Learning Dynamics**

訓練中のresidualやlearning dynamicsを用いて、現在学ぶべきfrontier examplesを選ぶ。

### 本研究との関係

「将来忘れる例を予測し、介入対象へ使う」という一般的な発想には先行研究がある。

ただし、これらは、

- continual learning
- language model update
- standard curriculum learning

であり、static CIFAR adversarial trainingやARDのrobust forgettingとは異なる。

---

# 3. ARDにおいて先行研究が扱っていること

## 3.1 IAD: 教師信頼性に応じたroute変更

**Reliable Adversarial Distillation with Unreliable Teachers, ICLR 2022**

IADは、教師を常に信頼せず、自然例・敵対的例上の教師性能に応じて教師targetへの依存を調整する。

これは、

> 教師が信頼できない場合に蒸留routeを変える

という重要な先行例である。

ただし、対象学生の長期学習履歴からfuture forgettingを予測しているわけではない。

---

## 3.2 SAAD: transfer consistencyとteacher entropyによるsample weighting

**Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation, TMLR 2026**

SAADは、学生生成攻撃が教師にも意味のある応答を引き起こすかをTASとして捉え、teacher entropyを安価なproxyとしてsample-wise weightingに使う。

SAAD-Cでは、敵対的蒸留に使いにくいサンプルをclean distillation側へrouteする。

これは、

> すべてのサンプルへ同じ教師信号を与えない

という先行研究である。

ただし、

- teacher entropyが主な観測量
- 学生の長期履歴をfuture forgetting predictorとして使わない
- teacher entropyとstudent historyのprospective predictive powerを比較しない

という違いがある。

---

## 3.3 Robustly Unlearnable Setと教師過信

**Toward Understanding Adversarial Distillation: Why Robust Teachers Fail, 2026**

この研究は、対象学生にとって頑健に表現しにくいサンプル上で、教師が高確信監督を与えると、学生がノイズ的特徴を記憶し、robust overfittingを起こし得ることを理論・実験的に示す。

主要な対処は、

- unlearnable set上で高entropyの教師を選ぶ
- 教師選択の段階でbad teacherを避ける

ことである。

### 未解決として残る点

- 既に選んだ教師を訓練中にどう修復するか
- 現在の対象学生自身から将来forgettingする例をどう早期予測するか
- 予測された例を除外、弱化、強化、route変更のどれに使うべきか
- 学習可能性とtraining utilityをどう分離するか

---

# 4. 質問への直接回答

## 4.1 「教師entropyより学生履歴の方が将来forgetting予測に有効」に直接対応する先行研究はあるか

### 回答

**近い先行研究はあるが、同じ問題設定の直接的な先行研究は主要一次文献では確認できない。**

既存研究は次のように分かれている。

| 既存研究 | 学生履歴 | 将来forgetting予測 | adversarial | ARD教師entropyとの比較 |
|---|---:|---:|---:|---:|
| Toneva et al. | あり | 主に事後的forgetting解析 | なし | なし |
| Data Quality Matters | あり | 主に全履歴のstability | あり | なし |
| Hacohen 2025 | あり | あり | なし | なし |
| Jin & Ren 2024 | あり | あり | なし | なし |
| SAAD | 間接的にcurrent student attack | なし | あり | entropyを使用 |
| Why Robust Teachers Fail | proxy unlearnable set | 教師選択 | あり | entropyを使用 |
| 今回の解析 | **対象学生の履歴** | **epoch 99→199をprospective予測** | **あり** | **直接比較** |

したがって、以下は新規性候補になる。

> **ARDで、学生自身のadversarial learning historyがteacher entropyより将来robust forgettingをよく予測することを、時間順序を守ったheld-out解析で示す。**

ただし、現時点の結果はseed 0のrunに条件付けられているため、複数seed、logging-only RSLAD、別architectureで再確認する必要がある。

---

## 4.2 将来forgettingすると予測されたサンプルをどう扱う研究はあるか

### 回答

**処理の候補は複数あるが、ARDで確立した標準解はない。**

既存研究の処理は以下に分かれる。

| 処理 | 先行する考え方 | 利点 | 主なリスク |
|---|---|---|---|
| 除外 | Data Quality Matters | ノイズ記憶を抑える | 有益なhard exampleを失う |
| loss downweight | sample reweighting、SAAD | 低コスト | 学習機会を失い自己実現的にhard化 |
| hard exampleを強化 | MART、GAIRAT、MAIL、SOVR | 境界改善 | unlearnable/noisy例へ過適合 |
| attackを弱化 | Friendly AT | 学習可能性を上げる | threat objectiveを弱める |
| teacherを弱化 | SAAD、temperature調整 | sharp supervisionを抑える | 正しく有益な教師信号も失う |
| clean distillationへroute | SAAD-C | clean情報を保持 | robust learningへの寄与が減る |
| true-labelへanchor | IAD的routing、label correction | teacher誤りを補正 | hard one-hot監督を再導入 |
| replay | continual learning研究 | forgettingを直接補償 | static ATでは計算増加・再過適合 |
| early stopping | robust overfitting研究 | last degradationを避ける | best精度自体は改善しない |

したがって、

\[
\boxed{
\text{将来忘れそう}
\Rightarrow
\text{何をすべきか}
}
\]

は未解決である。

これが今後の研究の中心になる。

---

# 5. 今回の独自結果

以下は先行研究ではなく、現在のリポジトリで得られたseed-0探索結果である。

## 5.1 Student historyがforgetting予測の中心

epoch 99のsignalからepoch 199までのsubsequent forgettingを予測した。

比較したモデルは以下である。

\[
\begin{aligned}
\text{Teacher-only}&:\quad[c_i]\\
\text{Student-only}&:\quad[u_i]\\
\text{Main effects}&:\quad[u_i,c_i]\\
\text{Main+product}&:\quad[u_i,c_i,u_ic_i]
\end{aligned}
\]

ここで、

\[
u_i
=
\frac{1-\operatorname{EMA}(m_i)}{2}
\]

は学生のhistorical robust-margin risk、

\[
c_i
=
1-\frac{H_T(x_i^{adv})}{\log C}
\]

はteacher low-entropy riskである。

### 主要結果

- Student-onlyはTeacher-onlyより、全4 runでAUROCが高い。
- 改善幅はChenで約+0.036〜+0.046。
- Bartoldsonでは約+0.332。
- Student-onlyへteacher entropyを加えても、AUROC変化は−0.0021〜+0.0010。
- product項の追加はAUROC +0.0009〜+0.0035で、主な予測情報源ではない。
- product項はlog-lossをわずかに改善し、calibrationへ微小な寄与を持つ可能性がある。

### 現時点の解釈

\[
\boxed{
\text{future forgettingの予測情報の大部分はstudent historyにある}
}
\]

teacher entropyは、student historyを既に知った後では、forgetting順位予測へほとんど情報を追加しない。

---

## 5.2 現行Joint介入は成功しなかった

現行Jointは、

\[
r_i=u_ic_i,
\qquad
\rho_i=0.5r_i
\]

としてteacher targetをuniformへsoftenした。

しかし、

- ChenではRSLADよりbest AAが低下
- Bartoldsonではbest AAが+0.20 ppに留まる
- Bartoldsonのlast AAとrobust-overfitting gapは悪化
- Entropy-onlyの方がBartoldsonのROを抑制

となった。

したがって、以下は支持されない。

\[
\boxed{
u_ic_i
\text{をそのままuniform softening強度へ使う}
}
\]

---

# 6. 先行研究と提案手法の境界

## 6.1 先行研究として主張できること

以下は既存研究である。

- training dynamicsやforgetting eventがサンプル特性を表す
- adversarial learning stabilityがrobust overfittingと関連する
- current difficultyやmarginによるsample reweighting
- unreliable teacherに対するKD route変更
- teacher entropy / TAS proxyによるsample weighting
- unlearnable samples上の教師過信がARD失敗に関係する
- forgetting予測をreplayへ使うcontinual learning / LM研究

## 6.2 今回の独自結果として主張候補になること

十分な追加検証後に、以下が独自貢献候補になる。

1. **ARDにおいて、学生の早期adversarial learning historyが将来robust forgettingを予測する。**
2. **その予測力はteacher entropyより強く、teacher entropyはstudent historyへほとんど追加ranking情報を与えない。**
3. **予測力の高いsignalを単純なsofteningへ使っても性能改善には直結しない。**
4. **prognostic signalとtraining utilityを分離しなければならない。**

## 6.3 今後提案する手法

現段階では、単一の完成手法ではなく、次の研究枠組みを提案する。

### 仮称

**Student-History-Guided Robust Forgetting Forecasting and Intervention**

### 構成

\[
\boxed{
\text{Predict}
\rightarrow
\text{Validate treatment utility}
\rightarrow
\text{Route supervision}
}
\]

#### Predict

学生履歴からfuture forgetting probabilityを推定する。

候補特徴:

- robust margin EMA
- robust correctness frequency
- first robustly learned epoch
- longest correct streak
- 正→誤遷移数
- 誤→正遷移数
- margin slope
- margin variance

#### Validate treatment utility

同一の固定maskを使い、介入だけを変える。

- no intervention
- uniform softening
- KD downweight
- temperature increase
- true-label anchor
- clean-only route
- attack easing
- exclusion

#### Route supervision

教師情報はfuture forgettingの主予測信号ではなく、介入の安全性を判断する補助信号として使う。

候補:

- teacher correctness
- teacher safety margin
- teacher wrong-confidence
- clean→adv margin低下
- clean→adv JS divergence
- student誤分類時のteacher conditional utility

---

# 7. 今後の実験方針

## Phase 1: Frozen artifact audit

新規訓練なしで、既存artifactの予測情報を確定する。

### 必須比較

\[
[c_i]
\quad
[u_i]
\quad
[u_i,c_i]
\quad
[u_i,c_i,u_ic_i]
\]

指標:

- AUROC
- AUPRC
- log-loss
- class-stratified bootstrap CI
- calibration curve
- prevalence

### 判定

student-onlyが再現的にteacher-onlyを上回るかを確認する。

---

## Phase 2: Logging-only RSLAD

signalが訓練介入によって変化したという交絡を除く。

RSLAD lossは変更せず、以下だけ記録する。

- student margin trajectory
- correctness history
- forgetting events
- teacher entropy
- teacher correctness
- teacher margin response

### 実行条件

- Chen RSLAD seeds 1/2
- Bartoldson RSLAD seeds 1/2
- early window: epoch 10/20/40/60/80/100
- outcome: 以後のforgetting、final error、best→last degradation

### 目的

- どの時点からforgetting予測が可能か
- seed間で予測力が維持されるか
- teacher間で同じpredictorが使えるか

---

## Phase 3: Frozen oracle intervention

RSLAD由来のpre-intervention maskを固定する。

比較:

1. offline oracle
2. class-matched random × 3以上
3. same-\(\rho\) permutation
4. no intervention

全armで以下を揃える。

- sample数
- class分布
- \(\rho\) multiset
- optimizer
- attack
- schedule

### Go基準

- oracle best AA ≥ random平均 +0.5 pp
- oracleが全randomを上回る
- clean低下 ≤0.5 pp

### No-Go基準

- oracleがrandom平均以下

No-Goならuniform softening路線を停止する。

---

## Phase 4: Intervention screening

oracleまたは固定student-risk maskを使い、介入だけを比較する。

| Arm | 処理 |
|---|---|
| Baseline | RSLAD |
| Softening | teacher targetをuniformへ混合 |
| Downweight | adversarial KD weight低下 |
| Temperature | teacher temperature上昇 |
| True-label anchor | teacher targetをground truth側へ補正 |
| Clean-only | adversarial KDを止めclean KDを保持 |
| Attack easing | \(\epsilon\)またはstep数を一時的に低下 |
| Exclusion | adversarial updateから除外 |

### 目的

次を切り分ける。

- predictorが悪い
- interventionが悪い
- 両方が悪い
- 特定のteacher stateでのみ有効

---

## Phase 5: Teacher reliability routing

student-high-riskサンプルに限定して、teacher stateを判断する。

### 初期router

| Student | Teacher | 処理候補 |
|---|---|---|
| low risk | 任意 | 通常RSLAD |
| high risk | correct、positive safety margin | KDを維持 |
| high risk | correctだが無反応 | temperature / clean-onlyを比較 |
| high risk | wrong-confidence | KD downweightまたはtrue-label anchor |

teacher entropy単独や\(u_ic_i\)の積を主routerにしない。

---

## Phase 6: 確認実験

開発に使わないseed 1/2で固定評価する。

### v2 Go基準

- Entropy比平均best AA ≥ +0.5 pp
- 両seedで差が非負
- robust-overfitting gap非悪化
- clean低下 ≤0.5 pp

成功した場合のみ、

- Chenで非劣化確認
- MobileNetV2
- CIFAR-100
- Tiny-ImageNet

へ進む。

---

# 8. 成功・失敗を切り分ける研究設計

本研究では、結果を次のように分類する。

## 結果A: Signal成功、介入成功

- student historyがfuture forgettingを予測
- oracleがrandomを上回る
- 実用predictorとroutingがEntropyを上回る

→ 新しいARD method paperへ進む。

## 結果B: Signal成功、uniform softening失敗、別介入成功

- predictorは有効
- target softeningは不適切
- downweight、anchor、clean-only等が有効

→ 「予測と介入の分離」を主貢献にする。

## 結果C: Signal成功、すべての介入失敗

- future forgettingは予測できる
- しかしsample-wise制御ではrobust accuracyを改善できない

→ analysis/benchmark paperとして成立可能。
「predictability does not imply training utility」というnegative resultが中心になる。

## 結果D: logging-only・複数seedでSignalも再現しない

- 現在の結果は介入runまたはseed 0固有

→ student-history主路線を停止し、DMRDなど「何を蒸留するか」の別方向へpivotする。

---

# 9. 主張してはいけないこと

現時点では、以下を主張しない。

- robustly unlearnable setをオンラインで正確に特定した
- student riskが因果的にforgettingを起こす
- teacher entropyが不要である
- \(u_ic_i\)の積が有害教師信号を表す
- future forgetting sampleは除外すべき
- seed 0のsample-level bootstrapがtraining-seed uncertaintyを表す
- 現行JointがRSLAD/SAADを改善した

---

# 10. 論文としての最終形

## 10.1 Interventionが成功した場合

### 仮タイトル

**Predict Forgetting from the Student, Route Supervision with the Teacher: Student-History-Guided Adversarial Robust Distillation**

### 貢献

1. student historyによるfuture robust forgetting予測
2. teacher entropyとの直接比較
3. prognostic signalとprescriptive signalの分離
4. teacher reliability routing
5. RSLAD/SAADを上回る新規ARD手法

## 10.2 Interventionが成功しなかった場合

### 仮タイトル

**Predictable but Not Necessarily Actionable: Robust Forgetting Dynamics in Adversarial Distillation**

### 貢献

1. future robust forgettingの予測可能性
2. teacher entropyの予測限界
3. learnabilityとtraining utilityの分離
4. 複数の介入が失敗する条件
5. future ARD研究のbenchmarkとnegative result

---

# 11. 推奨する直近の実行順序

```text
1. 現在のfrozen-oracle vs randomを完了
2. logging-only RSLAD seeds 1/2を準備
3. early-window predictorを固定
4. Student-only / teacher-only / additive / productを複数seedで比較
5. oracleの結果に基づきuniform softeningを継続または停止
6. 同一maskでintervention screening
7. teacher correctness/margin-response routerを追加
8. 開発に使わないseed 1/2でv2を判定
9. 成功時のみMobileNetV2・CIFAR-100へ拡張
10. 失敗時はDMRD等の別方向へpivot
```

---

# 12. 主要参考文献

1. Toneva, M. et al. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.** ICLR 2019. arXiv:1812.05159.
2. Dong, C., Liu, L., Shang, J. **Data Quality Matters for Adversarial Training: An Empirical Study.** 2021. arXiv:2102.07437.
3. Rice, L., Wong, E., Kolter, J. Z. **Overfitting in Adversarially Robust Deep Learning.** ICML 2020.
4. Wang, Y. et al. **Improving Adversarial Robustness Requires Revisiting Misclassified Examples.** ICLR 2020.
5. Zhang, J. et al. **Geometry-Aware Instance-Reweighted Adversarial Training.** ICLR 2021.
6. Liu, F. et al. **Probabilistic Margins for Instance Reweighting in Adversarial Training.** arXiv:2106.07904.
7. Zhang, J. et al. **Attacks Which Do Not Kill Training Make Adversarial Learning Stronger.** ICML 2020.
8. Hacohen, G., Tuytelaars, T. **Predicting the Susceptibility of Examples to Catastrophic Forgetting.** ICML 2025.
9. Jin, X., Ren, X. **What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement.** ICML 2024.
10. Zhou, D. et al. **Curriculum Learning by Optimizing Learning Dynamics.** AISTATS 2021.
11. Zhu, J. et al. **Reliable Adversarial Distillation with Unreliable Teachers.** ICLR 2022.
12. Lee, H., Chung, H. W. **Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation.** TMLR 2026. arXiv:2512.10275.
13. Lee, H., Chung, H. W. **Toward Understanding Adversarial Distillation: Why Robust Teachers Fail.** 2026. arXiv:2605.21999.
14. Zi, B. et al. **Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better.** ICCV 2021.
15. Croce, F., Hein, M. **Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks.** ICML 2020.

---

# 13. 最終判断

先行研究は、

- 学習履歴がサンプル品質を表す
- adversarial learning stabilityがROと関係する
- 教師過信がARD失敗と関係する
- future forgetting予測が他分野で利用可能である

ことを個別に示している。

しかし、

> **ARDで、学生自身の早期敵対的学習履歴がteacher entropyよりfuture robust forgettingをよく予測することを示し、その予測とsample-wise interventionのutilityを分離評価する**

研究には明確な余地がある。

したがって、次の主軸が最も妥当である。

\[
\boxed{
\text{Student historyでforgettingを予測する}
+
\text{介入の因果効果を固定maskで比較する}
+
\text{教師情報はroute選択へ限定する}
}
\]

この順序なら、最終的な介入が失敗しても、予測研究とnegative resultが独立した成果として残る。
