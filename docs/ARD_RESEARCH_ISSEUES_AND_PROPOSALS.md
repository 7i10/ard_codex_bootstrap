# Adversarial Robustness Distillation（ARD）の研究課題整理と独立提案

**更新日:** 2026年7月31日  
**対象:** 敵対的訓練（Adversarial Training; AT）とKnowledge Distillation（KD）を組み合わせたAdversarial Robustness Distillation（ARD）  
**目的:** これまでの議論と提示された要約を、重要点を落とさず補足・整理し、各研究課題に対する独立した研究提案を示す。

> 本稿では、各提案をいったん独立した研究テーマとして扱う。将来的に複数提案を統合する可能性はあるが、現段階では統合システムとしての整合性や卒論全体の構成は考慮しない。

---

## 目次

1. [ARDの基本的位置づけ](#1-ardの基本的位置づけ)
2. [ARDが通常のATを上回り得る理由](#2-ardが通常のatを上回り得る理由)
3. [現在のARD研究を貫く中心問題](#3-現在のard研究を貫く中心問題)
4. [研究課題1：良い教師をどのように判断するか](#4-研究課題1良い教師をどのように判断するか)
5. [研究課題2：学生の学習可能性を簡単に求められるか](#5-研究課題2学生の学習可能性を簡単に求められるか)
6. [研究課題3：教師から何を蒸留すべきか](#6-研究課題3教師から何を蒸留すべきか)
7. [研究課題4：有益で共適応しない教師・攻撃機構](#7-研究課題4有益で共適応しない教師攻撃機構)
8. [研究課題5：教師信号の信頼性をどう判断するか](#8-研究課題5教師信号の信頼性をどう判断するか)
9. [研究課題6：robust overfittingをどう監視・抑制するか](#9-研究課題6robust-overfittingをどう監視抑制するか)
10. [研究課題7：クラスごとの頑健性格差をどう縮小するか](#10-研究課題7クラスごとの頑健性格差をどう縮小するか)
11. [各独立提案の比較](#11-各独立提案の比較)
12. [共通評価プロトコル](#12-共通評価プロトコル)
13. [主要参考文献](#13-主要参考文献)

---

# 1. ARDの基本的位置づけ

## 1.1 ARDとは何か

ARDは、概念的には次の組合せである。

$$
\boxed{
\mathrm{ARD}
=
\mathrm{Adversarial\ Training}
\times
\mathrm{Knowledge\ Distillation}
}
$$

通常のATは、学生モデル自身に対する敵対的サンプルを生成し、正解ラベルに分類するよう学習する。

$$
\min_{\theta}
\mathbb{E}_{(x,y)}
\left[
\max_{\|\delta\|\leq\epsilon}
\mathcal{L}_{\mathrm{CE}}
\left(S_{\theta}(x+\delta),y\right)
\right]
$$

ARDでは、このAT損失に、教師モデルの出力・特徴・局所応答などを模倣する蒸留損失を加える。

$$
\mathcal{L}_{\mathrm{ARD}}
=
\mathcal{L}_{\mathrm{AT}}
+
\lambda_{\mathrm{KD}}
\mathcal{L}_{\mathrm{KD}}
$$

したがって、ARDは単に「教師モデルを圧縮するKD」ではない。多くのARDでは学生自身への敵対的サンプル生成を残すため、**ARDはATを教師信号によって誘導・正則化する枠組み**と捉えるのが正確である。

## 1.2 soft labelとは何か

教師のlogitを $z_T(x)$、temperatureを $\tau$ とすると、soft labelは次で表される。

$$
p_{T,k}^{(\tau)}(x)
=
\frac{\exp(z_{T,k}(x)/\tau)}
{\sum_j \exp(z_{T,j}(x)/\tau)}
$$

例えば、犬画像に対して、

$$
(\text{犬}:0.80,\ \text{猫}:0.10,\ \text{狼}:0.06,\ \text{船}:0.001,\ldots)
$$

のような分布である。これは必ずしも校正された真の確率ではなく、教師が持つ**クラス間の相対的なlogit構造**を表す。

## 1.3 label smoothingとの関係

label smoothingは、one-hot labelを平滑化する。

$$
q_y=1-\varepsilon,
\qquad
q_{k\neq y}=\frac{\varepsilon}{C-1}
$$

label smoothingとsoft-label distillationの共通点は、正解クラスへ過剰に確信することを抑え、最適化を滑らかにする点である。

一方、両者には重要な違いがある。

| 項目 | Label smoothing | 教師soft label |
|---|---|---|
| 非正解クラスへの配分 | 原則一様 | 入力・教師ごとに異なる |
| クラス間関係 | 表現しない | 猫と狼、船と飛行機などの相対関係を含み得る |
| サンプル難易度 | 原則反映しない | 教師の確信度・曖昧さとして反映され得る |
| 教師の頑健性 | 利用しない | robust teacherの応答を利用できる |
| 誤りの転写 | ない | 教師誤り・過信を学生へ転写する危険がある |

したがって、ARDの利点は、

$$
\text{soft targetによる一般的な正則化}
+
\text{robust teacher固有の入力依存情報}
$$

に分けて考えるべきである。

---

# 2. ARDが通常のATを上回り得る理由

## 2.1 hard labelより多くの制約を与える

通常のATにおけるone-hot labelは、基本的に「正解クラスを最大にせよ」としか教えない。一方、教師soft labelは全クラスの相対関係を制約する。

$$
D_{\mathrm{KL}}
\left(
p_T(x)
\|p_S(x_{\mathrm{adv}})
\right)
$$

を用いると、学生は正解クラスだけでなく、競合クラスとの相対関係も模倣する。これにより、学生が探索する解空間を、教師が獲得した頑健な関数の近傍へ誘導できる可能性がある。

## 2.2 小型学生の限られた容量を有効な解へ配分する

小型モデルの直接ATが難しい理由は、単なるパラメータ不足だけではない。

- 各訓練点ではなく、周囲の摂動集合全体を正しく分類する必要がある。
- 標準精度に有効な非頑健特徴へ依存しにくくなる。
- 内側攻撃が学生の現在状態に応じて変わり、最適化対象が非定常になる。
- clean accuracy、robust accuracy、蒸留、特徴整合などの目的が競合する。
- 学習率、正規化、architecture、attack strength、weight decay、データ拡張の影響が大きい。

したがって、観測される性能不足は、

$$
\text{表現能力不足}
+
\text{最適化困難性}
+
\text{学習設計の不整合}
$$

の複合結果である。ARDは教師信号によって、学生の限られた自由度をより有効な頑健解へ割り当てることを狙う。

## 2.3 学生が教師を超え得る理由

ARD学生は教師だけを模倣するわけではない。一般には、

- ground-truth label
- teacher soft label
- 学生自身への敵対的サンプル
- 学生固有の正則化・architecture
- データ拡張

を同時に利用する。

教師が誤った場合でもground truthが補正でき、学生自身への攻撃によって教師にはない学生固有の脆弱性を修正できる。そのため、特定のclean accuracyまたはrobust accuracyで学生が教師を上回ることはあり得る。ただし、**学生が教師を超えることはARDの一般的保証ではない**。

## 2.4 ImageNet規模での証拠

KDIGAはImageNetでResNetとViTを含む頑健性転写を評価している。一方、ARD、RSLAD、IAD、PeerAiD、SAAD、Robustly Unlearnable Setの主要検証は、CIFAR-10/100やTiny-ImageNetが中心である。

したがって、現状は次のように整理できる。

- **ImageNet規模でも頑健性蒸留が可能である証拠はある。**
- **ImageNet-1Kで、同一学生の強いAT baselineを同一計算予算で安定して上回る標準ARDは未確立である。**
- ImageNetへの拡張を目指す提案は、追加attack数、teacher/peer forward、保存統計、1000クラスでの統計安定性を最初から考慮する必要がある。

---

# 3. 現在のARD研究を貫く中心問題

## 3.1 強い教師は必ずしも良い教師ではない

従来の直感は、

$$
A_{\mathrm{rob}}(T_1)>A_{\mathrm{rob}}(T_2)
\Rightarrow
A_{\mathrm{rob}}(S\leftarrow T_1)>
A_{\mathrm{rob}}(S\leftarrow T_2)
$$

である。しかし、現実にはこの単調関係は成立しない。教師を強くしても学生性能が飽和し、さらに強い教師で学生性能が低下する場合がある。

2026年の研究は、この原因として学生固有の**Robustly Unlearnable Set**を提示している。これは、対象学生・脅威モデル・学習条件の下で頑健に学習することが困難な訓練サンプル集合である。

教師がこの集合に対して、

$$
p_T(y\mid x)\approx1
$$

のような高確信度監督を与えると、学生は教師が利用する頑健特徴を表現できないまま、その教師出力へ合わせようとする。その結果、サンプル固有ノイズや非一般化特徴を記憶し、

- robust training accuracyは上がる
- robust validation/test accuracyは下がる
- peak-to-last degradationが拡大する

という**robust overfitting**が起こり得る。

> **現在の核心は、教師の強さそのものではなく、教師の確信度と学生の学習可能性の整合性である。**

## 3.2 学生攻撃が教師には攻撃でない問題

学生への攻撃を、

$$
x_{\mathrm{adv}}^S=x+\delta_S
$$

とする。これは学生の決定境界・勾配に最適化されているため、教師にはほとんど影響しない場合がある。

学生は完全に崩れているのに、教師がclean時とほぼ同じ極端な出力を返すと、教師soft labelは学生の困難さを表さない。

$$
T(x)\approx T(x_{\mathrm{adv}}^S)
$$

学生にとって困難なサンプルでこの無反応・過信が起きると、実現困難な教師信号を強制し、robust overfittingを促す可能性がある。

一方、学生攻撃が教師にも完全に転移し、教師まで誤分類する場合も望ましくない。

> **理想は「誤分類は教師へ転移しないが、難しさは教師のmargin・確信度へ転移する」状態である。**

すなわち、教師は、

$$
T(x_{\mathrm{adv}}^S)=y
$$

を維持しつつ、

$$
m_T(x_{\mathrm{adv}}^S)<m_T(x)
$$

となることが望ましい。教師は**label-robustかつdifficulty-sensitive**であるべきである。

## 3.3 現在の中心的な未解決問題

$$
\boxed{
\text{教師が持つ頑健性のうち、学生が学習・一般化できる部分だけを、
どのサンプルで、どの時点に、どの形式で転写するか}
}
$$

この問題を以下の7課題に分ける。

---

# 4. 研究課題1：良い教師をどのように判断するか

## 4.1 現状の教師選択指標

実用上の優先順位は、概ね次のように整理できる。

1. **実際に蒸留した学生のhold-out robust accuracy**
2. 学生生成攻撃上の教師正解率と予測分布
3. TAS ratioまたはentropy proxy
4. proxy unlearnable set上の教師entropy
5. 教師checkpoint自身のrobust overfitting状況
6. 教師自身のAutoAttack accuracy
7. 教師のモデルサイズ・パラメータ数

教師自身のrobust accuracyは候補の足切りには必要だが、教師間の最終順位を決めるには不十分である。

## 4.2 TASとは何か

Transferable Adversarial Sample（TAS）は、学生生成攻撃が教師にも教師自身への攻撃と類似した応答を引き起こすかを測る。

教師clean出力を $T(x)$、学生攻撃時出力を $T(x+\delta_S)$、教師自身への攻撃時出力を $T(x+\delta_T)$ とすると、概念的には、

$$
\mathrm{KL}
\left(T(x+\delta_S)\|T(x)\right)
\geq
\mathrm{KL}
\left(T(x+\delta_S)\|T(x+\delta_T)\right)
$$

ならTASと判定する。

TASはサンプルごとに定義できる。データセット全体で集約すればteacher-level TAS ratioになり、各サンプルの判定・scoreを用いればsample-wise weightingに使える。

## 4.3 TASの明確な弱点

**TASは教師信号の有用性を直接保証しない。**

- 教師が正解しているかを見ていない。
- 教師と学生が同じ誤クラスへ崩れる場合も高TASになり得る。
- 教師自身への攻撃 $\delta_T$ が弱ければ基準が不正確になる。
- 攻撃が教師へ転移することと、その教師信号が学生に有益であることは同じではない。
- 高TASは、教師と学生が有害な脆弱性を共有していることを示す場合もある。
- 正確なTASには学生攻撃だけでなく教師攻撃も必要であり、計算量が増える。

したがって、raw TAS ratioだけでなく、少なくとも、

$$
\begin{aligned}
r_{\mathrm{correct\text{-}TAS}}
&=\Pr[\mathrm{TAS}\land T(x_{\mathrm{adv}}^S)=y],\\
r_{\mathrm{wrong\text{-}TAS}}
&=\Pr[\mathrm{TAS}\land T(x_{\mathrm{adv}}^S)\neq y]
\end{aligned}
$$

を分離して見るべきである。

## 4.4 SAADとentropy proxy

正確なTASには教師自身への攻撃が必要である。SAADは追加計算を避けるため、

$$
H(T(x+\delta_S))
$$

をTASの代理量として用い、サンプルごとに蒸留weightを調整する。

SAADでは、蒸留結果の悪い教師は学生生成PGD入力上で低entropyになりやすく、その学生ではadversarial varianceとrobust overfittingが大きい傾向が報告されている。

ただし、

$$
\boxed{\text{高entropy}\neq\text{良い教師}}
$$

である。誤分類して迷っている教師や、一様分布を出す無知な教師も高entropyになり得る。

## 4.5 Unlearnable-Entropy

Unlearnable-Entropyは、対象学生にとって頑健に学習しにくいproxy set $\mathcal S_U$ 上で、教師がどの程度過信しているかを見る。

$$
Q_{\mathrm{UE}}(T)
=
\frac{1}{|\mathcal S_U|}
\sum_{i\in\mathcal S_U}
H(T(x_{i,\mathrm{adv}}))
$$

この指標は教師ランキングに用いられる。学生が学習困難なサンプルで教師が低entropy・高確信度を出すと、学生のノイズ記憶を促進するという仮説に基づく。

しかし、Unlearnable-Entropyにも次の問題が残る。

- 教師が誤分類しているのに高entropyなら、高く評価され得る。
- 一様分布を出す教師も最大entropyになる。
- proxy unlearnable setがreference studentに依存する。
- 実際の学生とreference studentが異なると集合がずれる。
- checkpointによってlearnable/unlearnableが変化する。
- 攻撃強度、$\epsilon$、学生容量に依存する。
- binary集合では境界サンプルを表現しにくい。
- seed・訓練法によって判定が変わる。
- ImageNet規模でのproxy構築コストと安定性が十分検証されていない。

> **Unlearnable-EntropyはTASの完全な改良版ではない。TASは教師–学生間の攻撃応答を測り、Unlearnable-Entropyは学生の表現限界と教師過信の不整合を測る。**

---

## 独立提案1：Student-Compatible Teacher Selection（SCTS）

### 目的

教師単体のAA精度ではなく、**対象学生に対する適合性**によって教師を事前選択する。

### 仮説

良い教師は、次を同時に満たす。

$$
\boxed{
\text{教師単体の健全性}
+
\text{学生攻撃上の正解性}
+
\text{正解維持下の適度な反応}
+
\text{unlearnable集合上の非過信}
}
$$

### 手順

#### 1. 教師候補の足切り

同一のnormalization、解像度、$\epsilon$、攻撃条件で、

- clean accuracy
- PGD accuracy
- AutoAttack accuracy
- step/restart増加時の安定性

を確認する。ここでは教師ランキングより、不健全な教師の除外を目的とする。

#### 2. reference studentを固定

対象学生と同じarchitectureを短期間ATし、reference checkpoint $S_{\mathrm{ref}}$ を作る。

#### 3. 学生攻撃上の教師適合性を測る

$$
A_{T\leftarrow S}
=
\Pr[T(x_{\mathrm{adv}}^{S_{\mathrm{ref}}})=y]
$$

に加え、安全margin率、correct-TAS、wrong-TAS、teacher entropyを測る。

$$
C_{\mathrm{safe}}
=
\Pr[m_T(x_{\mathrm{adv}}^S)>m_{\min}]
$$

#### 4. proxy unlearnable set上の挙動

教師のentropyだけでなく、

- accuracy
- true-class probability
- safety margin
- entropy

を同時に測る。

#### 5. Pareto選択

一つの恣意的な重み付きscoreに潰さず、

- 学生攻撃正解率が高い
- wrong-TASが少ない
- safety marginが正
- unlearnable集合で極端に過信しない

教師をPareto候補として残し、短いpilot distillationで最終選択する。

### 検証

- 教師自身AAと最終学生AAの相関
- TAS ratioと最終学生AAの相関
- UE entropyと最終学生AAの相関
- SCTS各指標と最終学生AAの相関
- short pilot順位とfull training順位の一致率

### ImageNet拡張

フルtrain setではなく、class-balanced subsetと固定reference studentを使えば、複数教師をフル蒸留するより低コストで実行できる。教師候補の事前学習コストも含めて比較する。

### 新規性とリスク

新規性は、teacher robustness、correctness、TAS、unlearnabilityを**学生互換性という統一視点で事前選択へ使うこと**にある。最大のリスクは、reference studentやcheckpointへの依存である。

---

# 5. 研究課題2：学生の学習可能性を簡単に求められるか

## 5.1 何を「学習可能性」と呼ぶか

少なくとも次を分ける必要がある。

1. **現在の脆弱性**  
   現checkpointで攻撃が成功するか。

2. **学習安定性**  
   学習中に敵対的条件で何度、どの程度安定して正解したか。

3. **将来のrobust learnability**  
   十分な訓練後に、対象学生と脅威モデルで頑健に学習できるか。

4. **training utility**  
   そのサンプルを現在の更新に用いることがvalidation robustnessを改善するか。

**learnabilityとutilityは同じではない。** 学習可能でも既に十分学習済みなら追加更新の価値は小さい。現在は難しくても、教師やcurriculumにより高いutilityを持つ可能性がある。

## 5.2 複数モデルによる参照定義

現時点で最も直接的なのは、複数の訓練法・seed・peak checkpointでのrobust correctnessを集約する方法である。

$$
L_i^*
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf 1[S_m(x_{i,\mathrm{adv}})=y_i]
$$

2026年の研究では、CIFARで6種類の訓練パラダイム×10 seedの60モデルを用い、全モデル正解をlearnable、全モデル誤分類をunlearnableとする厳密集合を構築した。

これは比較的直接的だが、非常に高コストである。

## 5.3 学習履歴における正解回数

各サンプルが敵対的条件で正解したepochの割合を、learning stabilityとする。

$$
s_i
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf 1[S_t(x_{i,\mathrm{adv}})=y_i]
$$

Data Quality Matters for Adversarial Trainingでは、learning stabilityのサンプル順位がseed、訓練期間、PGD-AT、TRADES、MART、architecture間で比較的一貫することが報告されている。

ただし、これは全学習後に得られる**事後指標**である。

単純なforgetting回数だけでは不十分である。

- 最初から最後まで正解：forgetting 0回
- 一度も正解しない：forgetting 0回

となるためである。

見るべきなのは、

- adversarial correctness率
- first robustly learned epoch
- correct streakの長さ
- 正→誤、誤→正の遷移回数
- peak generalization時期の安定正解率

である。

> **重要なのはforgetting回数そのものより、「学生が最もよく一般化している時期に、そのサンプルを安定して頑健正解できるか」である。**

## 5.4 marginはlearnabilityを表すか

GAIRATは、PGDが誤分類へ到達するまでのstep数を、境界までの距離の近似として利用する。しかし、攻撃経路依存で離散的である。

MAILは、true-class probabilityと最大誤クラスprobabilityとの差によるprobabilistic marginを利用する。

marginはATで十分研究されているが、基本的には、

$$
\text{現在の境界までの近さ}
$$

を測る。現在marginが負でも、後で学習できる可能性がある。したがって、単一時点のmarginだけでは将来的learnabilityは分からない。

一方、

- margin平均
- marginの傾き
- 分散
- 正負反転回数
- early-windowでの改善速度

などのtrajectoryは補助指標になり得る。

## 5.5 restart、勾配、EMA、複数$\epsilon$

| 候補 | 主に測るもの | 将来learnabilityへの位置づけ |
|---|---|---|
| 複数restart間の予測分散 | attack uncertainty、局所境界の複雑さ | 直接的根拠は弱い |
| 勾配ノルム | 現在の更新強度・損失感度 | 有益な難例とノイズを分離できない |
| 勾配方向安定性 | 更新方向・模倣整合性 | 高コストで未確立 |
| EMA学生との一致 | temporal stability | 両方誤る場合があるため正解条件が必要 |
| 複数$\epsilon$での正解 | 現在のrobust radius形状 | 有望だが複数attackが必要 |
| teacher–student feature alignment | 模倣可能性・表現差 | architecture依存、learnabilityとは未確立 |

> **現状、将来のrobust learnabilityを学習初期に最も簡単・正確に表現する指標は確立されていない。これはARDだけでなくAT全般の研究課題である。**

## 5.6 難しいサンプルを強く学習すべきか

現在の研究では、hard、unstable、easy/small-loss、unlearnable、importantといった異なる概念が混在する。

- 境界に近いhard sampleを強く学習する研究がある。
- 低stability sampleがrobust overfittingに関連する研究がある。
- small-loss sampleへの過度な適合がoverfittingを生むという研究もある。

これらは必ずしも矛盾ではない。各研究が測る「難しさ」が異なるためである。

> **サンプルが難しいと判定できても、そのサンプルを強く学習すべきかは別問題である。**

---

## 独立提案2：Robust Learnability Proxy Benchmark（RLPB）

### Step 1：参照learnabilityを構築

複数の、

- seed
- PGD-AT、TRADES、FastAT、ARD
- 学生architecture
- peak checkpoint

から、

$$
L_i^*
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf 1[S_m(x_i^{\mathrm{adv}})=y_i]
$$

を作る。全正解・全誤りのbinary集合だけでなく、$L_i^*\in[0,1]$ の連続値を用いる。

### Step 2：安価な早期指標を比較

学習最初の10–30%から、

- robust correctness EMA
- first robustly learned epoch
- correctnessの上昇傾向
- longest correct streak
- adversarial marginの平均・傾き・分散
- 複数$\epsilon$での正解パターン
- EMAモデルとの**正解一致**

を計算する。

### Step 3：予測性能を評価

$$
\operatorname{AUROC}
\left(q_i,\mathbf 1[L_i^*\leq\tau]\right)
$$

に加え、

- Spearman相関
- calibration error
- architecture間転移
- $\epsilon$間転移
- seed間安定性
- 計算コスト

を評価する。

### Step 4：予測精度と訓練utilityを分ける

1. learnability predictionの精度
2. その指標を使ったAT性能
3. その指標を使ったARD性能

を別々に評価する。

proxyがunlearnable sampleを当てても、それを重み付けに使って性能が上がるとは限らない。

### Step 5：除外ではなく処理を比較

予測unlearnable sampleに対して、

- 除外
- AT lossを弱める
- $\epsilon_i$を下げる
- attack step数を減らす
- KD weightを下げる
- teacher temperatureを上げる
- clean lossだけ残す

を比較する。

### 期待される貢献

- AT全般に利用できるsample-wise robust learnability benchmarkを提供できる。
- 「現在難しい」と「将来も学習不能」を分離できる可能性がある。
- ARD、FastAT、curriculum、sample selectionへ応用できる。

### ImageNet拡張

運用時はサンプルごとの数個のscalarだけを保存するため、ImageNet-1Kでも実用的である。参照ラベル構築は高コストなため、ImageNet-100、class-balanced subset、単一reference modelによるproxyの順に縮約する。

---

# 6. 研究課題3：教師から何を蒸留すべきか

## 6.1 robust soft label

RSLADに代表されるrobust soft labelは、現在もARDの基盤である。

しかし、

$$
p_T(x)\approx p_S(x)
$$

でも、

$$
p_T(x+\delta)\not\approx p_S(x+\delta)
$$

となり得る。点 $x$ で出力が一致しても、その周辺での入力勾配、境界位置、曲率、特徴変化は一致しない。

> **robust soft labelは必要な基盤だが、それだけでは不十分である。**

## 6.2 KDIGA：入力勾配の蒸留

KDIGAはKnowledge Distillation with Input Gradient Alignmentの略である。

教師と学生の入力勾配を、

$$
\nabla_x\mathcal L_T(x,y)
\approx
\nabla_x\mathcal L_S(x,y)
$$

とする。

局所一次近似、

$$
\mathcal L(x+\delta)
\approx
\mathcal L(x)
+
\nabla_x\mathcal L(x)^\top\delta
$$

を考えると、教師と学生の入力勾配が近ければ、小さな摂動に対する局所応答も近づく。

入力勾配は、

- 局所的な損失増加方向
- 決定境界法線の近似
- 主な敵対的摂動方向

を含むため、追加情報として理論的根拠が強い。

ただし、

$$
\nabla_\theta
\left\|
\nabla_x\mathcal L_S-
\nabla_x\mathcal L_T
\right\|^2
$$

を最適化するには、パラメータ更新時に入力勾配をさらに微分する二階微分相当の処理が必要となり、計算・メモリコストが高い。

## 6.3 IGDM：間接的な入力勾配蒸留

IGDMはTaylor近似を利用し、教師の入力勾配方向へ移動した点の出力関係を合わせることで、直接二階微分を避けながら入力勾配情報を移す。

> **現状、入力勾配は有力だが、ImageNet規模ではKDIGAの直接整合よりIGDMのような間接整合の方が実用的である。**

## 6.4 決定境界までの距離

厳密な境界距離は、

$$
d_T(x)
=
\min_{\delta}\|\delta\|
\quad
\text{s.t. }T(x+\delta)\neq y
$$

である。しかし、深層ネットワークで各サンプルの最小摂動を正確に求めることは高コストである。

代理量として、

- logit margin
- softmax probability margin
- teacher–student slack
- attack成功までのstep数
- clean→adv経路上の中間点

が使われる。

DARWINのような経路ベース手法は、cleanと最終敵対点だけでなく、中間点を用いて境界幾何を間接的に蒸留する。

## 6.5 境界の法線方向

クラス対 $(y,k)$ のlogit差を、

$$
F_{y,k}(x)=z_y(x)-z_k(x)
$$

とすると、境界 $F_{y,k}(x)=0$ の局所法線は、

$$
\nabla_xF_{y,k}(x)
$$

である。

ただし、ImageNetの1000クラスでは多数のクラス対が存在し、どの境界法線を一致させるかが曖昧である。実装では、

- 正解と最大競合クラス
- 攻撃先クラス
- teacher top-$K$

へ限定する必要がある。

## 6.6 class margin・slack

クラス間marginを、

$$
m(x)=z_y(x)-\max_{k\neq y}z_k(x)
$$

とする。soft labelにも相対関係は含まれるが、正解と最大競合クラスのmarginを明示的に整合させることで、重要な境界を強く拘束できる。

ただし、点上のmarginが同じでも、

$$
\nabla_xm_T(x)
eq\nabla_xm_S(x)
$$

なら周辺挙動は異なる。

## 6.7 clean/adv間の特徴変化

教師と学生の特徴を $h_T,h_S$ とすると、

$$
\Delta h_T=h_T(x_{\mathrm{adv}})-h_T(x)
$$

$$
\Delta h_S=h_S(x_{\mathrm{adv}})-h_S(x)
$$

を整合させることで、攻撃によってどの特徴が維持・変化するかを移せる可能性がある。

一方、CNNとViT、異なる幅・層数では特徴座標が対応しない。高性能なadapterを使うと、学生ではなくadapterが教師特徴を再現する危険もある。

## 6.8 特徴分布全体の統計構造

STARSHIPは、sample-wiseなlogit・特徴一致だけでなく、

- feature covariance
- prediction-score Gram matrix
- clean/adv間のvariance gap

を教師と学生で整合する。

> **近年の重要な知見は、サンプル単位の一致だけでなく、集合レベルの特徴統計も頑健性転写に有益になり得ることである。**

ただし、batch size、クラス構成、projection層に依存し、ImageNet-1Kで常に最良とは確立していない。

## 6.9 局所曲率

二次近似は、

$$
f(x+\delta)
\approx
f(x)
+
\nabla_xf(x)^\top\delta
+
\frac12\delta^\top H_xf(x)\delta
$$

である。Hessian全体の蒸留は入力次元が大きく現実的でない。

現状は、

- 中間敵対点
- 有限差分
- 複数摂動点でのlogit consistency

によって間接的に二次挙動を拘束する方が現実的である。

## 6.10 現状の総合判断

現時点で最も妥当な階層は、

$$
\boxed{
\text{robust soft label}
+
\text{局所幾何または敵対的経路}
+
\text{必要に応じて分布レベル統計}
}
$$

である。

例として、

$$
\text{RSLAD}
+
\text{IGDM}
+
\text{STARSHIP型統計}
$$

は理にかなう。しかし、損失項を単純に全部加えると勾配競合、容量不足、ハイパーパラメータ増加、計算量増加が生じる。

**同一教師・学生・攻撃・訓練FLOPsで、各知識の寄与を公平比較した研究が不足している。**

---

## 独立提案3：Directional Margin-Response Distillation（DMRD）

### 目的

full gradient/Hessian matchingを避けながら、学生が実際に弱い攻撃方向上の局所境界応答を蒸留する。

### 仮説

学生攻撃方向 $\delta_S$ 上で教師と学生のmargin軌跡を合わせれば、

- 境界までの近さ
- 局所一次応答
- 境界法線方向への感度
- 一方向の局所曲率

を低コストに部分的に転写できる。

### 使用点

$$
x_0=x,
\qquad
x_{1/2}=x+\frac12\delta_S,
\qquad
x_1=x+\delta_S
$$

とする。追加PGDは生成しない。

### 競合クラスの固定

教師と学生で最大競合クラスが頻繁に入れ替わる問題を避けるため、教師clean時の最大競合クラス $k^*$ またはteacher top-$K$を固定する。

$$
m_M(x;k^*)=z^M_y(x)-z^M_{k^*}(x)
$$

### 損失

点上のmargin整合：

$$
\mathcal L_{\mathrm{point}}
=
\sum_{a\in\{0,1/2,1\}}
\left|
\tilde m_S(x_a)-\tilde m_T(x_a)
\right|
$$

一次応答整合：

$$
r_M
=
\frac{m_M(x_1)-m_M(x_0)}{\|\delta_S\|+\varepsilon}
$$

$$
\mathcal L_{\mathrm{slope}}=|r_S-r_T|
$$

有限差分曲率：

$$
c_M=m_M(x_1)-2m_M(x_{1/2})+m_M(x_0)
$$

$$
\mathcal L_{\mathrm{curve}}=|c_S-c_T|
$$

全体：

$$
\mathcal L
=
\mathcal L_{\mathrm{AT/RSLAD}}
+
\lambda_r\mathcal L_{\mathrm{slope}}
+
\lambda_c\mathcal L_{\mathrm{curve}}
$$

### 検証

- RSLAD
- RSLAD + KDIGA
- RSLAD + IGDM
- DARWIN
- STARSHIP
- DMRD

を、追加forward数または総FLOPsを揃えて比較する。

### ImageNet拡張

最初はendpointのslopeだけを用いれば、中間forwardすら不要にできる。効果が確認できた場合のみmidpointと曲率を追加する。追加attackがないためImageNetへ拡張しやすい。

### リスク

- 学生攻撃という一方向しか見ない。
- 全Jacobianや他クラス境界を表さない。
- 教師margin軌跡自体が学生に実現不能な場合がある。

---

# 7. 研究課題4：有益で共適応しない教師・攻撃機構

## 7.1 既存方式のトレードオフ

### 固定robust teacher

- 独立性が高く、学生の誤りへ追従しない。
- 現在学生の弱点やstudent-crafted attackへ適応しにくい。

### PeerAiD

- 事前学習済みrobust teacherを不要にする。
- 学生に対する攻撃をpeerにも学習させる。
- peer専用の別PGDを作らない。
- 学生が失敗する入力へ専門化した教師をオンラインで作る。

一方で、peerはstudent-crafted attackには強くても、自身へのwhite-box attackにはほとんど頑健でない場合がある。これは目的上必ずしも失敗ではないが、現在学生への極端な専門化・共適応を一般に防ぐ設計ではない。

### SAAD

固定教師の出力を更新するのではなく、学生攻撃上で有益と考えられるサンプルの蒸留weightを調整する。

### Ensemble Adversarial Training

外部モデル由来の攻撃を加えることで、自己生成攻撃だけへの過適合を緩和する。

## 7.2 共適応している兆候

- tutorはcurrent-student attackだけに強い。
- historical/external attackで精度が急落する。
- tutorとstudentが同じサンプルで同時に失敗する。
- 未知攻撃上のteacher conditional utilityが低い。
- peerとstudentの誤り相関が学習後半に増加する。

## 7.3 相互模倣の問題

studentのsoft labelをpeerへ渡し、peerのsoft labelをstudentへ戻すと、

$$
\text{学生の一時的誤り}
\rightarrow
\text{peer}
\rightarrow
\text{学生}
$$

という閉ループができる可能性がある。

相互学習が常に有害という意味ではないが、**学生の誤ったクラス関係がpeerへ入り、次の教師信号として戻る経路**は明示的に検証すべきである。

## 7.4 時間遅延・EMAが効く可能性

時間遅延そのものが必ず安定化するわけではない。安定化の中心は、

- 同一step内の即時フィードバックを切る
- EMAで短期変動を低域通過する
- peerとstudentの更新時間尺度を分離する

ことである。

$$
\bar\phi_t
=
\mu\bar\phi_{t-1}
+
(1-\mu)\phi_t
$$

とすると、現在stepのpeer変動はEMA教師へ $1-\mu$ しか反映されない。偶発的な予測変動を学生へ直ちに戻しにくくなる。

ただし、誤りが長期間継続すればEMAも追従する。遅延しすぎると、教師が現在学生の弱点とずれる。

## 7.5 PeerAiDの計算コスト

PeerAiDのCIFAR-100報告では、蒸留段階がRSLADの26.21時間に対して30.50時間で、約16%増である。ただし、これは**RSLADの蒸留段階に対する比較**であり、学生単体ATに対する20%増ではない。

また、ImageNet-1Kでも同じ比率になる保証はない。攻撃step数が少ないほどpeer forward/backwardの相対比率は大きくなる。

---

## 独立提案4：軽量・非同期Peer ARD（Asynchronous One-Way Peer ARD; AOPA）

### 概要

学生 $S$、学生より小さいpeer $P$、peerのEMAモデル $\bar P$ を同時学習する。事前学習済みrobust teacherは使わない。

### 学習手順

1. 学生 $S$ に対して敵対的サンプル $x_{\mathrm{adv}}^S$ を1回だけ生成する。
2. peer $P$ は学生攻撃をground truthで分類するよう学習する。
3. 学生は現在peerではなく、EMA peer $\bar P$ から一方向に学ぶ。
4. student soft labelをpeerへ戻さない。
5. peerはwarm-up後、4–8 iterationに1回だけ更新する。

### peer損失

$$
\mathcal L_P
=
\mathrm{CE}(P(x_{\mathrm{adv}}^S),y)
+
\beta
D_{\mathrm{KL}}
\left(
\bar P(x)
\|P(x_{\mathrm{adv}}^S)
\right)
$$

学生のsoft labelをpeerへ渡さないため、学生誤りの逆流を抑える。

### student損失

$$
\mathcal L_S
=
\mathcal L_{\mathrm{AT}}
+
\lambda_i
D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}(\bar P(x_{\mathrm{adv}}^S))
\|S(x_{\mathrm{adv}}^S)
\right)
$$

### なぜ効くと考えるか

#### 1. 学生が失敗する場所を直接peerが学ぶ

固定教師は現在学生の弱点へ適応しない。一方peerは $x_{\mathrm{adv}}^S$ を正解するよう学ぶため、学生がまさに失敗する局所領域で教師信号を作れる。

#### 2. 誤り循環を弱める

student→peer soft labelを削除し、peerはground truthと自身のEMA整合だけで学ぶ。学生の誤った非正解クラス構造がpeerを介して戻る経路を切る。

#### 3. EMA peerが短期変動を抑える

学生は瞬間的なpeerではなく、複数時点を平均したteacher targetを受け取る。

#### 4. 疎更新がコストと共適応を同時に抑える

peer backwardを間引くことで、計算コストを下げるだけでなく、peerが現在学生へ過度に追従する速度を抑える。

#### 5. ImageNetでも攻撃を一つに保つ

peer専用PGDを作らず、student attackを再利用する。

### 共適応の評価

現在学生攻撃上のconditional utility：

$$
U_{\mathrm{current}}
=
\Pr[
\bar P(x_{\mathrm{adv}}^{S_t})=y
\mid
S_t(x_{\mathrm{adv}}^{S_t})\neq y
]
$$

未知attack source $E$ 上のutility：

$$
U_{\mathrm{cross}}
=
\Pr[
\bar P(x_{\mathrm{adv}}^{E})=y
\mid
S_t(x_{\mathrm{adv}}^{E})\neq y
]
$$

共適応gap：

$$
G_{\mathrm{coadapt}}
=
U_{\mathrm{current}}-U_{\mathrm{cross}}
$$

を測る。

### ImageNet設計

- peerを学生の0.4–0.7倍FLOPsにする。
- peer backwardを4–8 stepに1回へ間引く。
- EMA peerのforwardはno-gradで行う。
- 低解像度・弱攻撃の形成段階と、224解像度・強攻撃の短いfine-tuning段階を分ける。
- 実用設定では通常ImageNet事前学習を全baselineで共通に使う。

追加コスト10–20%は**目標値**であり、ImageNet上での実測が必要である。

---

# 8. 研究課題5：教師信号の信頼性をどう判断するか

## 8.1 教師モデル全体の選択とsample-wise信頼性を分ける

- **教師選択:** どの教師モデルを使用するかというglobal問題。
- **教師信号信頼性:** 現在のサンプル・学生checkpointで、その教師出力をどの程度使うかというlocal問題。

強い教師でも、個々のサンプルでは誤分類・過信・無反応を起こすため、両者を分ける必要がある。

## 8.2 信頼性の軸

### 1. 教師が正解しているか

$$
c_i
=
\mathbf 1[
\arg\max T(x_{i,\mathrm{adv}}^S)=y_i
]
$$

教師が誤分類していれば、そのsoft labelはground-truth CEと逆向きの勾配を与え得る。

### 2. 教師が過信していないか

特に学生が長期間学習できないサンプルで、

$$
p_T(y\mid x_{\mathrm{adv}}^S)\approx1
$$

なら、学生へ実現不能な目標を強制する可能性がある。

ただし、容易なサンプルで教師が正しく低entropyであることは問題ではない。

> **高entropyが望ましいのは、特に学生が学習困難な入力で、教師が正解を維持している場合である。**

### 3. 学生攻撃に教師が反応しているか

学生が大きく崩れているのに教師がclean時と同一の極端な分布を返す場合、教師は学生の困難さを反映していない可能性がある。

### 4. 学生が誤る場所で教師が正解するか

$$
U_{\mathrm{cond}}
=
\Pr[
T(x_{\mathrm{adv}})=y
\mid
S(x_{\mathrm{adv}})\neq y
]
$$

全体accuracyより、学生の失敗を補完できるかが重要である。

### 5. 異なる攻撃でも安定して正しいか

現在学生への単一PGDで正しいだけでは、たまたま攻撃が教師へ転移しなかった可能性がある。

- 異なるrestart
- CE/CW/DLR loss
- 過去学生attack
- 外部architecture attack
- targeted attack

でも正しいかを定期的に評価する。

## 8.3 非転移性と反応性の見かけ上の矛盾

望ましい状態は、

$$
S(x_{\mathrm{adv}}^S)\neq y,
\qquad
T(x_{\mathrm{adv}}^S)=y
$$

かつ、

$$
T(x_{\mathrm{adv}}^S)\neq T(x)
$$

である。

例：

$$
T(x)=(0.95,0.04,0.01)
$$

から、

$$
T(x_{\mathrm{adv}}^S)=(0.60,0.30,0.10)
$$

へ変化するが正解クラスは維持する。

> **「誤分類は転移しないが、難しさは転移する」状態が理想である。**

## 8.4 正解条件付きentropy

$$
h_i^+
=
\mathbf 1[T(x_{\mathrm{adv}}^S)=y]
\frac{H(T(x_{\mathrm{adv}}^S))}{\log C}
$$

とすれば、誤分類時の高entropyを除外できる。

ただし、正解条件付きentropyも、どの競合クラス方向へ変化したかまでは表さない。

## 8.5 clean–adversarial変化量

$$
D_{\mathrm{JS}}
\left(T(x),T(x_{\mathrm{adv}}^S)\right)
$$

またはmargin低下量、

$$
\Delta m_T
=
m_T(x)-m_T(x_{\mathrm{adv}}^S)
$$

を用いる。

- 変化0：学生の困難さへ無反応の可能性。
- 適度な変化：難しさを認識しつつ正解維持。
- 極端な変化：教師自身が崩壊している可能性。

したがって、最大化するのではなく、正解性・safety marginと併せて適度な範囲を評価する。

## 8.6 正解クラスを除いたentropy

非正解クラスだけを再正規化して、

$$
q_k(x)=\frac{p_k(x)}{1-p_y(x)},\qquad k\neq y
$$

とする。

$$
H_{\mathrm{wrong}}(q)
=-\sum_{k\neq y}q_k\log q_k
$$

は、非正解確率が一つの競合クラスへ集中しているか、複数へ分散しているかを測る。

しかし、

- どのクラスが増えたかは分からない。
- 猫0.75・狼0.25と猫0.25・狼0.75は同じentropyになる。
- 高entropyが意味的に有益な曖昧さか、単なる無知かを区別できない。

したがって、教師信頼性には、

$$
\boxed{
\text{教師の正解維持}
+
\text{margin低下量}
+
D_{\mathrm{JS}}(q_{\mathrm{clean}},q_{\mathrm{adv}})
}
$$

の方が直接的である。

DKDは正解クラス関連知識と非正解クラス間知識を分離する考え方を示しているが、**$H_{\mathrm{wrong}}$ がARD教師信頼性の最良指標と確立されたわけではない。**

---

## 独立提案5：Correctness-Preserving Difficulty-Calibrated Gate（CPDG）

### 目的

entropyまたはTAS単独ではなく、教師が正解を維持しながら学生難易度に適切に反応しているかをsample-wiseに評価する。

### 1. 正解・safety gate

$$
c_i
=
\mathbf 1[m_T(x_{i,\mathrm{adv}}^S)>m_{\min}]
$$

誤分類または境界ぎりぎりならKDを停止し、ground-truth ATへ戻す。

### 2. 学生難易度

既存学生攻撃から、adversarial correctness EMAを更新する。

$$
s_i(t)
=
\rho s_i(t-1)
+
(1-\rho)
\mathbf 1[S_t(x_{i,\mathrm{adv}})=y_i]
$$

$$
d_i(t)=1-s_i(t)
$$

とする。

### 3. 教師応答量

$$
r_i
=
\frac{m_T(x_i)-m_T(x_{i,\mathrm{adv}}^S)}{|m_T(x_i)|+\varepsilon}
$$

またはJS divergenceを使う。

### 4. 難易度校正

学生が容易なら教師は高確信度のままでもよい。学生が困難なら教師に一定のmargin低下・分布変化を期待する。

$$
r_i^*=g(d_i)
$$

とし、

$$
w_i
=
c_i
\exp
\left(
-\frac{|r_i-r_i^*|}{\tau_r}
\right)
$$

でKD weightを決める。

### 5. 学生損失

$$
\mathcal L_i
=
\mathcal L_{\mathrm{AT},i}
+
\lambda w_i
D_{\mathrm{KL}}
\left(
p_T(x_{i,\mathrm{adv}}^S)
\|p_S(x_{i,\mathrm{adv}}^S)
\right)
$$

### 比較対象

- 一様KD
- correctness gate
- entropy weighting
- IAD
- SAAD
- raw TAS
- correct-TAS
- CPDG

### 独立性

この提案は固定教師ARDだけでも検証でき、軽量peerを必要としない。学生learnability scoreには単純なcorrectness EMAを使い、RLPBの高度な予測器を前提としない。

### ImageNet拡張

通常ARDで教師clean/adv出力を既に得ていれば、追加attackは不要である。サンプルごとの履歴はscalarで保存できる。

---

# 9. 研究課題6：robust overfittingをどう監視・抑制するか

## 9.1 直接検出する指標

### best checkpointとlast checkpointの差

$$
D_{\mathrm{RO}}
=
\max_t A_{\mathrm{rob,val}}(t)
-
A_{\mathrm{rob,val}}(T)
$$

これは最も直接的な事後指標である。

### robust train–validation gap

$$
G_t
=
A_{\mathrm{rob,train\text{-}eval}}(t)
-
A_{\mathrm{rob,val}}(t)
$$

trainとvalidationは同じ評価攻撃で測る必要がある。訓練中に生成した攻撃上のtrain accuracyをそのまま使うと、inner maximizationへの適合度を測るだけになる可能性がある。

## 9.2 原因を診断する指標

### サンプル別forgetting

- 一度学習した後に忘れられたサンプル
- 最初から一度も学習できないサンプル
- 最後まで安定して学習できるサンプル

を分ける。

peak checkpoint $t^*$ から現在時点へのforgotten rateを、

$$
R_{\mathrm{forget}}(t)
=
\Pr[
S_{t^*}(x_{\mathrm{adv}})=y
\land
S_t(x_{\mathrm{adv}})\neq y
]
$$

とする。

### teacher entropy

全サンプル平均ではなく、proxy unlearnable set上で、

- teacher/peer accuracy
- entropy
- true-class probability
- margin

を同時に見る。

### 蒸留損失の時間変化

$$
L_{\mathrm{KD,train}}\downarrow
$$

なのに、

$$
A_{\mathrm{rob,val}}\downarrow
$$

なら、教師模倣が進んでもrobust generalizationに寄与していない。

特に、

$$
L_{\mathrm{KD}}^{U}\downarrow
\quad\text{かつ}\quad
G_t\uparrow
$$

なら、unlearnableサンプル上の教師信号を記憶している可能性がある。

### clean/adv特徴分散

STARSHIPは、clean特徴とadversarial特徴のvariance gapが大きいほどrobust performanceが低い傾向を示している。

$$
G_{\mathrm{feat}}
=
\left\|
\Sigma_{\mathrm{adv}}-
\Sigma_{\mathrm{clean}}
\right\|_F
$$

ただし、これはrobust overfittingの標準的オンライン検出指標ではなく、表現上の原因診断である。

## 9.3 best checkpointを保存すれば十分か

独立validation setでrobust accuracyを定期評価し、best checkpointを保存すれば、lastより良い状態へ戻せる。実務上は必ず行うべきである。

しかし、early stoppingで解決できるのは主にpeak-to-last degradationである。

例えば、

- robust train accuracy：90%
- best validation robust accuracy：55%
- last validation robust accuracy：48%

なら、48%から55%へ戻せるが、90%と55%のgapは残る。

robust overfittingの問題点は、

- best時点でもrobust generalization gapが大きい。
- best epochを知るためにvalidation attackが必要。
- 攻撃lossや$\epsilon$によってbest epochが異なり得る。
- 後半の重いAT計算が無駄になる。
- seed・schedule・checkpoint選択への依存が大きい。
- 学習目的とrobust generalizationの不一致を解決していない。
- ARDでは、どの教師信号・サンプルが過学習を促したか分からない。

ことである。

> **robust overfitting研究の目的は、best checkpointを保存するだけでなく、best性能を高め、bestからlastまでの低下を減らし、train–validation gapも縮めることである。**

---

## 独立提案6：Robust-Overfitting Early Warning and Intervention（ROEWI）

### 目的

大きなvalidation低下が起きた後にbestへ戻るだけでなく、robust overfittingの前兆を検出し、教師信号またはpeer更新へ介入する。

### 監視用probe set

ImageNetでは全validation setへ毎epoch強攻撃を行うと高コストである。クラス均衡の固定probe subsetを作り、短いPGDで頻繁に評価する。候補best checkpointだけを強いPGD/AutoAttackで再評価する。

### 警報指標

- robust validation accuracyの負の傾き
- robust train–validation gapの拡大
- robust forgetting rateの上昇
- proxy unlearnable set上のKD loss低下
- teacher/peer低entropy・高確信度の増加

### 警報条件例

- validation robustnessが連続2–3回低下
- train robustnessまたはtraining objectiveは改善
- train–validation gapが拡大
- unlearnable集合上のKD lossが低下

が同時に生じた場合を、教師信号記憶型overfittingと判断する。

### 介入を独立比較

1. low-learnabilityかつ低entropy信号のKD weightを下げる。
2. 対象サンプルのteacher temperatureを上げる。
3. online peerを一時freezeまたは疎更新する。
4. student EMA/SWAを強める。
5. early stoppingする。

### 評価

- best robust accuracy
- last robust accuracy
- peak-to-last degradation
- robust train–validation gap
- 警報のlead time
- false alarm率
- 追加validationコスト
- clean accuracy

### ImageNet拡張

固定probe subset、疎な強評価、候補checkpointの二段階評価で計算量を制御する。test setは最終評価のみに使う。

---

# 10. 研究課題7：クラスごとの頑健性格差をどう縮小するか

## 10.1 現在のrobust fairness手法

- クラス別損失重み
- class-DRO / CVaR
- クラス別attack strength
- クラス別正則化係数
- クラス別soft-label temperature
- サンプル単位のhard-example weighting

などがある。

クラス単位の重み付けは簡単だが、hard class内のunlearnable/noisy sampleまで強く学習する危険がある。サンプル単位の方法は柔軟だが、worst-class robustnessを直接最適化しない。

## 10.2 ImageNetでの課題

1000クラス分のweightやEMAを保存する計算コスト自体は小さい。問題は、クラス別robust accuracyやfeature statisticsの推定分散である。

- validation画像数が限られる。
- checkpoint間の変動が大きい。
- 一部の外れクラスが学習全体を支配する。
- raw worst classは極端に不安定である。

したがって、1000クラスを完全独立に最適化するより、EMA、階層的縮約、bottom-tail平均を使う方が安定する。

## 10.3 通常のImageNet認識から得られる知見

balanced ImageNetでもクラスごとの精度格差が大きい。通常画像認識の研究では、単純なclassifier biasより、hard classの特徴分布が広く、他クラスとの重なりが大きい表現問題が指摘されている。

MR²（Margin Regularization）は、

- class-wise feature spreadを測る。
- spreadが大きいhard classへ大きなlogit marginを与える。
- クラス内特徴をcompactにするrepresentation margin lossを加える。

ことで、ImageNetのhard class性能を改善する。

> **通常認識でImageNetまで検証されたclass-wise representation methodを、adversarial feature geometryへ修正する方向は有望である。**

## 10.4 robust版で追加すべき要素

通常MR²のclean feature spreadだけでは、敵対的設定特有の、

- adversarial feature spread
- clean→adv feature drift
- adversarial margin低下
- 教師のクラス別誤り

を扱えない。

---

## 独立提案7：Adv-MR²-ARD

### 1. adversarial feature spread

学生の中間特徴を $h_S$ とし、クラス $c$ について、

$$
V_c^{\mathrm{adv}}
=
\mathbb E_{y=c}
\left[
\|h_S(x_{\mathrm{adv}})-\mu_c^{\mathrm{adv}}\|^2
\right]
$$

をEMAで記録する。

### 2. clean–adversarial feature drift

$$
D_c
=
\mathbb E_{y=c}
\left[
\|h_S(x_{\mathrm{adv}})-h_S(x)\|^2
\right]
$$

を測る。

- $V_c^{\mathrm{adv}}$ が大きい：敵対的特徴がクラス内で広く散る。
- $D_c$ が大きい：攻撃によってclean表現から大きく移動する。

### 3. クラス別robust margin

$$
r_c
=
\operatorname{Norm}
\left(
\alpha V_c^{\mathrm{adv}}
+
\beta D_c
-
\gamma M_c^{\mathrm{adv}}
\right)
$$

$$
\Gamma_c
=
\Gamma_0(1+\lambda r_c)
$$

とし、hard classへ大きなmarginを課す。

### 4. adversarial特徴をcompactにする

$$
\mathcal L_{\mathrm{compact}}
=
\frac1B
\sum_i
\|h_S(x_{i,\mathrm{adv}})-\mu_{y_i}^{\mathrm{adv}}\|^2
$$

を加える。

### 5. ARDでは教師信号もクラス別に調整

教師またはpeerが、そのクラスの学生攻撃を正しく分類する場合に限り、KD weightまたはtemperatureを調整する。

$$
\lambda_{\mathrm{KD},i}
=
\lambda_0
\mathbf 1[T(x_{i,\mathrm{adv}})=y_i]
q_{g(y_i)}
$$

hard classだから無条件にKDを強めると、教師のhard-class誤りを増幅する可能性がある。

### 6. 1000クラスでの階層的縮約

$$
\tilde V_c
=
\frac{n_c}{n_c+\kappa}V_c
+
\frac{\kappa}{n_c+\kappa}V_{g(c)}
$$

とし、クラス統計を、

- WordNet上位カテゴリ
- confusion pattern
- robust difficulty bin

のgroup平均へ縮約する。

### 評価指標

raw worst classだけでなく、

- mean class robust accuracy
- bottom 10% classes平均
- 5th percentile class accuracy
- class-wise標準偏差
- worst-group accuracy
- overall robust accuracy
- clean accuracy

を報告する。

### 検証順序

1. robust hard classで $V_c^{\mathrm{adv}}$、$D_c$ が大きいか。
2. class-wise robust accuracyとの相関があるか。
3. clean MR²よりadversarial統計が必要か。
4. 単純class reweightingより平均性能を保てるか。
5. AT-onlyで成立するか。
6. ARDへ拡張したとき教師バイアスを抑えられるか。

### ImageNet拡張

追加PGDは不要で、既存forwardの特徴を利用する。1000クラスのcentroid・variance保存は現実的である。階層的縮約で統計ノイズを抑える。

---

# 11. 各独立提案の比較

| 課題 | 提案 | robust teacher | 主な追加コスト | ImageNet適性 | 主な新規性 | 最大のリスク |
|---|---|---:|---:|---:|---|---|
| 良い教師 | SCTS | 必要 | 候補教師forward・短期pilot | 高 | 学生互換性による教師選択 | reference student依存 |
| 学習可能性 | RLPB | 不要でも可 | gold label構築は高、運用は低 | 中～高 | 将来learnabilityの早期予測 | 条件依存性が大きい |
| 蒸留対象 | DMRD | 必要 | 0–1個の追加中間forward | 高 | margin軌跡で局所幾何を近似 | 一方向しか見ない |
| 共適応 | AOPA | 不要 | peer forward/backward | 高 | teacher-free・一方向・EMA・疎更新 | 適応性を失う可能性 |
| 信頼性 | CPDG | 固定教師/peer | ほぼ低コスト | 非常に高 | 正解維持と難易度応答の統合 | 応答目標の設計 |
| overfitting | ROEWI | 不要でも可 | probe validation | 高 | 前兆から適応介入 | probe selection bias |
| クラス格差 | Adv-MR²-ARD | AT-only可 | feature統計、追加attackなし | 非常に高 | 通常ImageNet手法をrobust geometryへ拡張 | spread仮説が成立しない可能性 |

## 11.1 分析研究として強いもの

- **RLPB:** AT全般の未解決問題へ広く寄与する。
- **SCTS:** 教師選択問題を教師単体性能から学生互換性へ再定義する。

## 11.2 新しいARDアルゴリズムとして強いもの

- **AOPA:** teacher-free、ImageNet、共適応、計算量を同時に扱う。
- **DMRD:** soft labelと高コストgradient matchingの中間を狙う。
- **CPDG:** 低コストで既存ARDへ追加しやすい。

## 11.3 ImageNetで特に適するもの

- **Adv-MR²-ARD:** 1000クラスの格差へ直接対応する。
- **CPDG:** 追加attackが不要でスケールしやすい。
- **AOPA:** teacher事前学習を不要にできるがpeerコストの実測が必要。

---

# 12. 共通評価プロトコル

## 12.1 データセット段階

1. CIFAR-100
2. Tiny-ImageNet
3. ImageNet-100またはclass-balanced ImageNet subset
4. ImageNet-1K

ImageNet対応を主張する場合、ImageNet-100だけで一般化を結論づけず、可能ならImageNet-1Kの明確なbudget-controlled settingを含める。

## 12.2 学生architecture

異なる帰納バイアスを持つ最低2種類を使う。

- CNN：ResNet-18、MobileNet、ConvNeXt-Tinyなど
- Transformer：DeiT-Tiny、Tiny ViT系など

## 12.3 baseline

- PGD-AT
- TRADES
- RSLAD
- IAD
- PeerAiD
- 問題に応じてSAAD、IGDM、STARSHIP、ABSLD等

## 12.4 攻撃評価

頻繁な監視には固定PGDを使い、最終候補には、

- AutoAttack
- 多restart PGD
- CE/CW/DLR loss
- targeted attack
- transfer attack

を用いる。

peer・教師・攻撃sourceを扱う研究では、**訓練に直接使っていないattack source**を必ず含める。

## 12.5 checkpoint

- test setでcheckpointを選ばない。
- validation bestを保存する。
- bestとlastの両方を報告する。
- peak-to-last degradationを報告する。

## 12.6 計算量

次を分ける。

- robust teacher事前学習コスト
- 学生蒸留コスト
- peer更新コスト
- attack生成コスト
- validation attackコスト
- GPU時間
- FLOPs
- 最大GPUメモリ
- 推論コスト

## 12.7 主要指標

### 全提案共通

- clean accuracy
- PGD accuracy
- AutoAttack accuracy
- best/last robust accuracy
- seed平均・標準偏差
- training FLOPs/GPU時間

### 教師・peer研究

- teacher accuracy on student attacks
- conditional utility
- correct-TAS / wrong-TAS
- cross-attack utility
- co-adaptation gap

### learnability研究

- AUROC/AUPRC
- Spearman相関
- calibration error
- architecture/$\epsilon$/seed転移

### robust overfitting研究

- peak-to-last degradation
- robust train–validation gap
- robust forgetting rate

### class disparity研究

- mean class robust accuracy
- bottom 10%平均
- 5th percentile
- class-wise標準偏差
- worst-group accuracy

---

# 13. 主要参考文献

1. Goldblum, M., Fowl, L., Feizi, S., Goldstein, T. **Adversarially Robust Distillation.** AAAI 2020. [arXiv:1905.09747](https://arxiv.org/abs/1905.09747)
2. Zi, B., Zhao, S., Ma, X., Jiang, Y.-G. **Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better.** ICCV 2021. [arXiv:2108.07969](https://arxiv.org/abs/2108.07969)
3. Zhu, J. et al. **Reliable Adversarial Distillation with Unreliable Teachers.** ICLR 2022. [arXiv:2106.04928](https://arxiv.org/abs/2106.04928)
4. Maroto, J. et al. **On the Benefits of Knowledge Distillation for Adversarial Robustness.** 2022. [arXiv:2203.07159](https://arxiv.org/abs/2203.07159)
5. Shao, R., Yi, J., Chen, P.-Y., Hsieh, C.-J. **How and When Adversarial Robustness Transfers in Knowledge Distillation?** 2021. [arXiv:2110.12072](https://arxiv.org/abs/2110.12072)
6. Lee, H., Cho, S., Kim, C. **Indirect Gradient Matching for Adversarial Robust Distillation.** ICLR 2025. [arXiv:2312.03286](https://arxiv.org/abs/2312.03286)
7. Jung, J., Jang, H., Song, J., Lee, J. **PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor.** CVPR 2024. [arXiv:2403.06668](https://arxiv.org/abs/2403.06668)
8. Lee, H., Chung, H. W. **Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation.** TMLR 2026. [arXiv:2512.10275](https://arxiv.org/abs/2512.10275)
9. Lee, H., Chung, H. W. **Toward Understanding Adversarial Distillation: Why Robust Teachers Fail.** 2026. [arXiv:2605.21999](https://arxiv.org/abs/2605.21999)
10. Dong, C., Liu, L., Shang, J. **Data Quality Matters for Adversarial Training: An Empirical Study.** 2021. [arXiv:2102.07437](https://arxiv.org/abs/2102.07437)
11. Rice, L., Wong, E., Kolter, J. Z. **Overfitting in Adversarially Robust Deep Learning.** ICML 2020. [PMLR 119](https://proceedings.mlr.press/v119/rice20a.html)
12. Zhou, D. et al. **STARSHIP: Adversarially Robust Distillation via Reducing Student-Teacher Variance Gap.** ECCV 2024. [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00499.pdf)
13. Dong, J. et al. **Robust Distillation via Untargeted and Targeted Intermediate Adversarial Samples.** CVPR 2024. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_Robust_Distillation_via_Untargeted_and_Targeted_Intermediate_Adversarial_Samples_CVPR_2024_paper.html)
14. Yin, F. et al. **Adversarial Distillation Based on Slack Matching and Attribution Region Alignment.** CVPR 2024.
15. Zhao, B. et al. **Decoupled Knowledge Distillation.** CVPR 2022. [arXiv:2203.08679](https://arxiv.org/abs/2203.08679)
16. Xu, H. et al. **Anti-Bias Soft Label Distillation for Robust Fairness.** 2023. [arXiv:2312.05508](https://arxiv.org/abs/2312.05508)
17. Tian, Q. et al. **Class-wise Calibrated Fair Adversarial Training.** 2023. [arXiv:2303.14460](https://arxiv.org/abs/2303.14460)
18. Zhu, Z. et al. **Reducing Class-wise Performance Disparity via Margin Regularization.** 2026. [arXiv:2602.00205](https://arxiv.org/abs/2602.00205)
19. Wang, Z. et al. **Revisiting Adversarial Training at Scale.** CVPR 2024. [arXiv:2401.04727](https://arxiv.org/abs/2401.04727)
20. Croce, F., Hein, M. **Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks.** ICML 2020. [arXiv:2003.01690](https://arxiv.org/abs/2003.01690)

---

# 14. 最終整理

これまでの議論から、特に重要な点は次である。

1. **ARDの本質は、強い教師を模倣することではなく、学生に実現可能で一般化に有益な頑健知識だけを転写することである。**
2. **教師単体のrobust accuracyは必要条件だが、良い教師の十分条件ではない。**
3. **TASは攻撃応答の転移を測るが、教師正解性や教師信号の有用性を保証しない。**
4. **Unlearnable-Entropyは学生の表現限界と教師過信の不整合を扱うが、教師誤分類とproxy集合の不安定性が残る。**
5. **robust learnabilityの最も有力な簡易proxyはadversarial correctness履歴だが、学習初期の将来予測は未確立である。**
6. **robust soft labelはARDの基盤だが、局所境界応答や特徴統計を十分に転写しない。**
7. **動的peerでは、学生への適応性と共適応防止を分けて設計・評価する必要がある。**
8. **理想的な教師信号は「誤分類は転移しないが、難しさは転移する」状態である。**
9. **best checkpoint保存は必要だが、robust generalization gap、計算浪費、教師信号への過適合は解決しない。**
10. **ImageNetのクラス格差には、通常認識で検証されたfeature-geometry手法をadversarial statisticsへ拡張する方向が有望である。**

各独立提案は、この10点のうち異なる未解決部分を対象とする。現段階では、提案を一つの大規模システムへ統合するより、**各仮説が単独で成立するかを明確なbaselineと公平な計算予算で検証すること**が重要である。
