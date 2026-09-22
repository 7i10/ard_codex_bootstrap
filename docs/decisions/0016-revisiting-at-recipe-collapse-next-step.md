---
id: 0016
status: decided
created: 2026-09-22
campaign: plan0102-revisiting-at-recipe-full50-v1（Hamster GPU0のhand-run、seed 0、50/50 epoch完走、source SHA ebad235）
question: revisiting_atレシピ移植は50 epoch完走したが崩壊した（最終 clean 28.6% / PGD-10 0.05%、Arm Aは54.6% / 30.6%）。原因は「壊滅的過学習（catastrophic overfitting）とラベルリーク」で、移植コミットにバグは見つからなかった。ここから、崩壊の原因を切り分けるのか、切り分けるならどの単一変数を、それとも移植を打ち切ってplain PGD-ATに戻るのか。
options:
  A: 何もGPUを使わない。観測性の修正（mode一致したrobust精度の記録とCOフラグ）とフォワードのみの検査を先にやる。GPU約0時間
  B: 単一変数の切り分けアームを1本、25 epochだけ走らせる。重いデータ拡張だけを外す。GPU約10時間
  C: 訓練時の攻撃ステップ数を3から上げた新しいcontractを立てて1本走らせる。GPU約40〜50時間
  D: 移植を打ち切る。plain PGD-AT（Arm A）を現時点の最良レシピとして確定し、GPUはWorkstream Bに回す。GPU約0時間
recommendation: A
chosen: A
---

# 0016 revisiting_atレシピ崩壊のあと、次に何をするか

## 記録の所在について

このキャンペーンには `docs/experiments/` の記録ファイルがありません。この
contract 用のaggregatorが存在しないためで、plan 0102のhand-run 7本すべてと同じ
扱いです。数値と判断の根拠は plan 0102 の Progress log
（`docs/plans/0102-clean-preserving-robustness-and-lightweight-ard.md`、
2026-09-22のエントリ、コミット `015bc8d`）にあります。evidence ledgerの行も
ありません。

## 何が起きたか

50 epochすべて完走し、プロセスは正常終了しました。中身は崩壊しています。
以下はすべて内部検証のみ（held-outスライス、PGD-10、n=1、seed 0）で、公式の
結果ではありません。

| | clean | PGD-10 |
|---|---:|---:|
| このアーム（epoch 49） | 28.6% | **0.05%** |
| このアームのEMA重み（epoch 49） | 32.6% | 0.07% |
| Arm A = plain PGD-AT（epoch 49、本planで3回再現済み） | 54.6% | 30.6% |

PGD-10の0.05%は、1000クラスの偶然当たる確率0.1%を下回っています。clean側も
26.0 pt負けています。つまり「clean精度とロバスト性のトレードオフ上の別の点」
ではなく、単なる崩壊です。

崩壊の始まりは epoch 20〜21 にはっきり出ます。そのepochで3つが同時に壊れます。

- 訓練時の`train_robust_accuracy`が`train_clean_accuracy`を**追い越す**
- `train_loss`が単調に下がり始める（5.09 → epoch 49で2.44）
- `val_pgd_accuracy`がゼロへ滑り落ち始める

学習率は設計どおりでした（epoch 0-9で1e-4→1e-3の線形warmup、その後cosineで
epoch 49に1.54e-6）。epoch 21の学習率は8.2e-4で、cosineの途中です。つまり
スケジュールの段差が原因ではありません。

## 原因は何で、何ではないか

読み取り専用の`bug-investigator`調査で、移植コミット（`2339a3d`、`ebad235`）に
由来するバグ候補はすべて否定されました。いずれもコード上の根拠つきです。

- label smoothingは内側の攻撃に届いていない（`src/ard/attacks/pgd.py:170-171`は
  平滑化なしのCEを呼ぶ）
- soft-targetの分岐は動いていない（`target_policy`も`iad_inspired`も無く、実際
  `train_teacher_*_forward_calls`は全epochで0.0）
- `warmup_cosine_multiplier`は記録された学習率の系列を厳密に再現する
- `_adamw_parameter_groups`はパラメータを1つも取りこぼしていない
- 重いデータ拡張はピクセル[0,1]空間で、攻撃の**前**にclean画像へ適用されている

決め手は保存済みのサンプル単位の記録です。`run-bundle/panels/panel-epoch-49.jsonl`
の24サンプルのうち、**clean画像で間違える15サンプルのうち10サンプルが、攻撃後には
正解ラベルぴったりに当たっています**（真値198→clean 490→攻撃後198、461→854→461、
315→40→315、910→587→910、943→939→943）。逆にclean正解の9サンプルのうち攻撃で
壊れたのは2つだけです。真のラベルでCEを上げて作った摂動が、そのラベルを書き込む
スタンプになり、ネットワークがそれを読み取ることを覚えてしまった、という状態です。
外側の目的関数は、頑健な特徴を学ぶ代わりにそのスタンプを読む方が安く下がるので、
そちらに落ちます。これが壊滅的過学習とラベルリークの教科書的な signature です。

なお、この移植は事前に「レシピ全体のテスト」と範囲を決めてあります
（2026-09-21、review finding P2-7）。ですから正直な結論は「このレシピ全体を
この設定でMobileNetV4-Conv-Smallに載せると崩壊する」であって、**どの材料が
原因かは特定できていません**。後からこれを「学習率が原因だと示した」と引用して
はいけません。疑わしい材料は3つあり、まだ切り分けられていません。

1. 内側の攻撃が3ステップであること（"fast-AT"に近く、壊滅的過学習の既知のリスク。
   Arm Aから変えていない値で、SGDの下では安全だった）
2. AdamW、weight decay 5e-2、しかもBatchNormのaffineは（正しく）weight decay
   対象外になったこと。3.77MパラメータのBN主体の生徒に対しては強い設定
3. 重いデータ拡張。BNの running statistics が追随しなければならない統計の変化

## ついでに分かった、今後に効く事実

`train_clean_accuracy`と`train_robust_accuracy`は**同じ条件で測られていません**。
前者はeval mode・optimizer step の**後**（`trainer.py:1605-1606`）、後者は
train mode・step の**前**（`trainer.py:1225`、BNはバッチ統計）です。この非対称は
本planより古く、bootstrapコミット`3e309dd`まで遡ります。

ふだんは無害です。対照の`plan0102-weight-ema-full50-v1`はepoch 49で
train clean 49.56% / train robust 27.92%と正しい大小関係のままで、train robustは
自身のval PGD（30.58%）より下にあります。つまりこの非対称は、崩壊が起きた後に
見出しの数字（70.9%）を膨らませるだけで、崩壊の原因ではありません。

ただし結果として、**`train_robust_accuracy`だけを見ていても壊滅的過学習は検出
できません**。今回それを11 GPU時間ぶん（epoch 21から49まで）払って学びました。
同じ値はADR実行時の`gap_adaptive_step`（`trainer.py:1797-1799`）にもmodeが
混ざったまま入っています。

## ノイズ下限について

本planがこれまで使ってきた参照値はCIFARの control 対 control のばらつき
0.16〜1.88 ptで、ImageNetのノイズ下限はまだ測っていません。ただし今回問題に
なっている差は clean 26 pt、PGD 30 ptで、どのノイズ下限よりも桁違いに大きい
です。ですから「崩壊したかどうか」を判定する用途では、ノイズ下限は制約に
なりません。逆に、下の選択肢のうち**数pt以下の差を読む目的のものは1本では
決着しません**。該当する場合は各選択肢に明記します。

## GPU時間の見積り根拠

このアーム自身の実測値から出します。`train_seconds`は50 epochを通して約1409秒/
epochで安定していました（Hamsterの4090 1枚、world size 1、global batch 128)。
したがって50 epochで約19.6 GPU時間、25 epochで約9.8 GPU時間です。内側の攻撃の
ステップ数を上げる選択肢だけは、1 step あたりforward+backwardが1回増えるため
別に見積ります。現在Hamsterは2枚とも空いています。

---

## 選択肢A: GPUを使わず、観測性の修正とフォワードのみの検査を先にやる

**GPU時間**: 約0時間（フォワードのみの検査が1 GPU分程度）

**やること**は2つです。

1. **mode一致したrobust精度を記録する**。すでにある
   `with _evaluation_mode(self.model), torch.no_grad():`のブロック
   （`trainer.py:1605-1606`）の中で、生成済みの`adversarial`テンソルに対する
   forwardを1回足し、`train_robust_accuracy_eval_mode`として出す。既存の
   `train_robust_accuracy`の定義と値は**一切変えない**（過去の記録がすべて
   その定義に依存しているため）。あわせて
   `train_robust_accuracy > train_clean_accuracy`という真偽値のフラグを出す。
   フラグは記録するだけで、何も中断させず、何も自動調整しない。攻撃・epsilon・
   ステップ数・step size・random start・checkpoint選択には触れない。純粋な
   観測性の追加です。
2. **保存済みcheckpointへのフォワードのみの検査**。`last.pt`を読み、訓練時と
   同じ1バッチに対して、同じテンソル上で5つの数字を出す。(a) eval modeの
   clean精度、(b) train modeのclean精度、(c) eval modeでの訓練用3-step攻撃後の
   精度、(d) train modeでの同じ攻撃後の精度、(e) 攻撃の`max_abs_delta`と
   `trace_step_losses`。訓練はしません。

**何が分かるか**: 70.9%という数字のうち、どれだけが「BNのmode差」で、どれだけが
「ラベルリーク」かが分かります。(b)≈(d)≈0.70なら前者が主因、(c)≈(d)≈0.70で
10-step攻撃がほぼ0なら純粋にステップ数由来の壊滅的過学習です。後者なら選択肢C
（ステップ数を上げる）に意味があり、前者ならCは無駄になります。さらに、次に
どのアームを走らせても、崩壊をepoch 21で検出して止められるようになります。

**事前登録する判定ルール**: (c)が(d)より20 pt以上低ければ「見出しの70.9%は
主としてmodeの非対称による見かけ」と判定し、`train_robust_accuracy`単体を
ロバスト性の指標として読むことを今後禁止する。(c)≈(d)かつ同じバッチへの
10-step攻撃が5%未満なら「ステップ数由来の壊滅的過学習」と判定し、選択肢Cの
事前確率を上げる。

**リスク**: `trainer.py`に触るので、本planの自分のルールによりGPUを使う前に
`scientific-reviewer`のレビューが必要です（今回は観測性のみでGPUを使わないため、
レビューは次のアームの前までに通ればよい）。`trace_step_losses`はdebug用の
フィールドで14個のidentityフィールドには含まれないため、これを有効にしても
contractは変わりません。値の差は大きいので、ノイズ下限は問題になりません。

## 選択肢B: 重いデータ拡張だけを外したアームを25 epoch走らせる

**GPU時間**: 約9.8時間（実測1409秒/epoch × 25。データ拡張を外すぶん、わずかに
速くなる可能性はあります）

**やること**: `imagenet_mobilenetv4_revisiting_at_recipe.yaml`から
`dataset.imagenet_heavy_augmentation`だけを`false`にした新しいconfigを作り、
25 epochだけ走らせる。他の材料（AdamW、学習率、weight decay、label smoothing、
weight-EMA、3-stepの攻撃）はすべて据え置き。攻撃のidentityは変えないので、
新しいprotocol idは不要です（`tracking.group`とconfig名のみ追加）。

**何が分かるか**: 3つの容疑者のうち1つが単独で外れます。崩壊は epoch 20-21 に
出るので、25 epochで判定がつきます。50 epoch走らせる必要はありません。

**事前登録する判定ルール**: epoch 25までに(1)`train_robust_accuracy`が
`train_clean_accuracy`を上回らず、かつ(2)`val_pgd_accuracy`がepoch 9の値
（13.59%）の半分を下回らなければ、「重いデータ拡張が崩壊の必要条件だった」と
判定する。どちらか一方でも破れば「重いデータ拡張は主因ではない」と判定し、
残る容疑者（AdamW/weight decay、3-stepの攻撃）に絞る。

**リスク**: 単一変数を外しても崩壊が残った場合、容疑者が2つ残るだけで、もう1本
必要になります（さらに約10 GPU時間）。また、これはあくまで「崩壊するか否か」の
判定であって、clean精度で Arm A に勝てるかどうかの判定ではありません。数pt
規模の優劣を読むには n=1・25 epochでは足りません。

## 選択肢C: 訓練時の攻撃ステップ数を上げた新しいcontractを立てる

**GPU時間**: 約40〜50時間（50 epoch。内側の攻撃を3→10 stepにすると1バッチ
あたりのforward/backwardが増えるため、実測1409秒/epochの2〜2.5倍を見込む）

**やること**: 訓練時の攻撃ステップ数は CLAUDE.md ルール6の保護対象です。ですから
これは既存configの編集ではなく、**新しいscientific contract**として、新しい
protocol idと新しいconfig、独立した plan エントリを立てる必要があります。
壊滅的過学習に対する文献上の標準的な対処です。

**何が分かるか**: 「3ステップという弱い内側攻撃が崩壊の原因か」が決まります。
これが当たれば、レシピ移植そのものは救える可能性があります。

**事前登録する判定ルール**: 選択肢Bと同じ2条件をepoch 25で適用する。さらに
50 epoch完走時に `val_pgd_accuracy` が Arm A の30.6%を下回らなければ「レシピは
ステップ数を上げれば機能する」と判定する。下回る場合、差が2 pt未満なら
「判定不能、seedを増やさないと読めない」とし、勝ったとは書かない。

**リスク**: 一番高価で、しかも選択肢Aのフォワード検査をやる前に走らせると、
外れたときに40〜50 GPU時間を捨てることになります。ステップ数を上げることは
攻撃を強める方向なので規則6には抵触しませんが、Arm Aとの比較は「訓練レシピの
比較」ではなくなり、Arm Aと同じ土俵ではなくなります。この点は新しいplanに
明記が必要です。

## 選択肢D: 移植を打ち切る

**GPU時間**: 約0時間

**やること**: revisiting_atレシピの移植を打ち切り、plain PGD-AT（Arm A）を
現時点で最良の訓練レシピとして確定する。HamsterのGPUはWorkstream B（軽量な
ImageNet規模のARD後継）か、既存checkpointへのAutoAttack（decision 0014が
まだ`pending`）に回す。

**何が分かるか**: 新しいことは何も分かりません。ただし、本planのWorkstream Aは
これまでフル50 epochのアームを4本走らせ（weight-EMA 2種、TRADES+weight-EMA、
sharp-temperature ADR）、**1本も plain PGD-AT の両軸を超えていません**。今回の
移植は超えなかっただけでなく崩壊しました。打ち切りは現実的な選択肢です。

**事前登録する判定ルール**: 判定するものがないため、ルールなし。これは
「これ以上この方向にGPUを使わない」という資源配分の判断です。

**リスク**: 崩壊の原因を記録しないまま閉じると、将来3-stepの攻撃と重い拡張を
組み合わせた別のアームで同じことを繰り返します。打ち切る場合でも、選択肢Aの
観測性修正（GPU 0時間）だけは入れておく価値があります。

---

## 推奨: A

選択肢Aを推します。理由は3つです。

第一に、GPUをほぼ使いません。そしてAの結果は、BとCのどちらに価値があるかを
直接決めます。(c)≈(d)でかつ10-step攻撃がほぼ0なら、Cは筋が良く40〜50 GPU時間を
払う意味があります。(b)≈(d)なら、Cは的外れで、その40〜50時間は捨てになります。
高い選択肢の前に、ほぼ無料で事前確率を動かせる検査があるなら、先にそれをやる
べきです。

第二に、観測性の修正は今回すでに11 GPU時間を無駄にしています。崩壊はepoch 21で
確定していたのに、`train_robust_accuracy`が上がって見えたせいで、epoch 49まで
走り切りました。フラグが入っていれば止められました。このplanは今後もアームを
走らせるので、同じ損失が繰り返されます。

第三に、`train_robust_accuracy`と`train_clean_accuracy`がmode不一致だという
事実は、この1回の実験より寿命が長い知見です。記録しないと、次のアームで必ず
再発見することになります。`docs/debugging/0030-*.md` に残すことも併せて
検討してください（本postrunでは範囲外としてまだ書いていません）。

**この推奨が変わる条件**: もし人間の判断として Workstream A 自体をもう畳むので
あれば、Aの第一の理由（BとCの事前確率を動かす）は消えるので、素直にDを選び、
そのうえでAの観測性修正だけを入れるのが良いです。逆に、レシピ移植をどうしても
救いたい強い理由があるなら、Aを飛ばしてCに行くこと自体は筋が通ります。ただし
その場合、外れたときに40〜50 GPU時間を失うことを承知のうえで、という条件つき
です。

なお選択肢Bは、Aのフォワード検査をやったあとでも選べます。順番を入れ替える
理由は見つかりませんでした。
